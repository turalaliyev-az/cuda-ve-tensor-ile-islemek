// v4l2_cuda_blur_async.cu
#include <iostream>
#include <vector>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include <sys/mman.h>
#include <cuda_runtime.h>
#include <signal.h>

#define WIDTH 3840
#define HEIGHT 2160
#define BLOCK_SIZE 16
#define KERNEL_RADIUS 1

volatile bool keep_running = true;
void sigint_handler(int) { keep_running = false; }

#define CUDA_CHECK(call)                                                \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            std::cerr << "CUDA error " << cudaGetErrorString(err)        \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1);                                                     \
        }                                                                \
    } while (0)

// Horizontal separable pass (reads input, writes tmp)
__global__ void blurHorizontal(const uint8_t* in, uint8_t* tmp, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int sum = 0;
    for (int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; ++k) {
        int nx = x + k;
        if (nx >= 0 && nx < width) sum += in[y * width + nx];
    }
    tmp[y * width + x] = static_cast<uint8_t>(sum / (2 * KERNEL_RADIUS + 1));
}

// Vertical separable pass (reads tmp, writes out)
__global__ void blurVertical(const uint8_t* tmp, uint8_t* out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int sum = 0;
    for (int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; ++k) {
        int ny = y + k;
        if (ny >= 0 && ny < height) sum += tmp[ny * width + x];
    }
    out[y * width + x] = static_cast<uint8_t>(sum / (2 * KERNEL_RADIUS + 1));
}

int main() {
    signal(SIGINT, sigint_handler);

    // 1) Open camera
    const char* dev = "/dev/video0";
    int fd = open(dev, O_RDWR | O_NONBLOCK, 0);
    if (fd < 0) { perror("open"); return -1; }

    // 2) Set format
    v4l2_format fmt = {};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = WIDTH;
    fmt.fmt.pix.height = HEIGHT;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_GREY; // grayscale (1 byte per pixel)
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    if (ioctl(fd, VIDIOC_S_FMT, &fmt) < 0) { perror("VIDIOC_S_FMT"); close(fd); return -1; }

    int bytesperline = fmt.fmt.pix.bytesperline;
    int sizeimage = fmt.fmt.pix.sizeimage;
    if (bytesperline < WIDTH) bytesperline = WIDTH;
    if (sizeimage < WIDTH * HEIGHT) sizeimage = bytesperline * HEIGHT;
    std::cout << "Configured: bytesperline=" << bytesperline << " sizeimage=" << sizeimage << std::endl;

    // 3) Request MMAP buffers
    v4l2_requestbuffers req = {};
    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (ioctl(fd, VIDIOC_REQBUFS, &req) < 0) { perror("VIDIOC_REQBUFS"); close(fd); return -1; }
    if (req.count < 1) { std::cerr << "Insufficient buffer memory\n"; close(fd); return -1; }

    // mmap buffers
    struct Buffer { void* start; size_t length; };
    std::vector<Buffer> buffers(req.count);
    for (unsigned int i = 0; i < req.count; ++i) {
        v4l2_buffer buf = {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        if (ioctl(fd, VIDIOC_QUERYBUF, &buf) < 0) { perror("VIDIOC_QUERYBUF"); close(fd); return -1; }
        void* mmap_ptr = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
        if (mmap_ptr == MAP_FAILED) { perror("mmap"); close(fd); return -1; }
        buffers[i].start = mmap_ptr;
        buffers[i].length = buf.length;
        // queue buffer initially
        if (ioctl(fd, VIDIOC_QBUF, &buf) < 0) perror("VIDIOC_QBUF");
    }

    // start streaming
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_STREAMON, &type) < 0) { perror("VIDIOC_STREAMON"); for (auto &b: buffers) munmap(b.start, b.length); close(fd); return -1; }

    // 4) Pinned host double buffers (input and output)
    size_t host_frame_bytes = (size_t)WIDTH * (size_t)HEIGHT;
    uint8_t *h_in[2] = {nullptr, nullptr}, *h_out[2] = {nullptr, nullptr};
    CUDA_CHECK(cudaHostAlloc((void**)&h_in[0], host_frame_bytes, cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc((void**)&h_in[1], host_frame_bytes, cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc((void**)&h_out[0], host_frame_bytes, cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc((void**)&h_out[1], host_frame_bytes, cudaHostAllocPortable));

    // 5) Device buffers: two input buffers (double buffer), one tmp (for separable), two output buffers
    uint8_t *d_in[2] = {nullptr, nullptr}, *d_tmp = nullptr, *d_out[2] = {nullptr, nullptr};
    CUDA_CHECK(cudaMalloc(&d_in[0], host_frame_bytes));
    CUDA_CHECK(cudaMalloc(&d_in[1], host_frame_bytes));
    CUDA_CHECK(cudaMalloc(&d_tmp, host_frame_bytes));
    CUDA_CHECK(cudaMalloc(&d_out[0], host_frame_bytes));
    CUDA_CHECK(cudaMalloc(&d_out[1], host_frame_bytes));

    // 6) Streams
    cudaStream_t streams[2];
    CUDA_CHECK(cudaStreamCreate(&streams[0]));
    CUDA_CHECK(cudaStreamCreate(&streams[1]));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((WIDTH + BLOCK_SIZE - 1) / BLOCK_SIZE, (HEIGHT + BLOCK_SIZE - 1) / BLOCK_SIZE);

    std::cout << "Entering capture loop. Ctrl-C to stop.\n";

    // frame index toggles between 0 and 1
    int fb_idx = 0;

    while (keep_running) {
        // Dequeue a filled buffer (nonblocking; use poll/select for robust code)
        v4l2_buffer buf = {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        if (ioctl(fd, VIDIOC_DQBUF, &buf) < 0) {
            // no buffer ready; small sleep and continue
            usleep(1000);
            continue;
        }
        unsigned int idx = buf.index;
        if (idx >= buffers.size()) {
            std::cerr << "Invalid buffer index\n";
            if (ioctl(fd, VIDIOC_QBUF, &buf) < 0) perror("VIDIOC_QBUF");
            continue;
        }

        // Copy row-by-row from mmap buffer into pinned linear host input buffer (h_in[fb_idx])
        uint8_t* src = static_cast<uint8_t*>(buffers[idx].start);
        for (int y = 0; y < HEIGHT; ++y) {
            size_t src_offset = (size_t)y * (size_t)bytesperline;
            if (src_offset + WIDTH <= buffers[idx].length) {
                memcpy(h_in[fb_idx] + (size_t)y * WIDTH, src + src_offset, WIDTH);
            } else {
                // clip/fill zeros if buffer shorter than expected
                size_t avail = (buffers[idx].length > src_offset) ? (buffers[idx].length - src_offset) : 0;
                if (avail) memcpy(h_in[fb_idx] + (size_t)y * WIDTH, src + src_offset, std::min((size_t)WIDTH, avail));
                if (avail < (size_t)WIDTH) memset(h_in[fb_idx] + (size_t)y * WIDTH + avail, 0, WIDTH - avail);
            }
        }

        // requeue immediately so driver can refill buffer
        if (ioctl(fd, VIDIOC_QBUF, &buf) < 0) { perror("VIDIOC_QBUF (requeue)"); }

        // Async Host->Device copy on stream[fb_idx]
        CUDA_CHECK(cudaMemcpyAsync(d_in[fb_idx], h_in[fb_idx], host_frame_bytes, cudaMemcpyHostToDevice, streams[fb_idx]));

        // Launch separable blur in that stream:
        blurHorizontal<<<grid, block, 0, streams[fb_idx]>>>(d_in[fb_idx], d_tmp, WIDTH, HEIGHT);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) { std::cerr << "Kernel H error: " << cudaGetErrorString(err) << std::endl; break; }

        blurVertical<<<grid, block, 0, streams[fb_idx]>>>(d_tmp, d_out[fb_idx], WIDTH, HEIGHT);
        err = cudaGetLastError();
        if (err != cudaSuccess) { std::cerr << "Kernel V error: " << cudaGetErrorString(err) << std::endl; break; }

        // Async Device->Host copy of result into pinned h_out[fb_idx]
        CUDA_CHECK(cudaMemcpyAsync(h_out[fb_idx], d_out[fb_idx], host_frame_bytes, cudaMemcpyDeviceToHost, streams[fb_idx]));

        // Optionally: process h_out[fb_idx] on host (but must ensure stream[fb_idx] finished or use event)
        // For demo: print sample every N frames, but only after the stream completes.
        // We'll do a non-blocking check with cudaStreamQuery: if complete, print; otherwise skip.
        if (cudaStreamQuery(streams[fb_idx]) == cudaSuccess) {
            static int print_counter = 0;
            if ((print_counter++ % 60) == 0) {
                std::cout << "--- sample 5x5 (frame idx " << fb_idx << ") ---\n";
                for (int y = 0; y < 5; ++y) {
                    for (int x = 0; x < 5; ++x) {
                        std::cout << (int)h_out[fb_idx][y * WIDTH + x] << " ";
                    }
                    std::cout << "\n";
                }
            }
        } else {
            // if not ready yet, we don't block — we'll print it in a later iteration when ready
        }

        // Toggle double buffer index
        fb_idx ^= 1;
    }

    std::cout << "Stopping capture\n";

    // stop streaming
    if (ioctl(fd, VIDIOC_STREAMOFF, &type) < 0) perror("VIDIOC_STREAMOFF");

    // cleanup
    for (auto &b : buffers) { munmap(b.start, b.length); }
    CUDA_CHECK(cudaFree(d_in[0])); CUDA_CHECK(cudaFree(d_in[1]));
    CUDA_CHECK(cudaFree(d_tmp));
    CUDA_CHECK(cudaFree(d_out[0])); CUDA_CHECK(cudaFree(d_out[1]));
    CUDA_CHECK(cudaFreeHost(h_in[0])); CUDA_CHECK(cudaFreeHost(h_in[1]));
    CUDA_CHECK(cudaFreeHost(h_out[0])); CUDA_CHECK(cudaFreeHost(h_out[1]));
    CUDA_CHECK(cudaStreamDestroy(streams[0])); CUDA_CHECK(cudaStreamDestroy(streams[1]));
    close(fd);
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
