// v4l2_cuda_blur.cu
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

// Simple 3x3 box-blur kernel (grayscale)
__global__ void blurKernel(uint8_t* output, const uint8_t* input, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int sum = 0;
    int count = 0;
    for (int dy = -KERNEL_RADIUS; dy <= KERNEL_RADIUS; ++dy) {
        for (int dx = -KERNEL_RADIUS; dx <= KERNEL_RADIUS; ++dx) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                sum += input[ny * width + nx];
                ++count;
            }
        }
    }
    output[y * width + x] = static_cast<uint8_t>(sum / count);
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

    // After setting format, get bytesperline and sizeimage from fmt
    int bytesperline = fmt.fmt.pix.bytesperline;
    int sizeimage = fmt.fmt.pix.sizeimage;
    if (bytesperline < WIDTH) {
        // driver may give 0 or less than width for some configs; fall back to WIDTH
        bytesperline = WIDTH;
    }
    if (sizeimage < WIDTH * HEIGHT) {
        // fallback; sizeimage sometimes zero
        sizeimage = bytesperline * HEIGHT;
    }
    std::cout << "Configured: bytesperline=" << bytesperline << " sizeimage=" << sizeimage << std::endl;

    // 3) Request buffers (mmap)
    v4l2_requestbuffers req = {};
    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (ioctl(fd, VIDIOC_REQBUFS, &req) < 0) { perror("VIDIOC_REQBUFS"); close(fd); return -1; }
    if (req.count < 1) { std::cerr << "Insufficient buffer memory\n"; close(fd); return -1; }

    // mmap all buffers
    struct Buffer {
        void* start;
        size_t length;
    };
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

        // queue buffers initially
        if (ioctl(fd, VIDIOC_QBUF, &buf) < 0) { perror("VIDIOC_QBUF"); /*continue*/ }
    }

    // start streaming
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_STREAMON, &type) < 0) { perror("VIDIOC_STREAMON"); for (auto &b: buffers) munmap(b.start, b.length); close(fd); return -1; }

    // Prepare intermediate host buffer (linear, WxH)
    size_t host_frame_bytes = (size_t)WIDTH * (size_t)HEIGHT;
    uint8_t* h_frame = new uint8_t[host_frame_bytes];
    if (!h_frame) { std::cerr << "Failed to allocate host frame\n"; return -1; }

    // CUDA device buffers
    uint8_t *d_input = nullptr, *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, host_frame_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, host_frame_bytes));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((WIDTH + BLOCK_SIZE - 1) / BLOCK_SIZE, (HEIGHT + BLOCK_SIZE - 1) / BLOCK_SIZE);

    std::cout << "Entering capture loop. Press Ctrl-C to stop.\n";

    while (keep_running) {
        // Dequeue a buffer (blocking using select/poll is better; here we simple loop with small sleep)
        v4l2_buffer buf = {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        if (ioctl(fd, VIDIOC_DQBUF, &buf) < 0) {
            // If no buffer ready, sleep shortly
            usleep(1000);
            continue;
        }
        // buf.index tells which mmap buffer we have
        unsigned int idx = buf.index;
        if (idx >= buffers.size()) {
            std::cerr << "Invalid buffer index\n";
            // requeue and continue
            if (ioctl(fd, VIDIOC_QBUF, &buf) < 0) perror("VIDIOC_QBUF");
            continue;
        }

        // Copy row-by-row from mmap buffer (which may have stride bytesperline) into linear h_frame (WIDTH bytes per row)
        uint8_t* src = static_cast<uint8_t*>(buffers[idx].start);
        for (int y = 0; y < HEIGHT; ++y) {
            // make sure not to read beyond mmap length
            size_t src_offset = (size_t)y * (size_t)bytesperline;
            if (src_offset + WIDTH > buffers[idx].length) {
                // fallback: clamp
                size_t avail = (buffers[idx].length > src_offset) ? (buffers[idx].length - src_offset) : 0;
                if (avail > 0) memcpy(h_frame + (size_t)y * WIDTH, src + src_offset, std::min((size_t)WIDTH, avail));
                else memset(h_frame + (size_t)y * WIDTH, 0, WIDTH);
            } else {
                memcpy(h_frame + (size_t)y * WIDTH, src + src_offset, WIDTH);
            }
        }

        // Now copy corrected linear host frame to device
        CUDA_CHECK(cudaMemcpy(d_input, h_frame, host_frame_bytes, cudaMemcpyHostToDevice));

        // Kernel: process
        blurKernel<<<grid, block>>>(d_output, d_input, WIDTH, HEIGHT);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy back
        CUDA_CHECK(cudaMemcpy(h_frame, d_output, host_frame_bytes, cudaMemcpyDeviceToHost));

        // At this point h_frame contains processed grayscale image (linear WIDTH*HEIGHT)
        // For demo, print top-left 5x5 block (once per many frames to reduce spam)
        static int frame_counter = 0;
        if ((frame_counter++ % 60) == 0) {
            std::cout << "--- sample 5x5 ---\n";
            for (int y = 0; y < 5; ++y) {
                for (int x = 0; x < 5; ++x) {
                    std::cout << (int)h_frame[y * WIDTH + x] << " ";
                }
                std::cout << "\n";
            }
        }

        // Re-queue the buffer so driver can fill it again
        if (ioctl(fd, VIDIOC_QBUF, &buf) < 0) {
            perror("VIDIOC_QBUF (requeue)");
            break;
        }
    }

    std::cout << "Stopping capture\n";

    // stop streaming
    if (ioctl(fd, VIDIOC_STREAMOFF, &type) < 0) perror("VIDIOC_STREAMOFF");

    // cleanup
    for (auto &b : buffers) {
        munmap(b.start, b.length);
    }
    delete[] h_frame;
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    close(fd);
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
