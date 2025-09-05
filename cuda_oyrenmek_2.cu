#include <cuda_runtime.h>
#include <iostream>

#define WIDTH 8
#define HEIGHT 8
#define KERNEL_RADIUS 1   // 3x3 box filter için

// 2D Blur Kernel
__global__ void blurKernel(float* output, const float* input, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float sum = 0.0f;
    int count = 0;

    // 3x3 komşuluk
    for (int dy = -KERNEL_RADIUS; dy <= KERNEL_RADIUS; dy++) {
        for (int dx = -KERNEL_RADIUS; dx <= KERNEL_RADIUS; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                sum += input[ny * width + nx];
                count++;
            }
        }
    }

    output[y * width + x] = sum / count;
}

int main() {
    float h_input[WIDTH*HEIGHT];
    float h_output[WIDTH*HEIGHT];

    // Input diziyi doldur
    for(int i = 0; i < WIDTH*HEIGHT; i++)
        h_input[i] = i;

    float *d_input, *d_output;
    cudaMalloc(&d_input, WIDTH*HEIGHT*sizeof(float));
    cudaMalloc(&d_output, WIDTH*HEIGHT*sizeof(float));

    // Host -> Device
    cudaMemcpy(d_input, h_input, WIDTH*HEIGHT*sizeof(float), cudaMemcpyHostToDevice);

    // Kernel launch
    dim3 threadsPerBlock(4,4);
    dim3 numBlocks((WIDTH+3)/4, (HEIGHT+3)/4);
    blurKernel<<<numBlocks, threadsPerBlock>>>(d_output, d_input, WIDTH, HEIGHT);

    // Device -> Host
    cudaMemcpy(h_output, d_output, WIDTH*HEIGHT*sizeof(float), cudaMemcpyDeviceToHost);

    // Sonuçları yazdır
    for(int y = 0; y < HEIGHT; y++) {
        for(int x = 0; x < WIDTH; x++) {
            std::cout << h_output[y*WIDTH + x] << "\t";
        }
        std::cout << std::endl;
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
 