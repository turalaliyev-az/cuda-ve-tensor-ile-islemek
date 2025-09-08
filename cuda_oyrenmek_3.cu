#include <iostream>
#include <cuda_runtime.h>

// 4K çözünürlük
#define WIDTH 3840
#define HEIGHT 2160
#define BLOCK_SIZE 16
#define KERNEL_RADIUS 1   // 3x3 box filter için
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
        
int main(){
    
 size_t size = WIDTH * HEIGHT * sizeof(float);

    // Host
    float* h_input = new float[WIDTH * HEIGHT];
    float* h_output = new float[WIDTH * HEIGHT];

    // Test üçün dummy data: hər pikselin dəyəri = 1
    for (int i = 0; i < WIDTH * HEIGHT; i++) h_input[i] = 1.0f;

    // Device
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    // Host -> Device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Thread block və grid ölçüləri
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((WIDTH + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (HEIGHT + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Kernel launch
    blurKernel<<<grid, block>>>(d_output, d_input, WIDTH, HEIGHT);
    cudaDeviceSynchronize();

    // Device -> Host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Nümunə üçün 5x5 hissəsini çap edək
    for (int y = 0; y < 5; y++) {
        for (int x = 0; x < 5; x++) {
            std::cout << h_output[y * WIDTH + x] << " ";
        }
        std::cout << std::endl;
    }

    // Cleanup
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}