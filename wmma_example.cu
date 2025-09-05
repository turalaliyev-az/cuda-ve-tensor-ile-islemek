#include <cuda_runtime.h>
#include <iostream>

__global__ void addKernel(int *a, int *b, int *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int arraySize = 5;
    const int a[arraySize] = {1, 2, 3, 4, 5};
    const int b[arraySize] = {10, 20, 30, 40, 50};
    int c[arraySize] = {0};

    int *dev_a = nullptr;
    int *dev_b = nullptr;
    int *dev_c = nullptr;

    cudaMalloc((void**)&dev_a, arraySize * sizeof(int));
    cudaMalloc((void**)&dev_b, arraySize * sizeof(int));
    cudaMalloc((void**)&dev_c, arraySize * sizeof(int));

    cudaMemcpy(dev_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    addKernel<<<1, arraySize>>>(dev_a, dev_b, dev_c, arraySize);

    cudaMemcpy(c, dev_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Result: ";
    for (int i = 0; i < arraySize; i++) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}