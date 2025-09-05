#include <cuda_runtime.h>
#include <iostream>

__global__ void addKernel(int *c, int *a, int *b, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int N = 16;
    int h_a[N], h_b[N], h_c[N];

    // Dizileri dolduralım
    for(int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i*10;
    }

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N*sizeof(int));
    cudaMalloc(&d_b, N*sizeof(int));
    cudaMalloc(&d_c, N*sizeof(int));

    // Host -> Device kopyala
    cudaMemcpy(d_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N*sizeof(int), cudaMemcpyHostToDevice);

    // Kernel launch: 4 thread per block
    addKernel<<<1, N>>>(d_c, d_a, d_b, N);

    // Device -> Host kopyala
    cudaMemcpy(h_c, d_c, N*sizeof(int), cudaMemcpyDeviceToHost);

    // Sonuçları yazdır
    for(int i = 0; i < N; i++) {
        std::cout << h_a[i] << " + " << h_b[i] << " = " << h_c[i] << std::endl;
    }

    // Belleği temizle
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
