#include <iostream>

__global__ void check() {
    printf("Hola desde el bloque %d, hilo %d\n", blockIdx.x, threadIdx.x);
}

int main() {
    check<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
//nvcc -O3 -arch=sm_75 mi_proyecto.cu -o ejecutable