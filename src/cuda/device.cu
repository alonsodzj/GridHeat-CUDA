#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <algorithm> // Para std::swap

// Definimos el tamaño del bloque (16x16 es el estándar de oro en CUDA para matrices 2D)
#define BLOCK_X 16
#define BLOCK_Y 16

// -------------------------------------------------------------------------
// KERNEL DE CUDA
// -------------------------------------------------------------------------
__global__ void compute_stencil_cuda_shared(const float* T_old, float* T_new, int W, int H) {
    // 1. MEMORIA COMPARTIDA: Es como una caché L1 manual ultrarrápida.
    // Reservamos espacio para el bloque (16x16) MÁS 1 píxel de halo/borde por cada lado.
    // Por tanto: 16 + 2 = 18.
    __shared__ float s_tile[BLOCK_Y + 2][BLOCK_X + 2];

    // Identificadores locales del hilo dentro del bloque
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Identificadores globales del hilo en toda la matriz
    int gx = blockIdx.x * blockDim.x + tx;
    int gy = blockIdx.y * blockDim.y + ty;

    // 2. CARGA COOPERATIVA EN MEMORIA COMPARTIDA
    // Cada hilo carga su propio píxel central en el tile compartido.
    // Se suma +1 a los índices de shared para dejar espacio al halo.
    if (gx < W && gy < H) //esto lo vimos en clase para no exceder los índices
	{
        s_tile[ty + 1][tx + 1] = T_old[gy * W + gx];
    } else {
        s_tile[ty + 1][tx + 1] = 0.0f; // Padding de ceros si el bloque excede la matriz
    }

    // Ahora los hilos que están en los bordes del bloque cargan el halo (apron)
    // Halo superior (solo los hilos de la primera fila del bloque)
    if (ty == 0 && gy > 0) {
        s_tile[0][tx + 1] = T_old[(gy - 1) * W + gx];
    }
    // Halo inferior (solo los hilos de la última fila del bloque)
    if (ty == blockDim.y - 1 && gy < H - 1) {
        s_tile[blockDim.y + 1][tx + 1] = T_old[(gy + 1) * W + gx];
    }
    // Halo izquierdo (solo los hilos de la primera columna del bloque)
    if (tx == 0 && gx > 0) {
        s_tile[ty + 1][0] = T_old[gy * W + (gx - 1)];
    }
    // Halo derecho (solo los hilos de la última columna del bloque)
    if (tx == blockDim.x - 1 && gx < W - 1) {
        s_tile[ty + 1][blockDim.x + 1] = T_old[gy * W + (gx + 1)];
    }

    // 3. SINCRONIZACIÓN BARRERA
    // ¡CRÍTICO! Esperamos a que TODOS los hilos del bloque hayan terminado de cargar 
    // sus datos en s_tile antes de que nadie empiece a calcular.
    __syncthreads();

    // 4. CÁLCULO DEL STENCIL (Aprovechando la memoria compartida)
    // Respetamos los bordes de la matriz global (no calculamos en x=0, x=W-1, y=0, y=H-1)
    if (gx > 0 && gx < W - 1 && gy > 0 && gy < H - 1) {
        
        // El hilo lee de su vecindario en la memoria compartida, no en la memoria global (VRAM).
        // Esto reduce drásticamente los accesos a la memoria principal de la tarjeta gráfica.
        float sum = s_tile[ty][tx + 1]     + // Arriba
                    s_tile[ty + 2][tx + 1] + // Abajo
                    s_tile[ty + 1][tx]     + // Izquierda
                    s_tile[ty + 1][tx + 2];  // Derecha

        // Escribimos el resultado en la memoria global (VRAM)
        T_new[gy * W + gx] = sum * 0.25f;
    }
}

// -------------------------------------------------------------------------
// FUNCIONES HOST (CPU)
// -------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Uso: " << argv[0] << " <DIM_N> <ITER>" << std::endl;
        return 1; 
    }

    int DIM_N = std::stoi(argv[1]);
    int NUM_ITERS = std::stoi(argv[2]);

    const int W = DIM_N; 
    const int H = DIM_N; 
    const size_t bytes = W * H * sizeof(float);

    // Matrices en el Host (RAM de la CPU)
    std::vector<float> h_old(W * H, 0.0f);
    std::vector<float> h_new(W * H, 0.0f);

    // Inicializar borde superior
    for(int j = 0; j < W; ++j) {
        h_old[j] = 100.0f;
        h_new[j] = 100.0f; 
    }

    // Punteros para el Device (VRAM de la GPU)
    float *d_old, *d_new;
    cudaMalloc(&d_old, bytes);
    cudaMalloc(&d_new, bytes);

    // Transferimos los datos iniciales de Host (CPU) a Device (GPU)
    cudaMemcpy(d_old, h_old.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_new, h_new.data(), bytes, cudaMemcpyHostToDevice);

    // Configuración del Grid y Blocks
    dim3 blockSize(BLOCK_X, BLOCK_Y); // 16 x 16
    // Calculamos cuántos bloques necesitamos para cubrir toda la matriz. 
    // Usamos (W + BLOCK_X - 1) / BLOCK_X para redondear siempre hacia arriba.
    dim3 gridSize((W + blockSize.x - 1) / blockSize.x, 
                  (H + blockSize.y - 1) / blockSize.y);

    std::cout << "Lanzando kernel en GPU con Grid: (" << gridSize.x << "x" << gridSize.y << ") "
              << "y Block: (" << blockSize.x << "x" << blockSize.y << ")\n";

    auto start = std::chrono::high_resolution_clock::now();

    // Bucle temporal en el Host (Lanza múltiples kernels)
    for(int i = 0; i < NUM_ITERS; ++i) {
        compute_stencil_cuda_shared<<<gridSize, blockSize>>>(d_old, d_new, W, H);
        
        // Intercambiamos los punteros DE LA GPU en la CPU. 
        // ¡No movemos datos! Solo le decimos que en la siguiente iteración, 
        // d_old apunte a d_new y viceversa. Es instantáneo.
        std::swap(d_old, d_new);
    }

    // Sincronizamos la GPU para asegurarnos de que ha terminado todas las iteraciones
    // antes de parar el cronómetro. (CUDA ejecuta de forma asíncrona por defecto).
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Tiempo de ejecución (CUDA): " << std::fixed << std::setprecision(5) << diff.count() << " segundos" << std::endl;

    // Al final del bucle temporal, por el swap, el resultado final siempre queda en d_old.
    // Lo traemos de vuelta a la CPU.
    cudaMemcpy(h_old.data(), d_old, bytes, cudaMemcpyDeviceToHost);

    // Imprimir resultado central de control
    std::cout << "Valor central final: " << h_old[(H/2)*W + (W/2)] << std::endl;

    // Limpieza de VRAM
    cudaFree(d_old);
    cudaFree(d_new);

    return 0;
}