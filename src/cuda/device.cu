#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm> // Para std::swap

#define BLOCK_X 16
#define BLOCK_Y 16

// GPU TIMER
// Usamos eventos CUDA para medir el tiempo directamente en la GPU.
// Es más preciso que chrono porque no incluye latencias del driver de CPU.
struct GpuTimer {
    cudaEvent_t start, stop;
    GpuTimer()  { cudaEventCreate(&start); cudaEventCreate(&stop); }
    ~GpuTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void Start() { cudaEventRecord(start); }
    void Stop()  { cudaEventRecord(stop); cudaEventSynchronize(stop); }
    float ElapsedMs() { float ms; cudaEventElapsedTime(&ms, start, stop); return ms; }
};

// KERNEL DE CUDA
// Añadimos __restrict__ a los punteros para indicarle al compilador que
// T_old y T_new nunca se solapan en memoria. Esto habilita optimizaciones
__global__ void compute_stencil_cuda_shared(const float* __restrict__ T_old, float* __restrict__ T_new, int W, int H) {
    // 1. MEMORIA COMPARTIDA: Es como una caché L1 manual ultrarrápida.
    // Reservamos espacio para el bloque (32x16) MÁS 1 píxel de halo/borde por cada lado.
    // Por tanto: BLOCK_X + 2 = 34, BLOCK_Y + 2 = 18.
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
    if (gx < W && gy < H) // esto lo vimos en clase para no exceder los índices
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

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Uso: " << argv[0] << " <DIM_N> <ITER>" << std::endl;
        return 1; 
    }

    int DIM_N = std::stoi(argv[1]);
    int NUM_ITERS = std::stoi(argv[2]);

    const int W = DIM_N; 
    const int H = DIM_N; 
    const size_t SIZE_BYTES = W * H * sizeof(float);

    // Matrices en el Host (RAM de la CPU) usando la clase std::vector para aprovechar el rendimiento
    std::vector<float> h_old(W * H, 0.0f);
    std::vector<float> h_new(W * H, 0.0f);

    // Inicializar borde superior
    for(int j = 0; j < W; ++j) {
        h_old[j] = 100.0f;
        h_new[j] = 100.0f; 
    }

    // Punteros para el Device (VRAM de la GPU)
    float *d_old, *d_new;
	
	// Reservo memoria para el Device
    cudaMalloc(&d_old, SIZE_BYTES);
    cudaMalloc(&d_new, SIZE_BYTES);

    // Transferimos los datos iniciales de Host (CPU) a Device (GPU)
    cudaMemcpy(d_old, h_old.data(), SIZE_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_new, h_new.data(), SIZE_BYTES, cudaMemcpyHostToDevice);

    // Configuración del Grid y Blocks
    dim3 blockSize(BLOCK_X, BLOCK_Y); // hilos por bloque
    // Calculamos cuántos bloques necesitamos para cubrir toda la matriz. 
    // Usamos (W + BLOCK_X - 1) / BLOCK_X para redondear siempre hacia arriba.
    dim3 gridSize((W + blockSize.x - 1) / blockSize.x, 
                  (H + blockSize.y - 1) / blockSize.y);

    std::cout << "Lanzando kernel en GPU con Grid: (" << gridSize.x << "x" << gridSize.y << ") "
              << "y Block: (" << blockSize.x << "x" << blockSize.y << ")\n";

    // Creamos el timer de GPU antes del bucle
    GpuTimer timer;
    timer.Start();

    // Bucle temporal en el Host (Lanza múltiples kernels)
    for(int i = 0; i < NUM_ITERS; ++i) {
        compute_stencil_cuda_shared<<<gridSize, blockSize>>>(d_old, d_new, W, H);
        
        // Intercambiamos los punteros DE LA GPU en la CPU. 
        // ¡No movemos datos! Solo le decimos que en la siguiente iteración, 
        // d_old apunte a d_new y viceversa. Es instantáneo.
        std::swap(d_old, d_new);
    }

    // Paramos el timer. Internamente llama a cudaEventSynchronize,
    // así que ya no necesitamos cudaDeviceSynchronize() explícito.
    timer.Stop();

    std::cout << "Tiempo de ejecución (CUDA): " << std::fixed << std::setprecision(5) 
              << timer.ElapsedMs() / 1000.0f << " segundos" << std::endl;

    // Al final del bucle temporal, por el swap, el resultado final siempre queda en d_old.
    // Lo traemos de vuelta a la CPU.
    cudaMemcpy(h_old.data(), d_old, SIZE_BYTES, cudaMemcpyDeviceToHost);

    // Imprimir resultado central de control
    std::cout << "Valor central final: " << h_old[(H/2)*W + (W/2)] << std::endl;

    // Limpieza de VRAM una vez terminados todos los kernels
    cudaFree(d_old);
    cudaFree(d_new);

    return 0;
}