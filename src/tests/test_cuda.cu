#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

// -------------------------------------------------------------------------
// GPU TIMER
// -------------------------------------------------------------------------
struct GpuTimer {
    cudaEvent_t start, stop;
    GpuTimer()  { cudaEventCreate(&start); cudaEventCreate(&stop); }
    ~GpuTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void Start() { cudaEventRecord(start); }
    void Stop()  { cudaEventRecord(stop); cudaEventSynchronize(stop); }
    float ElapsedMs() { float ms; cudaEventElapsedTime(&ms, start, stop); return ms; }
};

// -------------------------------------------------------------------------
// KERNELS — uno por configuración de bloque para evitar templates en el benchmark
// Usamos un template para no duplicar código, el compilador genera los 3 kernels
// -------------------------------------------------------------------------
template<int BX, int BY>
__global__ void stencil_kernel(const float* __restrict__ T_old, float* __restrict__ T_new, const int W, const int H) {
    __shared__ float s_tile[BY + 2][BX + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int gx = blockIdx.x * BX + tx;
    int gy = blockIdx.y * BY + ty;

    // Carga central
    if (gx < W && gy < H)
        s_tile[ty + 1][tx + 1] = T_old[gy * W + gx];
    else
        s_tile[ty + 1][tx + 1] = 0.0f;

    // Halo superior
    if (ty == 0 && gy > 0)
        s_tile[0][tx + 1] = T_old[(gy - 1) * W + gx];

    // Halo inferior
    if (ty == BY - 1 && gy < H - 1)
        s_tile[BY + 1][tx + 1] = T_old[(gy + 1) * W + gx];

    // Halo izquierdo
    if (tx == 0 && gx > 0)
        s_tile[ty + 1][0] = T_old[gy * W + (gx - 1)];

    // Halo derecho
    if (tx == BX - 1 && gx < W - 1)
        s_tile[ty + 1][BX + 1] = T_old[gy * W + (gx + 1)];

    __syncthreads();

    if (gx > 0 && gx < W - 1 && gy > 0 && gy < H - 1) {
        float sum = s_tile[ty][tx + 1]     +
                    s_tile[ty + 2][tx + 1] +
                    s_tile[ty + 1][tx]     +
                    s_tile[ty + 1][tx + 2];
        T_new[gy * W + gx] = sum * 0.25f;
    }
}

// -------------------------------------------------------------------------
// FUNCIÓN DE BENCHMARK — ejecuta N repeticiones con una config de bloque
// Devuelve vector con los tiempos de cada repetición en ms
// -------------------------------------------------------------------------
template<int BX, int BY>
std::vector<float> run_benchmark(
    const float* d_src,        // datos iniciales ya en GPU (no se modifican)
    int W, int H, int NUM_ITERS, int NUM_REPS)
{
    const size_t bytes = W * H * sizeof(float);

    // Reservamos buffers propios para este benchmark
    // así cada config empieza siempre desde los mismos datos
    float *d_old, *d_new;
    cudaMalloc(&d_old, bytes);
    cudaMalloc(&d_new, bytes);

    dim3 blockSize(BX, BY);
    dim3 gridSize((W + BX - 1) / BX, (H + BY - 1) / BY);

    std::vector<float> times(NUM_REPS);
    GpuTimer timer;

    for (int rep = 0; rep < NUM_REPS; ++rep) {
        // Restauramos los datos iniciales antes de cada repetición
        // para que todas las reps sean comparables
        cudaMemcpy(d_old, d_src, bytes, cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_new, d_src, bytes, cudaMemcpyDeviceToDevice);

        float* cur_old = d_old;
        float* cur_new = d_new;

        timer.Start();
        for (int i = 0; i < NUM_ITERS; ++i) {
            stencil_kernel<BX, BY><<<gridSize, blockSize>>>(cur_old, cur_new, W, H);
            std::swap(cur_old, cur_new);
        }
        timer.Stop();

        times[rep] = timer.ElapsedMs();
    }

    cudaFree(d_old);
    cudaFree(d_new);
    return times;
}

// -------------------------------------------------------------------------
// ESTADÍSTICAS — media, min, max y desviación típica
// -------------------------------------------------------------------------
struct Stats {
    float mean, min, max, stddev;
};

Stats compute_stats(const std::vector<float>& v) {
    Stats s;
    s.min = *std::min_element(v.begin(), v.end());
    s.max = *std::max_element(v.begin(), v.end());
    s.mean = std::accumulate(v.begin(), v.end(), 0.0f) / v.size();
    float var = 0;
    for (float x : v) var += (x - s.mean) * (x - s.mean);
    s.stddev = std::sqrt(var / v.size());
    return s;
}

// -------------------------------------------------------------------------
// IMPRESIÓN — muestra cada repetición y el resumen estadístico
// -------------------------------------------------------------------------
void print_results(const std::string& label, int bx, int by,
                   int W, int NUM_ITERS, const std::vector<float>& times)
{
    Stats st = compute_stats(times);

    std::cout << "\n╔══════════════════════════════════════════════════════╗\n";
    std::cout << "  Bloque: " << label << "  (" << bx << "×" << by << " = " << bx*by << " hilos)\n";
    std::cout << "  Matriz: " << W << "×" << W << "  |  Iteraciones: " << NUM_ITERS << "\n";
    std::cout << "╠══════════════════════════════════════════════════════╣\n";

    for (int i = 0; i < (int)times.size(); ++i) {
        std::cout << "  Rep " << std::setw(2) << (i+1) << ":  "
                  << std::fixed << std::setprecision(3) << std::setw(9) << times[i] << " ms"
                  << "  (" << std::setprecision(5) << times[i]/1000.0f << " s)\n";
    }

    std::cout << "╠══════════════════════════════════════════════════════╣\n";
    std::cout << "  Media:   " << std::fixed << std::setprecision(3) << std::setw(9) << st.mean   << " ms\n";
    std::cout << "  Mínimo:  " << std::setw(9) << st.min    << " ms\n";
    std::cout << "  Máximo:  " << std::setw(9) << st.max    << " ms\n";
    std::cout << "  Std dev: " << std::setw(9) << st.stddev << " ms\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n";
}

// -------------------------------------------------------------------------
// MAIN
// -------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Uso: " << argv[0] << " <DIM_N> <ITER> [REPS=10]\n";
        return 1;
    }

    const int DIM_N    = std::stoi(argv[1]);
    const int NUM_ITERS = std::stoi(argv[2]);
    const int NUM_REPS  = (argc >= 4) ? std::stoi(argv[3]) : 10;
    const int W = DIM_N, H = DIM_N;
    const size_t bytes = W * H * sizeof(float);

    std::cout << "=======================================================\n";
    std::cout << "  BENCHMARK STENCIL CUDA\n";
    std::cout << "  Matriz: " << W << "×" << H
              << "  |  Iteraciones: " << NUM_ITERS
              << "  |  Repeticiones: " << NUM_REPS << "\n";
    std::cout << "=======================================================\n";

    // Preparamos los datos iniciales en CPU
    std::vector<float> h_init(W * H, 0.0f);
    for (int j = 0; j < W; ++j) {
        h_init[j] = 100.0f;        // borde superior
        h_init[j] = 100.0f;
    }

    // Subimos los datos iniciales a GPU una sola vez
    // Cada benchmark los copiará desde aquí con DeviceToDevice (muy rápido)
    float* d_src;
    cudaMalloc(&d_src, bytes);
    cudaMemcpy(d_src, h_init.data(), bytes, cudaMemcpyHostToDevice);

    // -------------------------------------------------------
    // Ejecutamos los 3 benchmarks
    // -------------------------------------------------------
    auto times_8  = run_benchmark< 8,  8>(d_src, W, H, NUM_ITERS, NUM_REPS);
    auto times_16 = run_benchmark<16, 16>(d_src, W, H, NUM_ITERS, NUM_REPS);
    auto times_32 = run_benchmark<32, 32>(d_src, W, H, NUM_ITERS, NUM_REPS);

    // -------------------------------------------------------
    // Resultados individuales
    // -------------------------------------------------------
    print_results("8×8",   8,  8,  W, NUM_ITERS, times_8);
    print_results("16×16", 16, 16, W, NUM_ITERS, times_16);
    print_results("32×32", 32, 32, W, NUM_ITERS, times_32);

    // -------------------------------------------------------
    // Tabla comparativa final
    // -------------------------------------------------------
    Stats s8  = compute_stats(times_8);
    Stats s16 = compute_stats(times_16);
    Stats s32 = compute_stats(times_32);

    std::cout << "\n\n╔═══════════════════════════════════════════════════════════╗\n";
    std::cout <<   "  COMPARATIVA FINAL (media de " << NUM_REPS << " repeticiones)\n";
    std::cout <<   "╠═══════════════╦══════════════╦══════════════╦═════════════╣\n";
    std::cout <<   "  Bloque        │   Media (ms) │    Min (ms)  │  Std dev    \n";
    std::cout <<   "╠═══════════════╬══════════════╬══════════════╬═════════════╣\n";

    auto row = [](const std::string& name, const Stats& s) {
        std::cout << "  " << std::left << std::setw(13) << name
                  << " │ " << std::right << std::fixed << std::setprecision(3)
                  << std::setw(12) << s.mean
                  << " │ " << std::setw(12) << s.min
                  << " │ " << std::setw(9)  << s.stddev << "\n";
    };

    row("8×8",   s8);
    row("16×16", s16);
    row("32×32", s32);

    // Ganador
    float best = std::min({s8.mean, s16.mean, s32.mean});
    std::string winner = (best == s8.mean) ? "8×8" : (best == s16.mean) ? "16×16" : "32×32";
    std::cout << "╠═══════════════════════════════════════════════════════════╣\n";
    std::cout << "  Ganador: " << winner
              << "  (" << std::fixed << std::setprecision(3) << best << " ms media)\n";
    std::cout << "╚═══════════════════════════════════════════════════════════╝\n";

    cudaFree(d_src);
    return 0;
}