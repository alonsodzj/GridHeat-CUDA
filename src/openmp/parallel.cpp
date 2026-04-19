#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <omp.h>

void print_value(const int DIM_N, const int W, const int H, const float* matrix) {
    if (DIM_N <= 20) {
        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < W; ++j)
                std::cout << std::setw(8) << matrix[i * W + j];
            std::cout << "\n";
        }
    } else {
        std::cout << "Valor central: " << matrix[(H/2)*W + (W/2)] << std::endl;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Uso: " << argv[0] << " <DIM_N> <ITER>" << std::endl;
        return 1;
    }

    const int DIM_N    = std::stoi(argv[1]);
    const int NUM_ITERS = std::stoi(argv[2]);
    const int W = DIM_N, H = DIM_N;

    std::vector<float> buf0(W * H, 0.0f);
    std::vector<float> buf1(W * H, 0.0f);
    for (int j = 0; j < W; ++j)
        buf0[j] = buf1[j] = 100.0f;

    float* grids[2] = { buf0.data(), buf1.data() };

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Dimensiones: " << W << "x" << H
              << " (" << NUM_ITERS << " iters)" << std::endl;

    // ── Parámetros de cache blocking ──────────────────────────────────────────
    // K = iteraciones por tile. Más K = más reutilización de caché, pero tile
    // de entrada crece (T + 2K). Ajusta según tu L3 (aquí asumimos ~8MB L3).
    // Fórmula: (TILE_W + 2K) * (TILE_H + 2K) * 2 buffers * 4 bytes < L3_size
    const int K       = 8;    // iteraciones temporales por tile
    const int TILE_W  = 128;  // tile de SALIDA (el de entrada será TILE_W + 2K)
    const int TILE_H  = 128;

    double start_time = omp_get_wtime();

    #pragma omp parallel
    {
        #pragma omp single
        std::cout << "Hilos: " << omp_get_num_threads() << std::endl;

        // Recorremos la matriz en bloques de salida de TILE_W x TILE_H
        // Cada bloque avanza K iteraciones de forma independiente
        // Los bloques NO tienen dependencias entre sí dentro de las K iters
        // porque usamos el halo expandido como región de lectura
        #pragma omp for schedule(static) collapse(2)
        for (int by = 1; by < H - 1; by += TILE_H) {
            for (int bx = 1; bx < W - 1; bx += TILE_W) {

                // Procesamos este tile en pasos de K iteraciones globales
                for (int iter_base = 0; iter_base < NUM_ITERS; iter_base += K) {
                    const int steps = std::min(K, NUM_ITERS - iter_base);

                    // En cada paso k (0..steps-1):
                    //   - El tile de SALIDA válido en el paso k es:
                    //       [by, by+TILE_H) x [bx, bx+TILE_W)
                    //   - El tile de ENTRADA necesario (halo) es:
                    //       [by-k-1, by+TILE_H+k+1) x [bx-k-1, bx+TILE_W+k+1)
                    //   Pero para simplificar, leemos con halo máximo (K) y
                    //   solo escribimos la región interior válida.
                    for (int k = 0; k < steps; ++k) {
                        // índice del buffer fuente/destino para esta micro-iter
                        const int src = (iter_base + k)     & 1;
                        const int dst = (iter_base + k + 1) & 1;

                        const float* __restrict__ T_old = grids[src];
                        float*       __restrict__ T_new = grids[dst];

                        // Región de salida válida para esta micro-iteración:
                        // encogemos el tile en (K-k) por cada borde para
                        // respetar el cono de dependencia
                        const int margin = K - k; // halo que ya no podemos calcular
                                                   // correctamente sin más contexto
                        const int y0 = std::max(1,     by - k);
                        const int y1 = std::min(H - 1, by + TILE_H + k);
                        const int x0 = std::max(1,     bx - k);
                        const int x1 = std::min(W - 1, bx + TILE_W + k);

                        for (int y = y0; y < y1; ++y) {
                            const int actual = y * W;
                            const int arriba = actual - W;
                            const int abajo  = actual + W;

                            #pragma omp simd
                            for (int x = x0; x < x1; ++x) {
                                T_new[actual + x] =
                                    (T_old[arriba + x] +
                                     T_old[abajo  + x] +
                                     T_old[actual + x - 1] +
                                     T_old[actual + x + 1]) * 0.25f;
                            }
                        }
                    }
                }
            }
        }
    }

    // El buffer final correcto: NUM_ITERS pares → grids[0], impares → grids[1]
    float* result = grids[NUM_ITERS & 1];

    double end_time = omp_get_wtime();
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "Tiempo: " << end_time - start_time << " segundos." << std::endl;

    print_value(DIM_N, W, H, result);
    return 0;
}
// g++ -O3 -march=native -ffast-math -fopenmp stencil.cpp -o stencil