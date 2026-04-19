#include <vector>
#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm> // Para std::swap
#include <chrono>    // Para medir el tiempo que tarda
#include <omp.h>

void print_matrix(const std::vector<float>& matrix, const int W, const int H) {
    std::cout << "--- Matrix State ---" << std::endl;
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            std::cout << std::setw(8) << matrix[i * W + j]; 
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

void print_value(const int DIM_N, const int W, const int H, const std::vector<float>& matrix){
    if (DIM_N <= 20) {
        print_matrix(matrix, W, H);
    } else {
        std::cout << "Cálculo finalizado. Valor central: " << matrix[(H/2)*W + (W/2)] << std::endl;
    }
}

void compute_stencil_parallel(const int W, const int H, const std::vector<float>& T_old, std::vector<float>& T_new) {
    // Se paraleliza el bucle EXTERNO (filas). Esto es crítico para la caché:
    // C/C++ almacena los vectores en memoria contigua (Row-Major). Al repartir 
    // bloques enteros de filas a cada hilo, evitamos el "False Sharing" (falso uso 
    // compartido) y maximizamos los aciertos (hits) en la caché L1/L2.
    #pragma omp for schedule(static) //repartimos la carga estática porque cada iteración tarda lo mismo
    for (int i = 1; i < H - 1; ++i) {
        // Precalculamos los índices base de las filas para esta iteración.
        // Esto le ahorra a la CPU hacer multiplicaciones repetitivas en el bucle interno.
        const int actual = i * W;
        const int arriba = (i - 1) * W;
        const int abajo  = (i + 1) * W;

        // '#pragma omp simd' instruye al compilador a vectorizar este bucle interno.
        // Utiliza instrucciones avanzadas (como AVX en Intel/AMD) para calcular
        // múltiples 'floats' en un solo ciclo de reloj, ya que los datos de 'j' 
        // son perfectamente adyacentes en la memoria.
        #pragma omp simd
        for (int j = 1; j < W - 1; ++j) {
            T_new[actual + j] = (T_old[arriba + j] + 
                                 T_old[abajo + j]  + 
                                 T_old[actual + (j - 1)] + 
                                 T_old[actual + (j + 1)]) * 0.25f;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Uso: " << argv[0] << " <DIM_N> <ITER>" << std::endl;
        return 1; 
    }

    int DIM_N = 0;
    int NUM_ITERS = 0;

    try {
        DIM_N = std::stoi(argv[1]);
        NUM_ITERS = std::stoi(argv[2]);
    } catch (...) {
        std::cerr << "Error: Argumentos inválidos." << std::endl;
        return 1;
    }

    const int W = DIM_N; 
    const int H = DIM_N; 
    const int DIM = W * H;

    std::vector<float> matrix_old(DIM, 0.0f);
    std::vector<float> matrix_new(DIM, 0.0f);

    for(int j = 0; j < W; ++j) {
        matrix_old[j] = 100.0f;
        matrix_new[j] = 100.0f; 
    }
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Dimensiones: " << W << "x" << H << " (" << NUM_ITERS << " iters)" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    // Se crea el equipo de hilos UNA SOLA VEZ. Gran decisión arquitectónica.
    #pragma omp parallel
    for(int i = 0; i < NUM_ITERS; ++i) {
        
        // Llamamos a la versión PARALELIZADA
        compute_stencil_parallel(W, H, matrix_old, matrix_new);
        
        // Intercambiamos buffers.
        #pragma omp single
        {
            // NOTA DE SINCRONIZACIÓN:
            // 1. '#pragma omp for' (en compute_stencil) tiene una barrera implícita al final.
            //    Ningún hilo hará el 'swap' hasta que todos hayan acabado de calcular.
            // 2. '#pragma omp single' también tiene una barrera implícita al final.
            //    Ningún hilo empezará la siguiente iteración hasta que el 'swap' haya finalizado.
            std::swap(matrix_old, matrix_new);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    
    // Mostramos el tiempo paralelo real (cambié el texto por coherencia)
    std::cout << "Tiempo de ejecución (OpenMP): " << diff.count() << " segundos" << std::endl;

    print_value(DIM_N, W, H, matrix_old);

    return 0;
}
//g++ -O3 -fopenmp parallel.cpp -o pll