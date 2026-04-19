#include <vector>
#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>
#include <chrono>
#include <omp.h>

// Función auxiliar para reiniciar las matrices al estado inicial en cada test
void reset_matrix(std::vector<float>& m, const int W, const int H) {
    std::fill(m.begin(), m.end(), 0.0f); // Llenamos de ceros
    for(int j = 0; j < W; ++j) {
        m[j] = 100.0f; // Borde superior a 100
    }
}

void compute_stencil_parallel(const int W, const int H, const std::vector<float>& T_old, std::vector<float>& T_new) {
    #pragma omp for schedule(static)
    for (int i = 1; i < H - 1; ++i) {
        const int actual = i * W;
        const int arriba = (i - 1) * W;
        const int abajo  = (i + 1) * W;

        #pragma omp simd
        for (int j = 1; j < W - 1; ++j) {
            T_new[actual + j] = (T_old[arriba + j] + 
                                 T_old[abajo + j]  + 
                                 T_old[actual + (j - 1)] + 
                                 T_old[actual + (j + 1)]) * 0.25f;
        }
    }
}

void compute_stencil_sequencial(const int W, const int H, const std::vector<float>& T_old, std::vector<float>& T_new) {
    for (int i = 1; i < H - 1; ++i) {
        const int actual = i * W;
        const int arriba = (i - 1) * W;
        const int abajo  = (i + 1) * W;

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

    int DIM_N = 0, NUM_ITERS = 0;
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
    const int NUM_TESTS = 5; // Número de veces que se ejecutará la comparativa

    std::vector<float> matrix_old(DIM);
    std::vector<float> matrix_new(DIM);

    std::cout << "Iniciando Benchmark..." << std::endl;
    std::cout << "Dimensiones: " << W << "x" << H << " | Iteraciones: " << NUM_ITERS << " | Tests: " << NUM_TESTS << "\n" << std::endl;
    
    // Cabecera de la tabla solicitada
    std::cout << "Secuencial(s)\tParalelo(s)" << std::endl;
    std::cout << "-----------------------------------" << std::endl;

    double total_time_seq = 0.0;
    double total_time_par = 0.0;

    for (int test = 0; test < NUM_TESTS; ++test) {
        
        // --- 1. EJECUCIÓN SECUENCIAL ---
        reset_matrix(matrix_old, W, H);
        reset_matrix(matrix_new, W, H);

        auto start_seq = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < NUM_ITERS; ++i) {
            compute_stencil_sequencial(W, H, matrix_old, matrix_new);
            std::swap(matrix_old, matrix_new);
        }
        auto end_seq = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff_seq = end_seq - start_seq;
        total_time_seq += diff_seq.count();

        // Guardamos el valor central para verificar que la paralela da el mismo resultado
        float control_value = matrix_old[(H/2)*W + (W/2)];

        // --- 2. EJECUCIÓN PARALELA ---
        reset_matrix(matrix_old, W, H);
        reset_matrix(matrix_new, W, H);

        auto start_par = std::chrono::high_resolution_clock::now();
        #pragma omp parallel
        for(int i = 0; i < NUM_ITERS; ++i) {
            compute_stencil_parallel(W, H, matrix_old, matrix_new);
            #pragma omp single
            {
                std::swap(matrix_old, matrix_new);
            }
        }
        auto end_par = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff_par = end_par - start_par;
        total_time_par += diff_par.count();

        // Imprimimos la línea de esta ejecución (separada por tabulación)
        std::cout << std::fixed << std::setprecision(5) << diff_seq.count() << "\t" << diff_par.count() << std::endl;

        // Validamos por seguridad computacional
        if (std::abs(matrix_old[(H/2)*W + (W/2)] - control_value) > 1e-4) {
            std::cerr << "¡ADVERTENCIA! Discrepancia en los resultados en el test " << test+1 << std::endl;
        }
    }

    // --- 3. CÁLCULO DE MÉTRICAS ---
    std::cout << "-----------------------------------" << std::endl;
    double avg_seq = total_time_seq / NUM_TESTS;
    double avg_par = total_time_par / NUM_TESTS;
    
    // El Speedup (S) es el tiempo secuencial dividido por el tiempo paralelo.
    // Ej: Si S = 4.0, la versión paralela es 4 veces más rápida (400%).
    double speedup = avg_seq / avg_par;
    double speedup_percentage = speedup * 100.0;

    std::cout << std::setprecision(5);
    std::cout << "Media Secuencial : " << avg_seq << " s\n";
    std::cout << "Media Paralela   : " << avg_par << " s\n";
    std::cout << "\nRESULTADOS FINALES:\n";
    std::cout << "Factor Speedup   : " << std::setprecision(2) << speedup << "x\n";
    std::cout << "Porcentaje Speedup: " << speedup_percentage << "%\n";

    return 0;
}