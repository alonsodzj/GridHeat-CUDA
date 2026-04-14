#include <vector>
#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm> // Para std::swap

void print_matrix(const std::vector<float>& matrix, const int W, const int H) {
    std::cout << "--- Matrix State ---" << std::endl;
    for (int i = 0; i < H; ++i) { // i suele ser la fila (H)
        for (int j = 0; j < W; ++j) { // j suele ser la columna (W)
            std::cout << std::setw(8) << matrix[i * W + j]; 
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
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

    int DIM_N = 0;
    int NUM_ITERS = 0;

    try {
        DIM_N = std::stoi(argv[1]);
        NUM_ITERS = std::stoi(argv[2]);
    } catch (...) {
        std::cerr << "Error: Argumentos inválidos." << std::endl;
        return 1;
    }

    // Usamos const, no constexpr, porque se definen en ejecución
    const int W = DIM_N; 
    const int H = DIM_N; // Asumimos matriz cuadrada segun tu DIM_N*DIM_N
    const int DIM = W * H;

    std::vector<float> matrix_old(DIM, 0.0f);
    std::vector<float> matrix_new(DIM, 0.0f);

    // Inicializar borde superior
    for(int j = 0; j < W; ++j) {
        matrix_old[j] = 100.0f;
        matrix_new[j] = 100.0f; 
    }
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Dimensiones: " << W << "x" << H << " (" << NUM_ITERS << " iters)" << std::endl;

    // Proceso iterativo
    for(int i = 0; i < NUM_ITERS; ++i) {
        compute_stencil_sequencial(W, H, matrix_old, matrix_new);
        
        // Intercambiamos buffers: el nuevo ahora es el viejo para la siguiente vuelta
        std::swap(matrix_old, matrix_new);
    }

    // Imprimimos el resultado final
    print_matrix(matrix_old, W, H);

    return 0;
}