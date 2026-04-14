#include <vector>

//estos define me van a permitir depurar para diferentes tamaños de mi matriz tanto cuadradas como no.
#define MATRIX_N_DIM 256    //dimensión cuadrada para mi matriz
#define MATRIX_W_DIM 256    //dimensión horizontal para mi matriz
#define MATRIX_H_DIM 256    //dimensión vertical de mi matriz

//número de iteraciones a ejecutar (ahora no depende de la calidad de la solución)
#define NUM_ITER = 100

int main()
{
    constexpr int W = MATRIX_N_DIM;
    constexpr int H = MATRIX_N_DIM;
    constexpr int DIM = W*H;

    //1. Creo la matriz inicial con std::vector
    std::vector<float> matrix(DIM);

    //2. Relleno la matriz inicial para que tenga la placa a 100 grados y los bordes a 0 ya que serán aislantes.
    for(int i = 0;i < DIM;++i){
        if(i<W) //me encuentro en la primera línea relleno con 100 ya que es la placa
        {
            matrix[i] = 100;
        }else matrix[i] = 0;
    }
    //en este punto ya tengo la matriz inicializada

    //3. creo las copias de mi matriz para poder actualizar correctamente
    std::vector<float> matrix_old(std::size(matrix));
    std::vector<float> matrix_new(std::size(matrix));
}