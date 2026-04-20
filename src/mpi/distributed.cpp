//tengo que generar un esquema que me permita distribuir el cómputo de una imagen entre diferentes nodos
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm> // Para std::swap
#include <chrono>    // Para medir el tiempo que tarda
#include <mpi.h>