
# CuHeat-Simulation: Difusión de Calor 2D 🔥

Este repositorio contiene la implementación de un simulador de difusión de calor en una placa bidimensional. [cite_start]El proyecto evoluciona desde una versión secuencial pura (baseline) hasta implementaciones de alto rendimiento utilizando memoria compartida (**OpenMP**), memoria distribuida (**MPI**) y aceleración por hardware (**CUDA**).

---

## 📐 El Modelo Físico: Stencil de 5 Puntos

[cite_start]La simulación calcula la temperatura de un punto $(i, j)$ en el instante $t+1$ como la media aritmética de sus cuatro vecinos directos (arriba, abajo, izquierda, derecha) en el instante $t$:

$$T_{i,j}^{t+1}=\frac{T_{i-1,j}^{t}+T_{i+1,j}^{t}+T_{i,j-1}^{t}+T_{i,j+1}^{t}}{4}$$

### Condiciones de Contorno
[cite_start]Se aplican límites constantes al borde de la placa para permitir la difusión:
* [cite_start]**Borde Superior:** $100.0^\circ C$.
* [cite_start]**Resto de Bordes (Inferior, Izquierdo, Derecho):** $0.0^\circ C$.
* [cite_start]Los bordes son constantes y no se actualizan durante el cómputo.

---

## 📂 Estructura del Proyecto

[cite_start]El código está organizado de forma incremental siguiendo los requerimientos de la asignatura:

* [cite_start]`src/baseline/`: Implementación secuencial en C/C++ con matriz linealizada y doble buffer.
* [cite_start]`src/openmp/`: Paralelización de bucles evaluando el impacto del *scheduling* y evitando el *false sharing*.
* [cite_start]`src/mpi/`: Descomposición de dominio por filas con intercambio de halos y comunicaciones no bloqueantes.
* [cite_start]`src/cuda/`: Aceleración en GPU con accesos coalescentes y uso de **Shared Memory**.

---

## 🚀 Ejecución y Control

[cite_start]El programa acepta los siguientes parámetros por línea de comandos:
1. [cite_start]**N:** Dimensión de la matriz (ej. 2048).
2. [cite_start]**Iteraciones:** Número de pasos de tiempo a simular.

Ejemplo: `./calor 2048 1000`

---

## 📊 Análisis para el Examen Técnico
[cite_start]*(Sección para anotar métricas personales obtenidas durante las pruebas)* 

## Project structure

CuHeat-Project/
├── src/
│   ├── baseline/          # Código C/C++ secuencial puro 
│   ├── openmp/            # Versión con directivas pragma 
│   ├── mpi/               # Versión con descomposición de dominio 
│   ├── cuda/              # Versión con Kernels y Shared Memory 
│   └── hybrid/            # Implementación final combinada 
├── include/               # Cabeceras comunes (constantes de bordes)
├── scripts/               # Scripts para medir métricas y graficar
├── Makefile               # Automatización de la compilación de todas las versiones
└── README.md              # Documentación técnica para el examen