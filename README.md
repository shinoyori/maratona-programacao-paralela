# Maratona de Programação: Algoritmos Paralelos em OpenMP

Este repositório contém as soluções desenvolvidas para uma maratona/competição de programação com foco em **Computação Paralela** usando a API **OpenMP** (Open Multi-Processing). Os problemas consistem na implementação e otimização de algoritmos clássicos para alcançar o máximo de ganho de desempenho (*speedup*).

## Estrutura do Repositório

| Arquivo | Problema | Descrição Breve |
| :--- | :--- | :--- |
| `Maratona/A.c` | A | Otimização por Colônia de Formigas (ACO) para o Problema do Caixeiro Viajante (TSP). |
| `Maratona/B.c` | B | O Problema das N-Rainhas (N-Queens) utilizando busca com *backtracking*. |
| `Maratona/C.c` | C | Simulação de Propagação de Onda Acústica 3D por Diferenças Finitas (FDM). |
| `Maratona/D.c` | D | Solução de Sistema Linear Tridiagonal via Redução Cíclica Recorrente (RC). |

## Detalhes das Soluções e Paralelização

Todas as soluções utilizam a diretiva `#pragma omp` para paralelização em memória compartilhada.

### Problema A: Travelling Salesperson Problem (TSP) com ACO

* **Algoritmo:** Ant Colony Optimization (ACO).
* **Paralelização:** A fase de **construção da solução** para cada formiga em uma iteração é paralelizada. Cada formiga constrói seu caminho de forma independente, permitindo que a iteração avance mais rapidamente.

### Problema B: N-Queens

* **Algoritmo:** Busca em profundidade (*backtracking*) para encontrar todas as configurações válidas.
* **Paralelização:** O problema é decomposto em subproblemas independentes, onde o **paralelismo é aplicado no loop inicial** que itera sobre a coluna de partida na primeira linha do tabuleiro. Cada thread explora o espaço de busca a partir de um ponto de partida distinto.

### Problema C: Simulação de Onda Acústica 3D (FDM)

* **Algoritmo:** Solução da equação de onda acústica 3D por meio do método de Diferenças Finitas (FDM) de alta ordem.
* **Paralelização:** O **kernel central de atualização do campo de onda (`kernel_CPU_06_mod_3DRhoCte`)** é paralelizado. A maior parte do tempo de computação é gasta neste loop triplamente aninhado (i, j, k), onde o paralelismo é essencial para simulações grandes.

### Problema D: Sistema Tridiagonal com Redução Cíclica Recorrente (RC)

* **Algoritmo:** Recursive Cyclic Reduction (RC) para resolver sistemas de equações lineares tridiagonais.
* **Paralelização:** As **fases de Redução e de Substituição Inversa** do algoritmo (os laços de `l`) são paralelizadas.

## Compilação e Uso (GCC/G++)

Para compilar qualquer um dos arquivos `.c` (que requerem OpenMP) usando GCC ou G++, você deve incluir a flag `-fopenmp`:

```bash
# Compilação do Problema A
gcc -o A Maratona/A.c -lm -fopenmp

# Execução (exemplo de entrada: N_CIDADES N_FORMIGAS N_ITERACOES)
./A < input_A.txt
