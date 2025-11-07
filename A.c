#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <stdbool.h>
#include <omp.h> 


#define MY_RAND_MAX 0x7FFFFFFF
#define RANDNUM_W 521288629
#define RANDNUM_Z 362436069

unsigned int randum_w = RANDNUM_W;
unsigned int randum_z = RANDNUM_Z;

void srandnum(int seed) {
    unsigned int w, z;
    w = (seed * 104623) & 0xffffffff;
    randum_w = (w) ? w : RANDNUM_W;
    z = (seed * 48947) & 0xffffffff;
    randum_z = (z) ? z : RANDNUM_Z;
}

unsigned int randnum(void) {
    unsigned int u;
    randum_z = 36969 * (randum_z & 65535) + (randum_z >> 16);
    randum_w = 18000 * (randum_w & 65535) + (randum_w >> 16);
    u = (randum_z << 16) + randum_w;
    return (u);
}

// begin
#define MAX_CITIES 1000
#define MAX_ANTS   200

const double ALPHA = 1.0;
const double BETA  = 5.0;
const double RHO   = 0.5;
const double Q     = 100.0;

typedef struct {
    int path[MAX_CITIES];
    bool visited[MAX_CITIES];
    double cost;
} Ant;

static inline double urand01(void) {
    return (double)randnum() / ((double)MY_RAND_MAX + 1.0);
}


void generate_distance_matrix(double **dist, int n) {
    srandnum(0);
    for (int i = 0; i < n; i++)
    {
        for (int j = i; j < n; j++)
        {
            if (i == j)
            {
                dist[i][j] = 0.0;
            }
            else
            {
                double d = 10.0 + urand01() * 90.0;
                dist[i][j] = dist[j][i] = d;
            }
        }
    }
}

void initialize_pheromones(double **pher, int n) {
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            pher[i][j] = 1.0;
        }
    }
}


double path_cost(const int *path, double **dist, int n) {
    double cost = 0.0;
    for (int i = 0; i < n - 1; i++)
    {
        cost += dist[path[i]][path[i + 1]];
    }
    cost += dist[path[n - 1]][path[0]];
    return cost;
}


void construct_solution(Ant *ant, double **dist, double **pher, int n, 
                        int start_city, double* ant_r_values) {
    for (int i = 0; i < n; i++)
    {
        ant->visited[i] = false;
        ant->path[i] = -1;
    }

    int start = start_city;
    ant->path[0] = start;
    ant->visited[start] = true;

    for (int step = 1; step < n; step++)
    {
        int from = ant->path[step - 1];

        double sum_prob = 0.0;
        double prob[MAX_CITIES];
        for (int j = 0; j < n; j++)
        {
            if (!ant->visited[j])
            {
                double tau = pow(pher[from][j], ALPHA);
                double eta = pow(1.0 / dist[from][j], BETA);
                prob[j] = tau * eta;
                sum_prob += prob[j];
            } else
            {
                prob[j] = 0.0;
            }
        }

        double r = ant_r_values[step - 1] * sum_prob;

        double cum = 0.0;
        int next = -1;
        for (int j = 0; j < n; j++)
        {
            cum += prob[j];
            if (r <= cum && !ant->visited[j])
            {
                next = j;
                break;
            }
        }
        if (next == -1)
        {
            for (int j = 0; j < n; j++)
            {
                if (!ant->visited[j])
                {
                    next = j;
                    break;
                }
            }
        }

        ant->path[step] = next;
        ant->visited[next] = true;
    }

    ant->cost = path_cost(ant->path, dist, n);
}

void evaporate_pheromones(double **pher, int n) {
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            pher[i][j] *= (1.0 - RHO);
        }
    }
}

void deposit_pheromones(Ant *ants, double **pher, int num_ants, int n) {
    for (int k = 0; k < num_ants; k++)
    {
        double contrib = Q / ants[k].cost;
        for (int i = 0; i < n - 1; i++)
        {
            int a = ants[k].path[i];
            int b = ants[k].path[i + 1];
            pher[a][b] += contrib;
            pher[b][a] += contrib;
        }
        int a = ants[k].path[n - 1];
        int b = ants[k].path[0];
        pher[a][b] += contrib;
        pher[b][a] += contrib;
    }
}


double signature(const int *best_path, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++)
    {
        s += (i + 1) * (best_path[i] + 1);
    }
    return s;
}

int main(int argc, char **argv) {

    int n;
    int num_ants;
    int iterations;
    fscanf(stdin, "%d %d %d", &n, &num_ants, &iterations);

    double **dist = malloc(n * sizeof(double *));
    double **pher = malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++)
    {
        dist[i] = malloc(n * sizeof(double));
        pher[i] = malloc(n * sizeof(double));
    }

    Ant *ants = malloc(num_ants * sizeof(Ant));

    generate_distance_matrix(dist, n);
    initialize_pheromones(pher, n);


    int total_starts = iterations * num_ants;
    int* start_cities = malloc(total_starts * sizeof(int));
    
    long total_r_values = (long)iterations * num_ants * (n - 1);
    double* r_values = malloc(total_r_values * sizeof(double));

    srandnum(42);

    int s_idx = 0;
    long r_idx = 0;
    for (int i = 0; i < iterations; i++) {
        for (int k = 0; k < num_ants; k++) {
            start_cities[s_idx++] = randnum() % n;
            for (int s = 1; s < n; s++) {
                r_values[r_idx++] = urand01();
            }
        }
    }

    double best_cost = DBL_MAX;
    int best_path[MAX_CITIES];

    for (int iter = 0; iter < iterations; iter++)
    {
        
        #pragma omp parallel for
        for (int k = 0; k < num_ants; k++)
        {
            int s_idx_ant = iter * num_ants + k;
            long r_idx_ant = (long)s_idx_ant * (n - 1);

            construct_solution(&ants[k], dist, pher, n, 
                               start_cities[s_idx_ant], 
                               &r_values[r_idx_ant]);
        }
       


        for (int k = 0; k < num_ants; k++)
        {
            if (ants[k].cost < best_cost)
            {
                best_cost = ants[k].cost;
                for (int i = 0; i < n; i++)
                {
                    best_path[i] = ants[k].path[i];
                }
            }
        }

        evaporate_pheromones(pher, n);
        deposit_pheromones(ants, pher, num_ants, n);
    }
    printf("Best cost: %.4f\n", best_cost);
    printf("Signature: %.4e\n", signature(best_path, n));

    for (int i = 0; i < n; i++)
    {
        free(dist[i]);
        free(pher[i]);
    }
    free(dist);
    free(pher);
    free(ants);
    free(start_cities); 
    free(r_values);     

    return 0;
}