
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

static inline int pow2(int p) {
    return 1 << p;
}

int load(double * restrict a, double * restrict b, double * restrict c,
         double * restrict d, int dim_sist)
{
    for (int i = 0; i < dim_sist; i++)
       if(fscanf(stdin, "%lf", &a[i]) != 1) return -1;
    for (int i = 0; i < dim_sist; i++)
       if(fscanf(stdin, "%lf", &b[i]) != 1) return -1;
    for (int i = 0; i < dim_sist; i++)
       if(fscanf(stdin, "%lf", &c[i]) != 1) return -1;
    for (int i = 0; i < dim_sist; i++)
       if(fscanf(stdin, "%lf", &d[i]) != 1) return -1;
    return 0;
}

void alg_rc(double * restrict a, double * restrict b, double * restrict c,
            double * restrict d, double * restrict x, int p, int dim_sist)
{
    const int num_iter = p - 1;
    int num_eq_iter = dim_sist >> 1;
    int space = 1;
    int var;

    // Reduction
    for (int j = 0; j < num_iter; j++)
    {
        var = space;
        space <<= 1;
        
        if (num_eq_iter >= 64) {
            #pragma omp parallel for schedule(static) if(num_eq_iter >= 64)
            for (int l = 0; l < num_eq_iter; l++)
            {
                int i = (space * l) + space - 1;
                int i_esq = i - var;
                int i_dir = i + var;
                
                if (i_dir >= dim_sist) i_dir = dim_sist - 1;
                
                
                double b_esq = b[i_esq];
                double b_dir = b[i_dir];
                double a_esq = a[i_esq];
                double c_dir = c[i_dir];
                double c_esq = c[i_esq];
                double a_dir = a[i_dir];
                double d_esq = d[i_esq];
                double d_dir = d[i_dir];
                
                double tmp1 = a[i] / b_esq;
                double tmp2 = c[i] / b_dir;
                
                b[i] = b[i] - c_esq * tmp1 - a_dir * tmp2;
                d[i] = d[i] - d_esq * tmp1 - d_dir * tmp2;
                a[i] = -a_esq * tmp1;
                c[i] = -c_dir * tmp2;
            }
        } else {
            for (int l = 0; l < num_eq_iter; l++)
            {
                int i = (space * l) + space - 1;
                int i_esq = i - var;
                int i_dir = i + var;
                
                if (i_dir >= dim_sist) i_dir = dim_sist - 1;
                
                double b_esq = b[i_esq];
                double b_dir = b[i_dir];
                double tmp1 = a[i] / b_esq;
                double tmp2 = c[i] / b_dir;
                
                b[i] = b[i] - c[i_esq] * tmp1 - a[i_dir] * tmp2;
                d[i] = d[i] - d[i_esq] * tmp1 - d[i_dir] * tmp2;
                a[i] = -a[i_esq] * tmp1;
                c[i] = -c[i_dir] * tmp2;
            }
        }
        num_eq_iter >>= 1;
    }

    const int i1 = space - 1;
    const int i2 = (space << 1) - 1;
    
    double det = b[i2] * b[i1] - c[i1] * a[i2];
    double inv_det = 1.0 / det;
    
    x[i1] = (b[i2] * d[i1] - c[i1] * d[i2]) * inv_det;
    x[i2] = (d[i2] * b[i1] - d[i1] * a[i2]) * inv_det;

    int num_sol_iter = 2;
    for (int j = 0; j < num_iter; j++)
    {
        var = space >> 1;
        
        int i = (space >> 1) - 1;
        x[i] = (d[i] - c[i] * x[i + var]) / b[i];
        
        if (num_sol_iter > 64) {
            #pragma omp parallel for schedule(static) if(num_sol_iter > 64)
            for (int l = 1; l < num_sol_iter; l++)
            {
                int i = (space * l) + (space >> 1) - 1;
                x[i] = (d[i] - a[i] * x[i - var] - c[i] * x[i + var]) / b[i];
            }
        } else {
            for (int l = 1; l < num_sol_iter; l++)
            {
                int i = (space * l) + (space >> 1) - 1;
                x[i] = (d[i] - a[i] * x[i - var] - c[i] * x[i + var]) / b[i];
            }
        }
        
        space >>= 1;
        num_sol_iter <<= 1;
    }
}

int main(void) {
    int p;
    
    if(fscanf(stdin, "%d", &p) != 1) {
        fprintf(stderr, "Input error\n");
        return -1;
    }
    
    const int dim_sist = pow2(p);

    double* a = (double*) malloc(dim_sist * sizeof(double));
    double* b = (double*) malloc(dim_sist * sizeof(double));
    double* c = (double*) malloc(dim_sist * sizeof(double));
    double* d = (double*) malloc(dim_sist * sizeof(double));
    double* x = (double*) malloc(dim_sist * sizeof(double));
    
    if ((a == NULL) || (b == NULL) || (c == NULL) || (d == NULL) || (x == NULL))
    {
        fprintf(stderr, "Memory allocation error\n");
        free(a); free(b); free(c); free(d); free(x);
        return 1;
    }
        
    if(load(a, b, c, d, dim_sist))
    {
        fprintf(stderr, "Load error\n");
        free(a); free(b); free(c); free(d); free(x);
        return 1;
    }
    
    alg_rc(a, b, c, d, x, p, dim_sist);
    
    for (int i = 0; i < dim_sist; i++)
    {
        printf("%lf ", x[i]);
    }
    printf("\n");
    
    free(a);
    free(b);
    free(c);
    free(d);
    free(x);
    
    return 0;
}
