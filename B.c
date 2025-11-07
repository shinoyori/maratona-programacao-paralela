#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define MAX 100

typedef struct {
    int positions[MAX];
    int size;
} Solution;

typedef struct {
    Solution *solutions;
    int count;
    int capacity;
} SolutionList;

void init_solution_list(SolutionList *list) {
    list->capacity = 1000;
    list->solutions = (Solution*)malloc(list->capacity * sizeof(Solution));
    list->count = 0;
}

void add_solution(SolutionList *list, int *positions, int n) {
    if (list->count >= list->capacity) {
        list->capacity *= 2;
        list->solutions = (Solution*)realloc(list->solutions, list->capacity * sizeof(Solution));
    }
    memcpy(list->solutions[list->count].positions, positions, n * sizeof(int));
    list->solutions[list->count].size = n;
    list->count++;
}

void solve_recursive(int row, int n, int *d_col, int *d_diag1, int *d_diag2, 
                     int *stack, SolutionList *list) {
    if (row == n) {
        add_solution(list, stack, n);
        return;
    }
    
    for (int col = 0; col < n; col++) {
        if (!d_col[col] && !d_diag1[row + col] && !d_diag2[row - col + n]) {
            stack[row] = col;
            d_col[col] = 1;
            d_diag1[row + col] = 1;
            d_diag2[row - col + n] = 1;
            
            solve_recursive(row + 1, n, d_col, d_diag1, d_diag2, stack, list);
            
            d_col[col] = 0;
            d_diag1[row + col] = 0;
            d_diag2[row - col + n] = 0;
        }
    }
}

void solve_from_first_row(int first_col, int n, SolutionList *list) {
    int d_col[MAX] = {0};
    int d_diag1[MAX*2] = {0};
    int d_diag2[MAX*2] = {0};
    int stack[MAX];
    
    stack[0] = first_col;
    d_col[first_col] = 1;
    d_diag1[first_col] = 1;
    d_diag2[n] = 1;
    
    solve_recursive(1, n, d_col, d_diag1, d_diag2, stack, list);
}

int compare_solutions(const void *a, const void *b) {
    Solution *s1 = (Solution*)a;
    Solution *s2 = (Solution*)b;
    
    for (int i = 0; i < s1->size && i < s2->size; i++) {
        if (s1->positions[i] != s2->positions[i]) {
            return s1->positions[i] - s2->positions[i];
        }
    }
    return 0;
}

void SolveProblem(int n) {
    SolutionList *thread_lists;
    int num_threads;
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            num_threads = omp_get_num_threads();
            thread_lists = (SolutionList*)malloc(num_threads * sizeof(SolutionList));
            for (int i = 0; i < num_threads; i++) {
                init_solution_list(&thread_lists[i]);
            }
        }
    }
    
    #pragma omp parallel for schedule(dynamic)
    for (int first_col = 0; first_col < n; first_col++) {
        int tid = omp_get_thread_num();
        solve_from_first_row(first_col, n, &thread_lists[tid]);
    }
    
    int total_count = 0;
    for (int i = 0; i < num_threads; i++) {
        total_count += thread_lists[i].count;
    }
    
    Solution *all_solutions = (Solution*)malloc(total_count * sizeof(Solution));
    int offset = 0;
    for (int i = 0; i < num_threads; i++) {
        memcpy(&all_solutions[offset], thread_lists[i].solutions, 
               thread_lists[i].count * sizeof(Solution));
        offset += thread_lists[i].count;
        free(thread_lists[i].solutions);
    }
    free(thread_lists);
    
    qsort(all_solutions, total_count, sizeof(Solution), compare_solutions);
    
    for (int i = 0; i < total_count; i++) {
        printf("SOLUTION:");
        for (int j = 0; j < all_solutions[i].size; j++) {
            printf(" (%d,%d)", j + 1, all_solutions[i].positions[j] + 1);
        }
        printf("\n");
    }
    
    free(all_solutions);
}

int main(int argc, char **argv) {
    int N;
    scanf("%d", &N);
    SolveProblem(N);
    return 0;
}