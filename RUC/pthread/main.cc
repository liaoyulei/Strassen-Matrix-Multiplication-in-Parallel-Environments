#include <iostream>
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#define t1 128
#define t2 4096
#define N 256
#define ITERATIONS 10
using namespace std;
typedef struct st{
	float *A, *B, *C, *T, *C1, *C2, *C11, *C12, *C21, *C22;
	int n, i, j;
}st;
float A[N * N], B[N * N], C[N * N], C_cmp[N * N];
pthread_t tid[N * N];
st stu[N * N];

void *split(void *argv) {
	float *C11 = ((st *)argv)->C11, *C12 = ((st *)argv)->C12, *C21 = ((st *)argv)->C21, *C22 = ((st *)argv)->C22, *C = ((st *)argv)->C;
	int n = ((st *)argv)->n, i = ((st *)argv)->i, j = ((st *)argv)->j;
	C11[i * n + j] = C[i * 2 * n + j];
	C12[i * n + j] = C[i * 2 * n + j + n];
	C21[i * n + j] = C[(i + n) * 2 * n + j];
	C22[i * n + j] = C[(i + n) * 2 * n + j + n];
	return NULL;
}

void *merge(void *argv) {
	float *C11 = ((st *)argv)->C11, *C12 = ((st *)argv)->C12, *C21 = ((st *)argv)->C21, *C22 = ((st *)argv)->C22, *C = ((st *)argv)->C;
	int n = ((st *)argv)->n, i = ((st *)argv)->i, j = ((st *)argv)->j;
	C[i * 2 * n + j] = C11[i * n + j];
	C[i * 2 * n + j + n] = C12[i * n + j];
	C[(i + n) *2 * n + j] = C21[i * n + j];
	C[(i + n) * 2 * n + j + n] = C22[i * n + j];
	return NULL;
}

void *add(void *argv) {
	float *A = ((st *)argv)->A, *B = ((st *)argv)->B, *C = ((st *)argv)->C;
	int n = ((st *)argv)->n, i = ((st *)argv)->i, j = ((st *)argv)->j;
	C[i * n + j] = A[i * n + j] + B[i * n + j];
	return NULL;
}

void *sub(void *argv) {
	float *A = ((st *)argv)->A, *B = ((st *)argv)->B, *C = ((st *)argv)->C;
	int n = ((st *)argv)->n, i = ((st *)argv)->i, j = ((st *)argv)->j;
	C[i * n + j] = A[i * n + j] - B[i * n + j];
	return NULL;
}

void *mul(void *argv) {
	float *A = ((st *)argv)->A, *B = ((st *)argv)->B, *C = ((st *)argv)->C;
	int n = ((st *)argv)->n, i = ((st *)argv)->i, j = ((st *)argv)->j;
	C[i * n + j] = 0;
	for(int k = 0; k < n; k++) {
		C[i * n + j] += A[i * n + k] * B[k * n + j];
	}
	return NULL;
}

void *mul_add(void *argv) {
	float *A = ((st *)argv)->A, *B = ((st *)argv)->B, *T= ((st *)argv)->T, *C = ((st *)argv)->C;
	int n = ((st *)argv)->n, i = ((st *)argv)->i, j = ((st *)argv)->j;
	C[i * n + j] = T[i * n + j];
	for(int k = 0; k < n; k++) {
		C[i * n + j] += A[i * n + k] * B[k * n + j];
	}
	return NULL;
}

void *mul_sub_inc(void *argv) {
	float *A = ((st *)argv)->A, *B = ((st *)argv)->B, *T = ((st *)argv)->T, *C1 = ((st *)argv)->C1, *C2 = ((st *)argv)->C2;
	int n = ((st *)argv)->n, i = ((st *)argv)->i, j = ((st *)argv)->j;
	C1[i * n + j] = 0;
	for(int k = 0; k < n; k++) {
		C1[i * n + j] += A[i * n + k] * B[k * n + j];
	}
	C1[i * n + j] = T[i * n + j] - C1[i * n + j];
	C2[i * n + j] += T[i * n + j];
	return NULL;
}

void *mul_inc_inc_inc(void *argv) {
	float *A = ((st *)argv)->A, *B = ((st *)argv)->B, *C = ((st *)argv)->C, *T = ((st *)argv)->T, *C1 = ((st *)argv)->C1, *C2 = ((st *)argv)->C2;
	int n = ((st *)argv)->n, i = ((st *)argv)->i, j = ((st *)argv)->j;
	C[i * n + j] = 0;
	for(int k = 0; k < n; k++) {
		C[i * n + j] += A[i * n + k] * B[k * n + j];
	}
	C1[i * n + j] += C[i * n + j];
	C2[i * n + j] += C1[i * n + j];
	C1[i * n + j] += T[i * n + j];
	return NULL;
}

void strassen(float *A, float *B, float *C, int n) {
	if(n <= t1) {
		for(int j = 0; j < n; j++) {
			for(int i = 0; i < n; i++) {
				stu[i * N + j].A = A;
				stu[i * N + j].B = B;
				stu[i * N + j].C = C;
				stu[i * N + j].n = n;
				pthread_create(&tid[i * N + j], NULL, mul, &stu[i * N + j]);
			}
		}
		for(int i = 0; i < n; i++) {
			for(int j = 0; j < n; j++) {
				pthread_join(tid[i * N +j], NULL);
			}
		}
	}
	else{
		int m = n >> 1;
		float *A11 = (float *)malloc(sizeof(float) * m * m);
		float *A12 = (float *)malloc(sizeof(float) * m * m);
		float *A21 = (float *)malloc(sizeof(float) * m * m);
		float *A22 = (float *)malloc(sizeof(float) * m * m);
		float *B11 = (float *)malloc(sizeof(float) * m * m);
		float *B12 = (float *)malloc(sizeof(float) * m * m);
		float *B21 = (float *)malloc(sizeof(float) * m * m);
		float *B22 = (float *)malloc(sizeof(float) * m * m);
		float *C11 = (float *)malloc(sizeof(float) * m * m);
		float *C12 = (float *)malloc(sizeof(float) * m * m);
		float *C21 = (float *)malloc(sizeof(float) * m * m);
		float *C22 = (float *)malloc(sizeof(float) * m * m);
		float *T1 = (float *)malloc(sizeof(float) * m * m);
		float *T2 = (float *)malloc(sizeof(float) * m * m);
		if(n <= t2) {
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					stu[i * N + j].C11 = A11;
					stu[i * N + j].C12 = A12;
					stu[i * N + j].C21 = A21;
					stu[i * N + j].C22 = A22;
					stu[i * N + j].C = A;
					stu[i * N + j].n = m;
					pthread_create(&tid[i * N + j], NULL, split, &stu[i * N + j]);
				}
			}
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					pthread_join(tid[i * N + j], NULL);
				}
			}
			
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					stu[i * N + j].C11 = B11;
					stu[i * N + j].C12 = B12;
					stu[i * N + j].C21 = B21;
					stu[i * N + j].C22 = B22;
					stu[i * N + j].C = B;
					stu[i * N + j].n = m;
					pthread_create(&tid[i * N + j], NULL, split, &stu[i * N + j]);
				}
			}
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					pthread_join(tid[i * N + j], NULL);
				}
			}
			
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					stu[i * N + j].A = A11;
					stu[i * N + j].B = A21;
					stu[i * N + j].C = T1;
					stu[i * N + j].n = m;
					pthread_create(&tid[i * N + j], NULL, sub, &stu[i * N + j]);
				}
			}
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					pthread_join(tid[i * N + j], NULL);
				}
			}
			
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					stu[i * N + j].A = B22;
					stu[i * N + j].B = B12;
					stu[i * N + j].C = T2;
					stu[i * N + j].n = m;
					pthread_create(&tid[i * N + j], NULL, sub, &stu[i * N + j]);
				}
			}
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					pthread_join(tid[i * N + j], NULL);
				}
			}

			for(int j = 0; j < m; j++) {
				for(int i = 0; i < m; i++) {
					stu[i * N + j].A = T1;
					stu[i * N + j].B = T2;
					stu[i * N + j].C = C21;
					stu[i * N + j].n = m;
					pthread_create(&tid[i * N + j], NULL, mul, &stu[i * N + j]);
				}
			}
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					pthread_join(tid[i * N + j], NULL);
				}
			}

			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					stu[i * N + j].A = A21;
					stu[i * N + j].B = A22;
					stu[i * N + j].C = T1;
					stu[i * N + j].n = m;
					pthread_create(&tid[i * N + j], NULL, add, &stu[i * N + j]);
				}
			}
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					pthread_join(tid[i * N + j], NULL);
				}
			}

			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					stu[i * N + j].A = B12;
					stu[i * N + j].B = B11;
					stu[i * N + j].C = T2;
					stu[i * N + j].n = m;
					pthread_create(&tid[i * N + j], NULL, sub, &stu[i * N + j]);
				}
			}
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					pthread_join(tid[i * N + j], NULL);
				}
			}

			for(int j = 0; j < m; j++) {
				for(int i = 0; i < m; i++) {
					stu[i * N + j].A = T1;
					stu[i * N + j].B = T2;
					stu[i * N + j].C = C22;
					stu[i * N + j].n = m;
					pthread_create(&tid[i * N + j], NULL, mul, &stu[i * N + j]);
				}
			}
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					pthread_join(tid[i * N + j], NULL);
				}
			}
			
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					stu[i * N + j].A = T1;
					stu[i * N + j].B = A11;
					stu[i * N + j].C = T1;
					stu[i * N + j].n = m;
					pthread_create(&tid[i * N + j], NULL, sub, &stu[i * N + j]);
				}
			}
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					pthread_join(tid[i * N + j], NULL);
				}
			}

			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					stu[i * N + j].A = B22;
					stu[i * N + j].B = T2;
					stu[i * N + j].C = T2;
					stu[i * N + j].n = m;
					pthread_create(&tid[i * N + j], NULL, sub, &stu[i * N + j]);
				}
			}
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					pthread_join(tid[i * N + j], NULL);
				}
			}
			
			for(int j = 0; j < m; j++) {
				for(int i = 0; i < m; i++) {
					stu[i * N + j].A = T1;
					stu[i * N + j].B = T2;
					stu[i * N + j].C = C11;
					stu[i * N + j].n = m;
					pthread_create(&tid[i * N + j], NULL, mul, &stu[i * N + j]);
				}
			}
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					pthread_join(tid[i * N + j], NULL);
				}
			}

			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					stu[i * N + j].A = A12;
					stu[i * N + j].B = T1;
					stu[i * N + j].C = T1;
					stu[i * N + j].n = m;
					pthread_create(&tid[i * N + j], NULL, sub, &stu[i * N + j]);
				}
			}
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					pthread_join(tid[i * N + j], NULL);
				}
			}

			for(int j = 0; j < m; j++) {
				for(int i = 0; i < m; i++) {
					stu[i * N + j].A = T1;
					stu[i * N + j].B = B22;
					stu[i * N + j].T = C22;
					stu[i * N + j].C = C12;
					stu[i * N + j].n = m;
					pthread_create(&tid[i * N + j], NULL, mul_add, &stu[i * N + j]);
				}
			}
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					pthread_join(tid[i * N + j], NULL);
				}
			}
			
			for(int j = 0; j < m; j++) {
				for(int i = 0; i < m; i++) {
					stu[i * N + j].A = A11;
					stu[i * N + j].B = B11;
					stu[i * N + j].C = T1;
					stu[i * N + j].T = C21;
					stu[i * N + j].C1 = C11;
					stu[i * N + j].C2 = C12;
					stu[i * N + j].n = m;
					pthread_create(&tid[i * N + j], NULL, mul_inc_inc_inc, &stu[i * N + j]);
				}
			}
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					pthread_join(tid[i * N + j], NULL);
				}
			}

			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					stu[i * N + j].A = T2;
					stu[i * N + j].B = B21;
					stu[i * N + j].C = T2;
					stu[i * N + j].n = m;
					pthread_create(&tid[i * N + j], NULL, sub, &stu[i * N + j]);
				}
			}
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					pthread_join(tid[i * N + j], NULL);
				}
			}

			for(int j = 0; j < m; j++) {
				for(int i = 0; i < m; i++) {
					stu[i * N + j].A = A22;
					stu[i * N + j].B = T2;
					stu[i * N + j].T = C11;
					stu[i * N + j].C1 = C21;
					stu[i * N + j].C2 = C22;
					stu[i * N + j].n = m;
					pthread_create(&tid[i * N + j], NULL, mul_sub_inc, &stu[i * N + j]);
				}
			}
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					pthread_join(tid[i * N + j], NULL);
				}
			}

			for(int j = 0; j < m; j++) {
				for(int i = 0; i < m; i++) {
					stu[i * N + j].A = A12;
					stu[i * N + j].B = B21;
					stu[i * N + j].T = T1;
					stu[i * N + j].C = C11;
					stu[i * N + j].n = m;
					pthread_create(&tid[i * N + j], NULL, mul_add, &stu[i * N + j]);
				}
			}
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					pthread_join(tid[i * N + j], NULL);
				}
			}

			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					stu[i * N + j].C11 = C11;
					stu[i * N + j].C12 = C12;
					stu[i * N + j].C21 = C21;
					stu[i * N + j].C22 = C22;
					stu[i * N + j].C = C;
					stu[i * N + j].n = m;
					pthread_create(&tid[i * N + j], NULL, merge, &stu[i * N + j]);
				}
			}
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					pthread_join(tid[i * N + j], NULL);
				}
			}
		}
		else{
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					stu[i * N + j].C11 = A11;
					stu[i * N + j].C12 = A12;
					stu[i * N + j].C21 = A21;
					stu[i * N + j].C22 = A22;
					stu[i * N + j].C = A;
					stu[i * N + j].n = m;
					pthread_create(&tid[i * N + j], NULL, split, &stu[i * N + j]);
				}
			}
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					pthread_join(tid[i * N + j], NULL);
				}
			}
			
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					stu[i * N + j].C11 = B11;
					stu[i * N + j].C12 = B12;
					stu[i * N + j].C21 = B21;
					stu[i * N + j].C22 = B22;
					stu[i * N + j].C = B;
					stu[i * N + j].n = m;
					pthread_create(&tid[i * N + j], NULL, split, &stu[i * N + j]);
				}
			}
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					pthread_join(tid[i * N + j], NULL);
				}
			}
			
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					stu[i * N + j].A = A11;
					stu[i * N + j].B = A21;
					stu[i * N + j].C = T1;
					stu[i * N + j].n = m;
					pthread_create(&tid[i * N + j], NULL, sub, &stu[i * N + j]);
				}
			}
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					pthread_join(tid[i * N + j], NULL);
				}
			}
			
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					stu[i * N + j].A = B22;
					stu[i * N + j].B = B12;
					stu[i * N + j].C = T2;
					stu[i * N + j].n = m;
					pthread_create(&tid[i * N + j], NULL, sub, &stu[i * N + j]);
				}
			}
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					pthread_join(tid[i * N + j], NULL);
				}
			}
			
			strassen(T1, T2, C21, m);

			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					stu[i * N + j].A = A21;
					stu[i * N + j].B = A22;
					stu[i * N + j].C = T1;
					stu[i * N + j].n = m;
					pthread_create(&tid[i * N + j], NULL, add, &stu[i * N + j]);
				}
			}
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					pthread_join(tid[i * N + j], NULL);
				}
			}

			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					stu[i * N + j].A = B12;
					stu[i * N + j].B = B11;
					stu[i * N + j].C = T2;
					stu[i * N + j].n = m;
					pthread_create(&tid[i * N + j], NULL, sub, &stu[i * N + j]);
				}
			}
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					pthread_join(tid[i * N + j], NULL);
				}
			}

			strassen(T1, T2, C22, m);
			
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					stu[i * N + j].A = T1;
					stu[i * N + j].B = A11;
					stu[i * N + j].C = T1;
					stu[i * N + j].n = m;
					pthread_create(&tid[i * N + j], NULL, sub, &stu[i * N + j]);
				}
			}
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					pthread_join(tid[i * N + j], NULL);
				}
			}	

			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					stu[i * N + j].A = B22;
					stu[i * N + j].B = T2;
					stu[i * N + j].C = T2;
					stu[i * N + j].n = m;
					pthread_create(&tid[i * N + j], NULL, sub, &stu[i * N + j]);
				}
			}
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					pthread_join(tid[i * N + j], NULL);
				}
			}

			strassen(T1, T2, C11, m);
			
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					stu[i * N + j].A = A12;
					stu[i * N + j].B = T1;
					stu[i * N + j].C = T1;
					stu[i * N + j].n = m;
					pthread_create(&tid[i * N + j], NULL, sub, &stu[i * N + j]);
				}
			}
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					pthread_join(tid[i * N + j], NULL);
				}
			}

			strassen(T1, B22, C12, m);
			
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					stu[i * N + j].A = C12;
					stu[i * N + j].B = C22;
					stu[i * N + j].C = C12;
					stu[i * N + j].n = m;
					pthread_create(&tid[i * N + j], NULL, add, &stu[i * N + j]);
				}
			}
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					pthread_join(tid[i * N + j], NULL);
				}
			}

			strassen(A11, B11, T1, m);
			
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					stu[i * N + j].A = C11;
					stu[i * N + j].B = C12;
					stu[i * N + j].C = C12;
					stu[i * N + j].n = m;
					pthread_create(&tid[i * N + j], NULL, add, &stu[i * N + j]);
				}
			}
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					pthread_join(tid[i * N + j], NULL);
				}
			}

			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					stu[i * N + j].A = C12;
					stu[i * N + j].B = T1;
					stu[i * N + j].C = C12;
					stu[i * N + j].n = m;
					pthread_create(&tid[i * N + j], NULL, add, &stu[i * N + j]);
				}
			}
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					pthread_join(tid[i * N + j], NULL);
				}
			}

			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					stu[i * N + j].A = C11;
					stu[i * N + j].B = C21;
					stu[i * N + j].C = C11;
					stu[i * N + j].n = m;
					pthread_create(&tid[i * N + j], NULL, add, &stu[i * N + j]);
				}
			}
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					pthread_join(tid[i * N + j], NULL);
				}
			}

			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					stu[i * N + j].A = C11;
					stu[i * N + j].B = T1;
					stu[i * N + j].C = C11;
					stu[i * N + j].n = m;
					pthread_create(&tid[i * N + j], NULL, add, &stu[i * N + j]);
				}
			}
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					pthread_join(tid[i * N + j], NULL);
				}
			}

			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					stu[i * N + j].A = T2;
					stu[i * N + j].B = B21;
					stu[i * N + j].C = T2;
					stu[i * N + j].n = m;
					pthread_create(&tid[i * N + j], NULL, sub, &stu[i * N + j]);
				}
			}
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					pthread_join(tid[i * N + j], NULL);
				}
			}
			
			strassen(A22, T2, C21, m);
			
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					stu[i * N + j].A = C11;
					stu[i * N + j].B = C21;
					stu[i * N + j].C = C21;
					stu[i * N + j].n = m;
					pthread_create(&tid[i * N + j], NULL, sub, &stu[i * N + j]);
				}
			}
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					pthread_join(tid[i * N + j], NULL);
				}
			}
			
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					stu[i * N + j].A = C11;
					stu[i * N + j].B = C22;
					stu[i * N + j].C = C22;
					stu[i * N + j].n = m;
					pthread_create(&tid[i * N + j], NULL, add, &stu[i * N + j]);
				}
			}
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					pthread_join(tid[i * N + j], NULL);
				}
			}
			
			strassen(A12, B21, C11, m);
			
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					stu[i * N + j].A = C11;
					stu[i * N + j].B = T1;
					stu[i * N + j].C = C11;
					stu[i * N + j].n = m;
					pthread_create(&tid[i * N + j], NULL, add, &stu[i * N + j]);
				}
			}
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					pthread_join(tid[i * N + j], NULL);
				}
			}

			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					stu[i * N + j].C11 = C11;
					stu[i * N + j].C12 = C12;
					stu[i * N + j].C21 = C21;
					stu[i * N + j].C22 = C22;
					stu[i * N + j].C = C;
					stu[i * N + j].n = m;
					pthread_create(&tid[i * N + j], NULL, merge, &stu[i * N + j]);
				}
			}
			for(int i = 0; i < m; i++) {
				for(int j = 0; j < m; j++) {
					pthread_join(tid[i * N + j], NULL);
				}
			}
		}
		free(A11); 
		free(A12); 
		free(A21); 
		free(A22); 
		free(B11); 
		free(B12); 
		free(B21); 
		free(B22); 
		free(T1);
		free(T2);	
	} 

}

void compare(float *res1, float *res2, int n) {
	int fail = 0;
	for(int i = 0; i < n; i++) {
		float a, b;
		a = res1[i] < 0 ? -res1[i] : res1[i];
		b = res2[i] < 0 ? -res2[i] : res2[i];
		if(a < 0.01 && b < 0.01) {
			continue;
		}
		if(i<10) {
			printf("i = %d\t%lf\t%lf\n", i, a, b);
		}
		float diff = (a - b) / (a + 0.000001);
		if(diff < 0) {
			diff= -diff;
		}
		if(diff>0.0005) {
			fail++;
		}
	}
	printf("Number of errors: %d\n", fail);
}

double timestamp(){
	struct timeval tv;
	gettimeofday(&tv, 0);
	return tv.tv_sec + 1e-6 * tv.tv_usec;
}

int main() {
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
			A[i * N + j] = (float)rand() / (float)RAND_MAX;
			B[i * N + j] = (float)rand() / (float)RAND_MAX;
			C[i * N + j] = 0;
			C_cmp[i * N + j] = 0;
		}
	}

	for(int j = 0; j < N; j++) {
		for(int i = 0; i < N; i++) {
			for(int k = 0; k < N; k++) {
				C_cmp[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
			stu[i * N + j].i = i;
			stu[i * N + j].j = j;
		}
	}
	pthread_setconcurrency(N * N);
	strassen(A, B, C, N);
	compare(C, C_cmp, N * N);

	double time1 = timestamp();
	for(int numOfTimes = 0; numOfTimes < ITERATIONS; numOfTimes++) {
		strassen(A, B, C, N);
	}
	double time2 = timestamp();

	double time = (time2 - time1) / ITERATIONS;
	double flops = 2.0 * N * N * N;
	double gflopsPerSecond = flops / 1000000000 /time;
	double GB = 4.0 * N * N / 1000000000;
	double GBpS = 4.0 * N * N / 1000000000 / time;
	printf("GFLOPS/s = %lf\n", gflopsPerSecond);
	printf("GB/s = %lf\n", GBpS);
	printf("GFLOPS = %lf\n", flops / 1000000000);
	printf("GB = %lf\n", GB);
	printf("time(s) = %lf\n", time);
	return 0;
}
