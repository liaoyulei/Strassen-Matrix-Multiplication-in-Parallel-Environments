#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#define t1 64
#define t2 512
#define N 2048
#define ITERATIONS 10
using namespace std;
float A[N * N], B[N * N], C[N * N], C_cmp[N * N];

void split(float *C11, float *C12, float *C21, float *C22, float *C, int n) {
	#pragma omp parallel for schedule(guided)
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) {
			C11[i * n + j] = C[i * 2 * n + j];
			C12[i * n + j] = C[i * 2 * n + j + n];
			C21[i * n + j] = C[(i + n) * 2 * n + j];
			C22[i * n + j] = C[(i + n) * 2 * n + j + n];
		}
	}
}

void merge(float *C11, float *C12, float *C21, float *C22, float *C, int n) {
	#pragma omp parallel for schedule(guided)
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) {
			C[i * 2 * n + j] = C11[i * n + j];
			C[i * 2 * n + j + n] = C12[i * n + j];
			C[(i + n) *2 * n + j] = C21[i * n + j];
			C[(i + n) * 2 * n + j + n] = C22[i * n + j];
		}
	}
}

void add(float *A, float *B, float *C, int n) {
	#pragma omp parallel for schedule(guided)
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) {
			C[i * n + j] = A[i * n + j] + B[i * n + j];
		}
	}
}

void sub(float *A, float *B, float *C, int n) {
	#pragma omp parallel for schedule(guided)
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) {
			C[i * n + j] = A[i * n + j] - B[i * n + j];
		}
	}
}

void mul(float *A, float *B, float *C, int n) {
	#pragma omp parallel for schedule(guided)
	for(int j = 0; j < n; j++) {
		for(int i = 0; i < n; i++) {
			C[i * n + j] = 0;
			for(int k = 0; k < n; k++) {
				C[i * n + j] += A[i * n + k] * B[k * n + j];
			}
		}
	}
}

void mul_add(float *A, float *B, float *T, float *C, int n) {
	#pragma omp parallel for schedule(guided)
	for(int j = 0; j < n; j++) {
		for(int i = 0; i < n; i++) {
			C[i * n + j] = T[i * n + j];
			for(int k = 0; k < n; k++) {
				C[i * n + j] += A[i * n + k] * B[k * n + j];
			}
		}
	}
}

void mul_sub_inc(float *A, float *B, float *T, float *C1, float *C2, int n) {
	#pragma omp parallel for schedule(guided)
	for(int j = 0; j < n; j++) {
		for(int i = 0; i < n; i++) {
			C1[i * n + j] = 0;
			for(int k = 0; k < n; k++) {
				C1[i * n + j] += A[i * n + k] * B[k * n + j];
			}
			C1[i * n + j] = T[i * n + j] - C1[i * n + j];
			C2[i * n + j] += T[i * n + j];
		}
	}
}

void mul_inc_inc_inc(float *A, float *B, float *C, float *T, float *C1, float *C2, int n) {
	#pragma omp parallel for schedule(guided)
	for(int j = 0; j < n; j++) {
		for(int i = 0; i < n; i++) {
			C[i * n + j] = 0;
			for(int k = 0; k < n; k++) {
				C[i * n + j] += A[i * n + k] * B[k * n + j];
			}
			C1[i * n + j] += C[i * n + j];
			C2[i * n + j] += C1[i * n + j];
			C1[i * n + j] += T[i * n + j];
		}
	}
}

void strassen(float *A, float *B, float *C, int n) {
	if(n <= t1) {
		mul(A, B, C, n);
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
			split(A11, A12, A21, A22, A, m);
			split(B11, B12, B21, B22, B, m);
			sub(A11, A21, T1, m);
			sub(B22, B12, T2, m);
			mul(T1, T2, C21, m);
			add(A21, A22, T1, m);
			sub(B12, B11, T2, m);
			mul(T1, T2, C22, m);
			sub(T1, A11, T1, m);
			sub(B22, T2, T2, m);
			mul(T1, T2, C11, m);
			sub(A12, T1, T1, m);
			mul_add(T1, B22, C22, C12, m);
			mul_inc_inc_inc(A11, B11, T1, C21, C11, C12, m);
			sub(T2, B21, T2, m);
			mul_sub_inc(A22, T2, C11, C21, C22, m);
			mul_add(A12, B21, T1, C11, m);
			merge(C11, C12, C21, C22, C, m);	
		}
		else{
			split(A11, A12, A21, A22, A, m);
			split(B11, B12, B21, B22, B, m);
			sub(A11, A21, T1, m);
			sub(B22, B12, T2, m);
			strassen(T1, T2, C21, m);
			add(A21, A22, T1, m);
			sub(B12, B11, T2, m);
			strassen(T1, T2, C22, m);
			sub(T1, A11, T1, m);
			sub(B22, T2, T2, m);
			strassen(T1, T2, C11, m);
			sub(A12, T1, T1, m);
			strassen(T1, B22, C12, m);
			add(C12, C22, C12, m);
			strassen(A11, B11, T1, m);
			add(C11, C12, C12, m);
			add(C12, T1, C12, m);
			add(C11, C21, C11, m);
			add(C11, T1, C11, m);
			sub(T2, B21, T2, m);
			strassen(A22, T2, C21, m);
			sub(C11, C21, C21, m);
			add(C11, C22, C22, m);
			strassen(A12, B21, C11, m);
			add(C11, T1, C11, m);
			merge(C11, C12, C21, C22, C, m);	
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
#if 0 //测试正确性
	for(int j = 0; j < N; j++) {
		for(int i = 0; i < N; i++) {
			for(int k = 0; k < N; k++) {
				C_cmp[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}
	strassen(A, B, C, N);
	compare(C, C_cmp, N * N);
#endif
	double time1 = timestamp();
	for(int numOfTimes = 0; numOfTimes < ITERATIONS; numOfTimes++) {
		strassen(A, B, C, N);
	}
	double time2 = timestamp();
	printf("time(s) = %lf\n", (time2 - time1) / ITERATIONS);
	return 0;
}
