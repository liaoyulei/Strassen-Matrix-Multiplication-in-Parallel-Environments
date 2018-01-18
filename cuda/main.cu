#include <iostream>
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#define t1 4096
#define t2 4096
#define N 1
#define ITERATIONS 10
#define BLOCK_SIZE 32
using namespace std;
float A[N * N], B[N * N], C[N * N], C_cmp[N * N];

__global__ void split(float *C11, float *C12, float *C21, float *C22, float *C, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if(i < n && j < n) {
		C11[i * n + j] = C[i * 2 * n + j];
		C12[i * n + j] = C[i * 2 * n + j + n];
		C21[i * n + j] = C[(i + n) * 2 * n + j];
		C22[i * n + j] = C[(i + n) * 2 * n + j + n];
	}
}

__global__ void merge(float *C11, float *C12, float *C21, float *C22, float *C, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if(i < n && j < n) {
		C[i * 2 * n + j] = C11[i * n + j];
		C[i * 2 * n + j + n] = C12[i * n + j];
		C[(i + n) *2 * n + j] = C21[i * n + j];
		C[(i + n) * 2 * n + j + n] = C22[i * n + j];
	}
}

__global__ void add(float *A, float *B, float *C, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if(i < n && j < n) {
		C[i * n + j] = A[i * n + j] + B[i * n + j];
	}
}

__global__ void sub(float *A, float *B, float *C, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if(i < n && j < n) {
		C[i * n + j] = A[i * n + j] - B[i * n + j];
	}
}

__global__ void mul(float *A, float *B, float *C, int n) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	if(i < n && j < n) {
		C[i * n + j] = 0;
		for(int k = 0; k < n; k++) {
			C[i * n + j] += A[i * n + k] * B[k * n + j];
		}
	}
}

__global__ void mul_add(float *A, float *B, float *T, float *C, int n) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	if(i < n && j < n) {
		C[i * n + j] = T[i * n + j];
		for(int k = 0; k < n; k++) {
			C[i * n + j] += A[i * n + k] * B[k * n + j];
		}
	}
}

__global__ void mul_sub_inc(float *A, float *B, float *T, float *C1, float *C2, int n) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	if(i < n && j < n) {
		C1[i * n + j] = 0;
		for(int k = 0; k < n; k++) {
			C1[i * n + j] += A[i * n + k] * B[k * n + j];
		}
		C1[i * n + j] = T[i * n + j] - C1[i * n + j];
		C2[i * n + j] += T[i * n + j];
	}
}

__global__ void mul_inc_inc_inc(float *A, float *B, float *C, float *T, float *C1, float *C2, int n) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	if(i < n && j < n) {
		C[i * n + j] = 0;
		for(int k = 0; k < n; k++) {
			C[i * n + j] += A[i * n + k] * B[k * n + j];
		}
		C1[i * n + j] += C[i * n + j];
		C2[i * n + j] += C1[i * n + j];
		C1[i * n + j] += T[i * n + j];
	}
}

void strassen(float *A, float *B, float *C, int n) {
	float *A_gpu, *B_gpu, *C_gpu;
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	cudaMalloc((void **)&A_gpu, sizeof(float) * n * n);
	cudaMalloc((void **)&B_gpu, sizeof(float) * n * n);
	cudaMalloc((void **)&C_gpu, sizeof(float) * n * n);
	cudaMemcpy(A_gpu, A, sizeof(float) * n * n, cudaMemcpyHostToDevice);
	cudaMemcpy(B_gpu, B, sizeof(float) * n * n, cudaMemcpyHostToDevice);
	if(n <= t1) {
		dim3 grid((size_t)ceil((float)n / (float)block.x), (size_t)ceil((float)n / (float)block.y));
		mul<<<grid, block>>>(A_gpu, B_gpu, C_gpu, n);
		cudaThreadSynchronize();
	}
	else{
		int m = n >> 1;
		dim3 grid((size_t)ceil((float)m / (float)block.x), (size_t)ceil((float)m / (float)block.y));
		float *A11, *A12, *A21, *A22, *B11, *B12, *B21, *B22, *C11, *C12, *C21, *C22, *T1, *T2;
		cudaMalloc((void **)&A11, sizeof(float) * m * m);
		cudaMalloc((void **)&A12, sizeof(float) * m * m);
		cudaMalloc((void **)&A21, sizeof(float) * m * m);
		cudaMalloc((void **)&A22, sizeof(float) * m * m);
		cudaMalloc((void **)&B11, sizeof(float) * m * m);
		cudaMalloc((void **)&B12, sizeof(float) * m * m);
		cudaMalloc((void **)&B21, sizeof(float) * m * m);
		cudaMalloc((void **)&B22, sizeof(float) * m * m);
		cudaMalloc((void **)&C11, sizeof(float) * m * m);
		cudaMalloc((void **)&C12, sizeof(float) * m * m);
		cudaMalloc((void **)&C21, sizeof(float) * m * m);
		cudaMalloc((void **)&C22, sizeof(float) * m * m);
		cudaMalloc((void **)&T1, sizeof(float) * m * m);
		cudaMalloc((void **)&T2, sizeof(float) * m * m);
		if(n <= t2) {
			split<<<grid, block>>>(A11, A12, A21, A22, A_gpu, m);
			cudaThreadSynchronize();
			split<<<grid, block>>>(B11, B12, B21, B22, B_gpu, m);
			cudaThreadSynchronize();
			sub<<<grid, block>>>(A11, A21, T1, m);
			cudaThreadSynchronize();
			sub<<<grid, block>>>(B22, B12, T2, m);
			cudaThreadSynchronize();
			mul<<<grid, block>>>(T1, T2, C21, m);
			cudaThreadSynchronize();
			add<<<grid, block>>>(A21, A22, T1, m);
			cudaThreadSynchronize();
			sub<<<grid, block>>>(B12, B11, T2, m);
			cudaThreadSynchronize();
			mul<<<grid, block>>>(T1, T2, C22, m);
			cudaThreadSynchronize();
			sub<<<grid, block>>>(T1, A11, T1, m);
			cudaThreadSynchronize();
			sub<<<grid, block>>>(B22, T2, T2, m);
			cudaThreadSynchronize();
			mul<<<grid, block>>>(T1, T2, C11, m);
			cudaThreadSynchronize();
			sub<<<grid, block>>>(A12, T1, T1, m);
			cudaThreadSynchronize();
			mul_add<<<grid, block>>>(T1, B22, C22, C12, m);
			cudaThreadSynchronize();
			mul_inc_inc_inc<<<grid, block>>>(A11, B11, T1, C21, C11, C12, m);
			cudaThreadSynchronize();
			sub<<<grid, block>>>(T2, B21, T2, m);
			cudaThreadSynchronize();
			mul_sub_inc<<<grid, block>>>(A22, T2, C11, C21, C22, m);
			cudaThreadSynchronize();
			mul_add<<<grid, block>>>(A12, B21, T1, C11, m);
			cudaThreadSynchronize();
			merge<<<grid, block>>>(C11, C12, C21, C22, C_gpu, m);	
			cudaThreadSynchronize();
		}
		else{
			split<<<grid, block>>>(A11, A12, A21, A22, A_gpu, m);
			cudaThreadSynchronize();
			split<<<grid, block>>>(B11, B12, B21, B22, B_gpu, m);
			cudaThreadSynchronize();
			sub<<<grid, block>>>(A11, A21, T1, m);
			cudaThreadSynchronize();
			sub<<<grid, block>>>(B22, B12, T2, m);
			cudaThreadSynchronize();
			strassen(T1, T2, C21, m);
			add<<<grid, block>>>(A21, A22, T1, m);
			cudaThreadSynchronize();
			sub<<<grid, block>>>(B12, B11, T2, m);
			cudaThreadSynchronize();
			strassen(T1, T2, C22, m);
			sub<<<grid, block>>>(T1, A11, T1, m);
			cudaThreadSynchronize();
			sub<<<grid, block>>>(B22, T2, T2, m);
			cudaThreadSynchronize();
			strassen(T1, T2, C11, m);
			sub<<<grid, block>>>(A12, T1, T1, m);
			cudaThreadSynchronize();
			strassen(T1, B22, C12, m);
			add<<<grid, block>>>(C12, C22, C12, m);
			cudaThreadSynchronize();
			strassen(A11, B11, T1, m);
			add<<<grid, block>>>(C11, C12, C12, m);
			cudaThreadSynchronize();
			add<<<grid, block>>>(C12, T1, C12, m);
			cudaThreadSynchronize();
			add<<<grid, block>>>(C11, C21, C11, m);
			cudaThreadSynchronize();
			add<<<grid, block>>>(C11, T1, C11, m);
			cudaThreadSynchronize();
			sub<<<grid, block>>>(T2, B21, T2, m);
			cudaThreadSynchronize();
			strassen(A22, T2, C21, m);
			sub<<<grid, block>>>(C11, C21, C21, m);
			cudaThreadSynchronize();
			add<<<grid, block>>>(C11, C22, C22, m);
			cudaThreadSynchronize();
			strassen(A12, B21, C11, m);
			add<<<grid, block>>>(C11, T1, C11, m);
			cudaThreadSynchronize();
			merge<<<grid, block>>>(C11, C12, C21, C22, C_gpu, m);	
			cudaThreadSynchronize();
		}
		cudaFree(A11); 
		cudaFree(A12); 
		cudaFree(A21); 
		cudaFree(A22); 
		cudaFree(B11); 
		cudaFree(B12); 
		cudaFree(B21); 
		cudaFree(B22); 
		cudaFree(T1);
		cudaFree(T2);	
	} 
	cudaMemcpy(C, C_gpu, sizeof(float) * n * n, cudaMemcpyDeviceToHost);
	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(C_gpu);
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
