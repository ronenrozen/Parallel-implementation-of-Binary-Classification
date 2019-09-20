#include "Header.h"
#include <stdlib.h>
#define MAX_NUM_OF_THREADS 1000


__device__ int atomicAdd(int* address, int val);

int * nMiss;

/**********************************************
Algorithm function cuda version
***********************************************/
__device__ double fOnGPU(Model * mod, double * points, int K) {
	double sum = mod->bias;
	for (int i = 0; i < K; i++)
	{
		sum += mod->weights[i] * points[i];
	}
	return SIGN(sum);
}

/**********************************************
Each thread calculates different element of points
***********************************************/
__global__ void calculatePointsOnKernel(Vector *points, int numOfthreads, Model * mod, int K, int* nMiss) {
	int thread_index = threadIdx.x;
	int block_index = blockIdx.x;
	int index = thread_index + block_index * numOfthreads;

	int prediction = fOnGPU(mod, points[index].points, K);
	if (points[index].expected != prediction)
	{
		atomicAdd(nMiss, 1);
	}
}



/**********************************************
init the vectors for dev_results and dev_points
doing it once per process and not in every train
***********************************************/
cudaError_t initCuda(int numOfTasks, Vector **dev_points, Model **dev_mod, Vector *points) {
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return cudaStatus;
	}

	// Allocate GPU buffer for points.
	cudaStatus = cudaMalloc((void**)dev_points, MAX_POINTS * sizeof(Vector));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc for tasks failed!");
		return cudaStatus;
	}

	// Copy points from CPU to GPU.
	cudaStatus = cudaMemcpy(*dev_points, points, MAX_POINTS * sizeof(Vector), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy for tasks failed!");
		return cudaStatus;
	}

	// Allocate GPU buffer for Model.
	cudaStatus = cudaMalloc((void**)dev_mod, sizeof(Model));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc for dev_mod failed!");
		return cudaErrorUnknown;
	}
	// Allocate GPU buffer for nMiss.
	cudaStatus = cudaMalloc((void**)&nMiss, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc for nMiss failed!");
		return cudaErrorUnknown;
	}
	return cudaStatus;
}

/**********************************************
Main func of cuda, here we manage all the operations of the cuda

***********************************************/
cudaError_t calculateWithCuda(Vector *dev_points, int numOfPoints, double *q, Model * mod, Model * dev_mod, int K)
{

	cudaError_t cudaStatus;

	// Copy model from CPU to GPU.
	cudaStatus = cudaMemcpy(dev_mod, mod, sizeof(Model), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy for tasks failed!");
		return cudaStatus;
	}

	int blocks = numOfPoints / MAX_NUM_OF_THREADS > 1 ? numOfPoints / MAX_NUM_OF_THREADS : 1;
	int threads = numOfPoints / MAX_NUM_OF_THREADS > 1 ? MAX_NUM_OF_THREADS : numOfPoints;

	// Launch a kernel on the GPU with one thread for each element.
	calculatePointsOnKernel << <blocks, threads >> > (dev_points, threads, dev_mod, K, nMiss);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "resultKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching resultKernel!\n", cudaStatus);
		return cudaStatus;
	}

	int * nMissCPU = (int *)malloc(sizeof(int));
	// Copy nMiss from GPU buffer to CPU memory.
	cudaStatus = cudaMemcpy(nMissCPU, nMiss, sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy for final result failed!");
		return cudaStatus;
	}
	*q = (double)*nMissCPU / (double)numOfPoints;
	return cudaStatus;
}
