#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kmeans.h"

#include <stdio.h>

//constexpr int MAX_ITER = 100;
//constexpr int THREADS_PER_BLOCK = 256;
//static constexpr int BLOCKS_PER_GRID(int x) { return (x + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK; }

__device__ __host__
inline float DistanceSquared(float* points,
    int pointsSize,
    int pointId,
    float* centroids,
    int centroidsSize,
    int centroidId,
    int dimensions)
{
    float distance = 0.0f;
    for (int d = 0; d < dimensions; ++d) {
        float diff = points[d * pointsSize + pointId] - centroids[d * centroidsSize + centroidId];
        distance += diff * diff;
    }
    return distance;
}

__global__ void AssignPointsToCentroidsKernel(float* points,
	int pointsSize,
	float* centroids,
	int centroidsSize,
	int dimensions,
	int* assignments)
{
	int pointId = blockIdx.x * blockDim.x + threadIdx.x;
	if (pointId >= pointsSize) {
		return;
	}

	float minDistance = DistanceSquared(points, pointsSize, pointId, centroids, centroidsSize, 0, dimensions);
	assignments[pointId] = 0;
	for (int centroidId = 1; centroidId < centroidsSize; ++centroidId) {
		float distance = DistanceSquared(points, pointsSize, pointId, centroids, centroidsSize, centroidId, dimensions);
		if (distance < minDistance) {
			minDistance = distance;
			assignments[pointId] = centroidId;
		}
	}
}

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
