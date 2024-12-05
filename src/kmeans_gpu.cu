#include "device_launch_parameters.h"
#include "kmeans.h"
#include "reduce.cuh"
#include "cuda_helper.cuh"

#include <stdio.h>
#include <memory>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/constant_iterator.h>

constexpr int THREADS_PER_BLOCK = 256;
static constexpr int BLOCKS_PER_GRID(int x) { return (x + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK; }

__global__ void EmptyKernel() {}
void EmptyCUDACall() { EmptyKernel<<<1,1>>>(); }

__device__ __host__
inline float DistanceSquared(
	float* points,
	int numPoints,
	int pointId,
	float* centroids,
	int numClusters,
	int clusterId,
	int dimensions)
{
	float distance = 0.0f;
	for (int d = 0; d < dimensions; ++d)
	{
		float diff = points[d * numPoints + pointId] - centroids[d * numClusters + clusterId];
		distance += diff * diff;
	}
	return distance;
}

float DistanceSquaredHost(float* points, int numPoints, int pointId, float* centroids, int numClusters, int clusterId, int dimensions)
{
	return DistanceSquared(points, numPoints, pointId, centroids, numClusters, clusterId, dimensions);
}

__global__
void AssignPointsToCentroidsKernel(
	float* points,
	int numPoints,
	int dimensions,
	int numClusters,
	float* centroids,
	int* assignments,
	int* changedPoints)
{
	int pointId = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (pointId < numPoints)
	{
		int closestCluster = 0;
		float minDistance = FLT_MAX;

		for (int centroidId = 0; centroidId < numClusters; ++centroidId)
		{
			float distance = DistanceSquared(points, numPoints, pointId, centroids, numClusters, centroidId, dimensions);
			if (distance < minDistance)
			{
				minDistance = distance;
				closestCluster = centroidId;
			}
		}

		if (assignments[pointId] != closestCluster)
		{
			atomicAggInc(changedPoints);
			assignments[pointId] = closestCluster;
		}

		pointId += stride;
	}
}

__global__ void PartialSumsKernel(
	float* points,
	int numPoints,
	int* centroids,
	int numClusters,
	int dimensions,
	float* partial_array,
	int* partial_array_count,
	int threads_count)
{
	int pointId = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = pointId; i < numPoints; i += stride)
	{
		for (int d = 0; d < dimensions; ++d)
		{
			// Calculate the index of the point in the partial array
			int arrayInd = d * numClusters * threads_count + centroids[i] * threads_count + pointId;
			partial_array[arrayInd] += points[d * numPoints + i];
			partial_array_count[arrayInd]++;
		}
	}
}

cudaError_t KMeansGPU1(float* data, int numPoints, int dimensions, int numClusters, int maxIterations, float* centroids, int* assignments)
{
	cudaError_t cudaStatus;

	// Set initial centroids to the first k points
	for (int c = 0; c < numClusters; ++c)
		for (int d = 0; d < dimensions; ++d)
			centroids[d * numClusters + c] = data[d * numPoints + c];

	// Set cuda device
	cudaStatus = cudaSetDevice(0);
	CUDACHECK(cudaStatus);

	// Copy the data to the device
	thrust::device_vector<float> d_data(data, data + numPoints * dimensions);
	thrust::device_vector<float> d_centroids(centroids, centroids + numClusters * dimensions);
	thrust::device_vector<int> d_assignments(assignments, assignments + numPoints);
	thrust::device_vector<int> d_assignments_tmp(numPoints);
	thrust::device_vector<int> d_centroidsCount(numClusters);
	thrust::device_vector<int> d_deltas(numPoints);

	auto d_changedPoints = allocateCudaMemory<int>(1);
	if (!d_changedPoints)
	{
		fprintf(stderr, "Failed to allocate memory for d_changedPoints\n");
		return cudaErrorMemoryAllocation;
	}

	thrust::device_vector<int> d_pointsOrder(numPoints);
	thrust::sequence(d_pointsOrder.begin(), d_pointsOrder.end());

	for (int iter = 0; iter < maxIterations; ++iter)
	{
		// Copy zero to d_changedPoints
		cudaStatus = cudaMemset(d_changedPoints.get(), 0, sizeof(int));
		CUDACHECK(cudaStatus);

		// Assign points to centroids
		AssignPointsToCentroidsKernel << <BLOCKS_PER_GRID(numPoints), THREADS_PER_BLOCK >> > (
			thrust::raw_pointer_cast(d_data.data()),
			numPoints,
			dimensions,
			numClusters,
			thrust::raw_pointer_cast(d_centroids.data()),
			thrust::raw_pointer_cast(d_assignments.data()),
			d_changedPoints.get());

		cudaStatus = cudaDeviceSynchronize();
		CUDACHECK(cudaStatus);

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		CUDACHECK(cudaStatus);

		// Copy the changedPoints back to the host
		int changedPoints;
		cudaStatus = cudaMemcpy(&changedPoints, d_changedPoints.get(), sizeof(int), cudaMemcpyDeviceToHost);
		CUDACHECK(cudaStatus);

		// Write progress to console
		printf("Iteration %d: %d points changed clusters\n", iter + 1, changedPoints);

		// Stop if no point changed clusters
		if (changedPoints == 0)
			break;

		// Sort the sequence by cluster assignment for reordering
		thrust::copy(d_assignments.begin(), d_assignments.end(), d_assignments_tmp.begin());
		thrust::sort_by_key(d_assignments_tmp.begin(), d_assignments_tmp.end(), d_pointsOrder.begin());

		// Count the number of points in each cluster
		thrust::reduce_by_key(d_assignments_tmp.begin(), d_assignments_tmp.end(),
			thrust::make_constant_iterator(1),
			thrust::make_discard_iterator(),
			d_centroidsCount.begin());

		// Compute the new centroids
		for (int d = 0; d < dimensions; ++d)
		{
			thrust::copy(d_assignments.begin(), d_assignments.end(), d_assignments_tmp.begin());

			thrust::sort_by_key(d_assignments_tmp.begin(), d_assignments_tmp.end(), d_data.begin() + d * numPoints);

			thrust::reduce_by_key(d_assignments_tmp.begin(), d_assignments_tmp.end(),
				d_data.begin() + d * numPoints,
				thrust::make_discard_iterator(),
				d_centroids.begin() + d * numClusters);

			thrust::transform(d_centroids.begin() + d * numClusters, d_centroids.begin() + (d + 1) * numClusters,
				d_centroidsCount.begin(), d_centroids.begin() + d * numClusters, thrust::divides<float>());
		}
	}

	// Reorder the assignments
	thrust::copy(d_assignments_tmp.begin(), d_assignments_tmp.end(), d_assignments.begin());
	thrust::sort_by_key(d_pointsOrder.begin(), d_pointsOrder.end(), d_assignments.begin());

	// Copy the results back to the host
	thrust::copy(d_assignments.begin(), d_assignments.end(), assignments);
	thrust::copy(d_centroids.begin(), d_centroids.end(), centroids);

	return cudaStatus;
}

cudaError_t KMeansGPU2(float* data, int numPoints, int dimensions, int numClusters, int maxIterations, float* centroids, int* assignments)
{
	cudaError_t cudaStatus;

	constexpr int SUMS_THREADS = 2048;
	constexpr int blocks = BLOCKS_PER_GRID(SUMS_THREADS);

	// Set initial centroids to the first k points
	for (int c = 0; c < numClusters; ++c)
		for (int d = 0; d < dimensions; ++d)
			centroids[d * numClusters + c] = data[d * numPoints + c];

	// Set cuda device
	cudaStatus = cudaSetDevice(0);
	CUDACHECK(cudaStatus);

	// Allocate device memory
	auto d_data = allocateCudaMemory<float>(numPoints * dimensions);
	auto d_centroids = allocateCudaMemory<float>(numClusters * dimensions);
	auto d_assignments = allocateCudaMemory<int>(numPoints);
	auto d_changedPoints = allocateCudaMemory<int>(1);
	auto d_partial_array = allocateCudaMemory<float>(dimensions * numClusters * SUMS_THREADS);
	auto d_partial_array_count = allocateCudaMemory<int>(dimensions * numClusters * SUMS_THREADS);
	auto d_centroidsSums = allocateCudaMemory<float>(dimensions * numClusters * blocks);
	auto d_centroidsCounts = allocateCudaMemory<int>(dimensions * numClusters * blocks);

	if (!d_data || !d_centroids || !d_assignments || !d_changedPoints ||
		!d_partial_array || !d_partial_array_count || !d_centroidsSums || !d_centroidsCounts)
	{
		fprintf(stderr, "Failed to allocate memory on the device\n");
		return cudaErrorMemoryAllocation;
	}

	// Copy the data to the device
	cudaStatus = cudaMemcpy(d_data.get(), data, numPoints * dimensions * sizeof(float), cudaMemcpyHostToDevice);
	CUDACHECK(cudaStatus);

	cudaStatus = cudaMemcpy(d_centroids.get(), centroids, numClusters * dimensions * sizeof(float), cudaMemcpyHostToDevice);
	CUDACHECK(cudaStatus);

	cudaStatus = cudaMemcpy(d_assignments.get(), assignments, numPoints * sizeof(int), cudaMemcpyHostToDevice);
	CUDACHECK(cudaStatus);

	// Allocate memory for partial sums on the host
	auto h_centroidsSums = std::make_unique<float[]>(dimensions * numClusters * blocks);
	auto h_centroidsCounts = std::make_unique<int[]>(dimensions * numClusters * blocks);

	for (int iter = 0; iter < maxIterations; ++iter)
	{
		// Copy zero to d_changedPoints, d_partial_array and d_partial_array_count
		cudaStatus = cudaMemset(d_changedPoints.get(), 0, sizeof(int));
		CUDACHECK(cudaStatus);

		cudaStatus = cudaMemset(d_partial_array.get(), 0, dimensions * numClusters * SUMS_THREADS * sizeof(float));
		CUDACHECK(cudaStatus);

		cudaStatus = cudaMemset(d_partial_array_count.get(), 0, dimensions * numClusters * SUMS_THREADS * sizeof(int));
		CUDACHECK(cudaStatus);

		// Assign points to centroids
		AssignPointsToCentroidsKernel << <BLOCKS_PER_GRID(numPoints), THREADS_PER_BLOCK >> > (
			d_data.get(),
			numPoints,
			dimensions,
			numClusters,
			d_centroids.get(),
			d_assignments.get(),
			d_changedPoints.get());

		cudaStatus = cudaDeviceSynchronize();
		CUDACHECK(cudaStatus);

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		CUDACHECK(cudaStatus);

		// Copy the changedPoints back to the host
		int changedPoints;
		cudaStatus = cudaMemcpy(&changedPoints, d_changedPoints.get(), sizeof(int), cudaMemcpyDeviceToHost);
		CUDACHECK(cudaStatus);

		// Write progress to console
		printf("Iteration %d: %d points changed clusters\n", iter + 1, changedPoints);

		// Stop if no point changed clusters
		if (changedPoints == 0)
			break;

		// Launch the kernel
		PartialSumsKernel << <blocks, THREADS_PER_BLOCK >> > (
			d_data.get(),
			numPoints,
			d_assignments.get(),
			numClusters,
			dimensions,
			d_partial_array.get(),
			d_partial_array_count.get(),
			SUMS_THREADS);

		cudaStatus = cudaDeviceSynchronize();
		CUDACHECK(cudaStatus);

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		CUDACHECK(cudaStatus);

		// Calculate the new centroids
		for (int d = 0; d < dimensions; ++d)
		{
			for (int c = 0; c < numClusters; ++c)
			{
				ReduceKernel<THREADS_PER_BLOCK, float> << <blocks, THREADS_PER_BLOCK >> > (
					d_partial_array.get() + d * numClusters * SUMS_THREADS + c * SUMS_THREADS,
					d_centroidsSums.get() + d * numClusters * blocks + c * blocks,
					SUMS_THREADS);

				ReduceKernel<THREADS_PER_BLOCK, int> << <blocks, THREADS_PER_BLOCK >> > (
					d_partial_array_count.get() + d * numClusters * SUMS_THREADS + c * SUMS_THREADS,
					d_centroidsCounts.get() + d * numClusters * blocks + c * blocks,
					SUMS_THREADS);
			}
		}

		// Wait for the kernels to finish
		cudaStatus = cudaDeviceSynchronize();
		CUDACHECK(cudaStatus);

		// Check for any errors launching the kernels
		cudaStatus = cudaGetLastError();
		CUDACHECK(cudaStatus);

		// Copy the partial sums to the host
		cudaStatus = cudaMemcpy(h_centroidsSums.get(), d_centroidsSums.get(), dimensions * numClusters * blocks * sizeof(float), cudaMemcpyDeviceToHost);
		CUDACHECK(cudaStatus);

		cudaStatus = cudaMemcpy(h_centroidsCounts.get(), d_centroidsCounts.get(), dimensions * numClusters * blocks * sizeof(int), cudaMemcpyDeviceToHost);
		CUDACHECK(cudaStatus);

		// Calculate the new centroids on the host
		for (int d = 0; d < dimensions; ++d)
		{
			for (int c = 0; c < numClusters; ++c)
			{
				float sum = 0.0f;
				int count = 0;

				for (int i = 0; i < blocks; ++i)
				{
					sum += h_centroidsSums[d * numClusters * blocks + c * blocks + i];
					count += h_centroidsCounts[d * numClusters * blocks + c * blocks + i];
				}

				centroids[d * numClusters + c] = sum / count;
			}
		}

		// Copy the new centroids to the device
		cudaStatus = cudaMemcpy(d_centroids.get(), centroids, numClusters * dimensions * sizeof(float), cudaMemcpyHostToDevice);
		CUDACHECK(cudaStatus);
	}

	// Copy the assignments back to the host
	cudaStatus = cudaMemcpy(assignments, d_assignments.get(), numPoints * sizeof(int), cudaMemcpyDeviceToHost);
	CUDACHECK(cudaStatus);

	return cudaStatus;
}