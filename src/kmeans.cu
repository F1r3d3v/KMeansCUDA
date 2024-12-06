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

/* KERNELS */

__global__ void EmptyKernel() {}
void EmptyCUDACall() { EmptyKernel << <1, 1 >> > (); }

template <unsigned int dim>
__device__ __host__
inline float DistanceSquared(float* points, int numPoints, int pointId, float* centroids, int numClusters, int clusterId)
{
	float distance = 0.0f;

	unroll<dim>([&](std::size_t d) {
		float diff = points[d * numPoints + pointId] - centroids[d * numClusters + clusterId];
		distance += diff * diff;
		});

	return distance;
}

template <unsigned int dim>
__global__
void AssignPointsToCentroidsKernel(float* points, int numPoints, int numClusters, float* centroids, int* assignments, int* changedPoints)
{
	int pointId = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (pointId < numPoints)
	{
		int closestCluster = 0;
		float minDistance = FLT_MAX;

		for (int centroidId = 0; centroidId < numClusters; ++centroidId)
		{
			float distance = DistanceSquared<dim>(points, numPoints, pointId, centroids, numClusters, centroidId);
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

template <unsigned int dim>
__global__ void PartialSumsKernel(
	float* points,
	int numPoints,
	int* centroids,
	int numClusters,
	float* partial_array,
	int* partial_array_count,
	int threads_count)
{
	int pointId = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = pointId; i < numPoints; i += stride)
	{
		unroll<dim>([&](std::size_t d) {
			// Calculate the index of the point in the partial array
			int arrayInd = d * numClusters * threads_count + centroids[i] * threads_count + pointId;
			partial_array[arrayInd] += points[d * numPoints + i];
			partial_array_count[arrayInd]++;
			});
	}
}

/* ALGORITHMS */

template <unsigned int dim>
void KMeansCPU(float* data, int numPoints, int numClusters, int maxIterations, float* centroids, int* assignments)
{
	int changedPoints = 0;

	// Set initial centroids to the first k points
	for (int c = 0; c < numClusters; ++c)
	{
		unroll<dim>([&](std::size_t d) {
			centroids[d * numClusters + c] = data[d * numPoints + c];
			});
	}

	for (int iter = 0; iter < maxIterations; ++iter)
	{
		changedPoints = 0;

		// Assign points to the closest cluster
		for (int i = 0; i < numPoints; ++i)
		{
			int closestCluster = 0;
			float minDistance = FLT_MAX;

			for (int c = 0; c < numClusters; ++c)
			{
				float dist = DistanceSquared<dim>(data, numPoints, i, centroids, numClusters, c);
				if (dist < minDistance)
				{
					minDistance = dist;
					closestCluster = c;
				}
			}

			if (assignments[i] != closestCluster)
			{
				changedPoints++;
				assignments[i] = closestCluster;
			}
		}

		// Write progress to console
		printf("Iteration %d: %d points changed clusters\n", iter + 1, changedPoints);

		// Stop if no point changed clusters
		if (changedPoints == 0)
			break;

		// Calculate new centroids
		float* centroidSums = (float*)calloc(numClusters * dim, sizeof(float));
		int* pointsPerCluster = (int*)calloc(numClusters, sizeof(int));
		if (!centroidSums || !pointsPerCluster)
		{
			fprintf(stderr, "Memory allocation failed.\n");
			return;
		}

		for (int i = 0; i < numPoints; ++i)
		{
			int cluster = assignments[i];
			pointsPerCluster[cluster]++;

			unroll<dim>([&](std::size_t d) {
				centroidSums[d * numClusters + cluster] += data[d * numPoints + i];
				});
		}

		for (int c = 0; c < numClusters; ++c)
		{
			if (pointsPerCluster[c] == 0) continue;

			unroll<dim>([&](std::size_t d) {
				centroids[d * numClusters + c] = centroidSums[d * numClusters + c] / pointsPerCluster[c];
				});
		}

		free(centroidSums);
		free(pointsPerCluster);
	}
}

template <unsigned int dim>
cudaError_t KMeansGPU1(float* data, int numPoints, int numClusters, int maxIterations, float* centroids, int* assignments)
{
	cudaError_t cudaStatus;

	// Set initial centroids to the first k points
	for (int c = 0; c < numClusters; ++c)
	{
		unroll<dim>([&](std::size_t d) {
			centroids[d * numClusters + c] = data[d * numPoints + c];
			});
	}

	// Set cuda device
	cudaStatus = cudaSetDevice(0);
	CUDACHECK(cudaStatus);

	// Copy the data to the device
	thrust::device_vector<float> d_data(data, data + numPoints * dim);
	thrust::device_vector<float> d_centroids(centroids, centroids + numClusters * dim);
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
		AssignPointsToCentroidsKernel<dim> << <BLOCKS_PER_GRID(numPoints), THREADS_PER_BLOCK >> > (
			thrust::raw_pointer_cast(d_data.data()),
			numPoints,
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
		unroll<dim>([&](std::size_t d) {
			thrust::copy(d_assignments.begin(), d_assignments.end(), d_assignments_tmp.begin());
			thrust::sort_by_key(d_assignments_tmp.begin(), d_assignments_tmp.end(), d_data.begin() + d * numPoints);

			thrust::reduce_by_key(d_assignments_tmp.begin(), d_assignments_tmp.end(),
				d_data.begin() + d * numPoints,
				thrust::make_discard_iterator(),
				d_centroids.begin() + d * numClusters);

			thrust::transform(d_centroids.begin() + d * numClusters, d_centroids.begin() + (d + 1) * numClusters,
				d_centroidsCount.begin(), d_centroids.begin() + d * numClusters, thrust::divides<float>());
			});
	}

	// Reorder the assignments
	thrust::copy(d_assignments_tmp.begin(), d_assignments_tmp.end(), d_assignments.begin());
	thrust::sort_by_key(d_pointsOrder.begin(), d_pointsOrder.end(), d_assignments.begin());

	// Copy the results back to the host
	thrust::copy(d_assignments.begin(), d_assignments.end(), assignments);
	thrust::copy(d_centroids.begin(), d_centroids.end(), centroids);

	return cudaStatus;
}

template <unsigned int dim>
cudaError_t KMeansGPU2(float* data, int numPoints, int numClusters, int maxIterations, float* centroids, int* assignments)
{
	cudaError_t cudaStatus;

	constexpr int SUMS_THREADS = 2048;
	constexpr int blocks = BLOCKS_PER_GRID(SUMS_THREADS);

	// Set initial centroids to the first k points
	for (int c = 0; c < numClusters; ++c)
	{
		unroll<dim>([&](std::size_t d) {
			centroids[d * numClusters + c] = data[d * numPoints + c];
			});
	}

	// Set cuda device
	cudaStatus = cudaSetDevice(0);
	CUDACHECK(cudaStatus);

	// Allocate device memory
	auto d_data = allocateCudaMemory<float>(numPoints * dim);
	auto d_centroids = allocateCudaMemory<float>(numClusters * dim);
	auto d_assignments = allocateCudaMemory<int>(numPoints);
	auto d_changedPoints = allocateCudaMemory<int>(1);
	auto d_partial_array = allocateCudaMemory<float>(dim * numClusters * SUMS_THREADS);
	auto d_partial_array_count = allocateCudaMemory<int>(dim * numClusters * SUMS_THREADS);
	auto d_centroidsSums = allocateCudaMemory<float>(dim * numClusters * blocks);
	auto d_centroidsCounts = allocateCudaMemory<int>(dim * numClusters * blocks);

	if (!d_data || !d_centroids || !d_assignments || !d_changedPoints ||
		!d_partial_array || !d_partial_array_count || !d_centroidsSums || !d_centroidsCounts)
	{
		fprintf(stderr, "Failed to allocate memory on the device\n");
		return cudaErrorMemoryAllocation;
	}

	// Copy the data to the device
	cudaStatus = cudaMemcpy(d_data.get(), data, numPoints * dim * sizeof(float), cudaMemcpyHostToDevice);
	CUDACHECK(cudaStatus);

	cudaStatus = cudaMemcpy(d_centroids.get(), centroids, numClusters * dim * sizeof(float), cudaMemcpyHostToDevice);
	CUDACHECK(cudaStatus);

	cudaStatus = cudaMemcpy(d_assignments.get(), assignments, numPoints * sizeof(int), cudaMemcpyHostToDevice);
	CUDACHECK(cudaStatus);

	// Allocate memory for partial sums on the host
	auto h_centroidsSums = std::make_unique<float[]>(dim * numClusters * blocks);
	auto h_centroidsCounts = std::make_unique<int[]>(dim * numClusters * blocks);

	for (int iter = 0; iter < maxIterations; ++iter)
	{
		// Copy zero to d_changedPoints, d_partial_array and d_partial_array_count
		cudaStatus = cudaMemset(d_changedPoints.get(), 0, sizeof(int));
		CUDACHECK(cudaStatus);

		cudaStatus = cudaMemset(d_partial_array.get(), 0, dim * numClusters * SUMS_THREADS * sizeof(float));
		CUDACHECK(cudaStatus);

		cudaStatus = cudaMemset(d_partial_array_count.get(), 0, dim * numClusters * SUMS_THREADS * sizeof(int));
		CUDACHECK(cudaStatus);

		// Assign points to centroids
		AssignPointsToCentroidsKernel<dim> << <BLOCKS_PER_GRID(numPoints), THREADS_PER_BLOCK >> > (
			d_data.get(),
			numPoints,
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
		PartialSumsKernel<dim> << <blocks, THREADS_PER_BLOCK >> > (
			d_data.get(),
			numPoints,
			d_assignments.get(),
			numClusters,
			d_partial_array.get(),
			d_partial_array_count.get(),
			SUMS_THREADS);

		cudaStatus = cudaDeviceSynchronize();
		CUDACHECK(cudaStatus);

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		CUDACHECK(cudaStatus);

		// Calculate the new centroids
		unroll<dim>([&](std::size_t d) {
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
			});

		// Wait for the kernels to finish
		cudaStatus = cudaDeviceSynchronize();
		CUDACHECK(cudaStatus);

		// Check for any errors launching the kernels
		cudaStatus = cudaGetLastError();
		CUDACHECK(cudaStatus);

		// Copy the partial sums to the host
		cudaStatus = cudaMemcpy(h_centroidsSums.get(), d_centroidsSums.get(), dim * numClusters * blocks * sizeof(float), cudaMemcpyDeviceToHost);
		CUDACHECK(cudaStatus);

		cudaStatus = cudaMemcpy(h_centroidsCounts.get(), d_centroidsCounts.get(), dim * numClusters * blocks * sizeof(int), cudaMemcpyDeviceToHost);
		CUDACHECK(cudaStatus);

		// Calculate the new centroids on the host
		unroll<dim>([&](std::size_t d) {
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
			});

		// Copy the new centroids to the device
		cudaStatus = cudaMemcpy(d_centroids.get(), centroids, numClusters * dim * sizeof(float), cudaMemcpyHostToDevice);
		CUDACHECK(cudaStatus);
	}

	// Copy the assignments back to the host
	cudaStatus = cudaMemcpy(assignments, d_assignments.get(), numPoints * sizeof(int), cudaMemcpyDeviceToHost);
	CUDACHECK(cudaStatus);

	return cudaStatus;
}

/* RUNNERS */

void KMeansCPU_Runner(float* data, int numPoints, int dimensions, int numClusters, int maxIterations, float* centroids, int* assignments)
{
	switch (dimensions)
	{
	case 1:
		KMeansCPU<1>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 2:
		KMeansCPU<2>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 3:
		KMeansCPU<3>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 4:
		KMeansCPU<4>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 5:
		KMeansCPU<5>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 6:
		KMeansCPU<6>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 7:
		KMeansCPU<7>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 8:
		KMeansCPU<8>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 9:
		KMeansCPU<9>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 10:
		KMeansCPU<10>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 11:
		KMeansCPU<11>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 12:
		KMeansCPU<12>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 13:
		KMeansCPU<13>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 14:
		KMeansCPU<14>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 15:
		KMeansCPU<15>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 16:
		KMeansCPU<16>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 17:
		KMeansCPU<17>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 18:
		KMeansCPU<18>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 19:
		KMeansCPU<19>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 20:
		KMeansCPU<20>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	default:
		fprintf(stderr, "Unsupported dimension count!\n");
		exit(EXIT_FAILURE);
		break;
	}
}

cudaError_t KMeansGPU1_Runner(float* data, int numPoints, int dimensions, int numClusters, int maxIterations, float* centroids, int* assignments)
{
	cudaError_t cudaStatus;
	switch (dimensions)
	{
	case 1:
		cudaStatus = KMeansGPU1<1>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 2:
		cudaStatus = KMeansGPU1<2>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 3:
		cudaStatus = KMeansGPU1<3>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 4:
		cudaStatus = KMeansGPU1<4>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 5:
		cudaStatus = KMeansGPU1<5>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 6:
		cudaStatus = KMeansGPU1<6>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 7:
		cudaStatus = KMeansGPU1<7>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 8:
		cudaStatus = KMeansGPU1<8>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 9:
		cudaStatus = KMeansGPU1<9>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 10:
		cudaStatus = KMeansGPU1<10>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 11:
		cudaStatus = KMeansGPU1<11>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 12:
		cudaStatus = KMeansGPU1<12>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 13:
		cudaStatus = KMeansGPU1<13>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 14:
		cudaStatus = KMeansGPU1<14>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 15:
		cudaStatus = KMeansGPU1<15>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 16:
		cudaStatus = KMeansGPU1<16>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 17:
		cudaStatus = KMeansGPU1<17>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 18:
		cudaStatus = KMeansGPU1<18>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 19:
		cudaStatus = KMeansGPU1<19>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 20:
		cudaStatus = KMeansGPU1<20>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	default:
		fprintf(stderr, "Unsupported dimension count!\n");
		exit(EXIT_FAILURE);
		break;
	}

	return cudaStatus;
}

cudaError_t KMeansGPU2_Runner(float* data, int numPoints, int dimensions, int numClusters, int maxIterations, float* centroids, int* assignments)
{
	cudaError_t cudaStatus;
	switch (dimensions)
	{
	case 1:
		cudaStatus = KMeansGPU2<1>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 2:
		cudaStatus = KMeansGPU2<2>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 3:
		cudaStatus = KMeansGPU2<3>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 4:
		cudaStatus = KMeansGPU2<4>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 5:
		cudaStatus = KMeansGPU2<5>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 6:
		cudaStatus = KMeansGPU2<6>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 7:
		cudaStatus = KMeansGPU2<7>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 8:
		cudaStatus = KMeansGPU2<8>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 9:
		cudaStatus = KMeansGPU2<9>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 10:
		cudaStatus = KMeansGPU2<10>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 11:
		cudaStatus = KMeansGPU2<11>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 12:
		cudaStatus = KMeansGPU2<12>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 13:
		cudaStatus = KMeansGPU2<13>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 14:
		cudaStatus = KMeansGPU2<14>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 15:
		cudaStatus = KMeansGPU2<15>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 16:
		cudaStatus = KMeansGPU2<16>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 17:
		cudaStatus = KMeansGPU2<17>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 18:
		cudaStatus = KMeansGPU2<18>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 19:
		cudaStatus = KMeansGPU2<19>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	case 20:
		cudaStatus = KMeansGPU2<20>(data, numPoints, numClusters, maxIterations, centroids, assignments);
		break;
	default:
		fprintf(stderr, "Unsupported dimension count!\n");
		exit(1);
		break;
	}

	return cudaStatus;
}