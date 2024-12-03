#include "kmeans.h"

#include <cstdlib>
#include <cfloat>
#include <cstdio>

float calculateDistance(const float* point1, const float* point2, int dimensions)
{
	float sum = 0.0f;
	for (int d = 0; d < dimensions; ++d)
	{
		float diff = point1[d] - point2[d];
		sum += diff * diff;
	}
	return sum;
}

void KMeansCPU(float* data, int numPoints, int dimensions, int numClusters, int maxIterations, float* centroids, int* assignments)
{
	int changedPoints = 0;

	// Set initial centroids to the first k points
	for (int c = 0; c < numClusters; ++c)
		for (int d = 0; d < dimensions; ++d)
			centroids[c * dimensions + d] = data[c * dimensions + d];

	for (int iter = 0; iter < maxIterations; ++iter)
	{
		changedPoints = 0;

		// Assign points to the closest cluster
		for (int i = 0; i < numPoints; ++i)
		{
			int closestCluster = -1;
			float minDistance = FLT_MAX;

			for (int c = 0; c < numClusters; ++c)
			{
				float dist = calculateDistance(&data[i * dimensions], &centroids[c * dimensions], dimensions);
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
		float* centroidSums = (float*)calloc(numClusters * dimensions, sizeof(float));
		int* pointsPerCluster = (int*)calloc(numClusters, sizeof(int));

		for (int i = 0; i < numPoints; ++i)
		{
			int cluster = assignments[i];
			pointsPerCluster[cluster]++;
			for (int d = 0; d < dimensions; ++d)
			{
				centroidSums[cluster * dimensions + d] += data[i * dimensions + d];
			}
		}

		for (int c = 0; c < numClusters; ++c)
		{
			if (pointsPerCluster[c] == 0) continue;
			for (int d = 0; d < dimensions; ++d)
			{
				centroids[c * dimensions + d] = centroidSums[c * dimensions + d] / pointsPerCluster[c];
			}
		}

		free(centroidSums);
		free(pointsPerCluster);
	}
}