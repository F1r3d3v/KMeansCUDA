#include "kmeans.h"

#include <cstdlib>
#include <cfloat>
#include <cstdio>

float DistanceSquaredHost(float* points, int numPoints, int pointId, float* centroids, int numClusters, int clusterId, int dimensions);

void KMeansCPU(float* data, int numPoints, int dimensions, int numClusters, int maxIterations, float* centroids, int* assignments)
{
	int changedPoints = 0;

	// Set initial centroids to the first k points
	for (int c = 0; c < numClusters; ++c)
		for (int d = 0; d < dimensions; ++d)
			centroids[d * numClusters + c] = data[d * numPoints + c];

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
				float dist = DistanceSquaredHost(data, numPoints, i, centroids, numClusters, c, dimensions);
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
		if (!centroidSums || !pointsPerCluster)
		{
			fprintf(stderr, "Memory allocation failed.\n");
			return;
		}

		for (int i = 0; i < numPoints; ++i)
		{
			int cluster = assignments[i];
			pointsPerCluster[cluster]++;
			for (int d = 0; d < dimensions; ++d)
			{
				centroidSums[d * numClusters + cluster] += data[d * numPoints + i];
			}
		}

		for (int c = 0; c < numClusters; ++c)
		{
			if (pointsPerCluster[c] == 0) continue;
			for (int d = 0; d < dimensions; ++d)
			{
				centroids[d * numClusters + c] = centroidSums[d * numClusters + c] / pointsPerCluster[c];
			}
		}

		free(centroidSums);
		free(pointsPerCluster);
	}
}