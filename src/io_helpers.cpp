#include <stdio.h>
#include <stdlib.h>
#include "io_helpers.h"

int loadFileTxt(const char* filename, int* numPoints, int* dimensions, int* numClusters, float** data)
{
	FILE* file = fopen(filename, "r");
	if (!file)
	{
		fprintf(stderr, "Failed to open file: %s\n", filename);
		return 0;
	}

	if (fscanf(file, "%d %d %d", numPoints, dimensions, numClusters) != 3)
	{
		fprintf(stderr, "Failed to read data from file: %s\n", filename);
		fclose(file);
		return 0;
	}

	// Allocate memory for data: numPoints * dimensions
	*data = (float*)malloc((*numPoints) * (*dimensions) * sizeof(float));
	if (*data == NULL)
	{
		fprintf(stderr, "Memory allocation failed.\n");
		fclose(file);
		return 0;
	}

	// Load data points
	for (int i = 0; i < *numPoints; ++i)
	{
		for (int d = 0; d < *dimensions; ++d)
		{
			if (fscanf(file, "%f", &((*data)[i * (*dimensions) + d])) != 1)
			{
				fprintf(stderr, "Failed to read data point from file: %s\n", filename);
				free(*data);
				fclose(file);
				return 0;
			}
		}
	}

	fclose(file);
	return 1;
}

int loadFileBin(const char* filename, int* numPoints, int* dimensions, int* numClusters, float** data)
{
	FILE* file = fopen(filename, "rb");
	if (!file)
	{
		fprintf(stderr, "Failed to open file: %s\n", filename);
		return 0;
	}

	fread(numPoints, sizeof(int), 1, file);
	fread(dimensions, sizeof(int), 1, file);
	fread(numClusters, sizeof(int), 1, file);

	// Allocate memory for data: numPoints * dimensions
	*data = (float*)malloc((*numPoints) * (*dimensions) * sizeof(float));
	if (*data == NULL)
	{
		fprintf(stderr, "Memory allocation failed.\n");
		fclose(file);
		return 0;
	}

	// Load data points
	fread(*data, sizeof(float), (*numPoints) * (*dimensions), file);

	fclose(file);
	return 1;
}

int writeFileTxt(const char* filename, float* centroids, int numClusters, int dimensions, int* assignments, int numPoints)
{
	FILE* file = fopen(filename, "w");
	if (!file)
	{
		fprintf(stderr, "Failed to open output file: %s\n", filename);
		return 0;
	}

	// Write centroids
	for (int i = 0; i < numClusters; ++i)
	{
		for (int d = 0; d < dimensions; ++d)
			fprintf(file, "  %8.4f", centroids[i * dimensions + d]);

		fprintf(file, "\n");
	}

	// Write cluster assignments
	for (int i = 0; i < numPoints; ++i)
		fprintf(file, "  %d\n", assignments[i]);

	fclose(file);
	return 1;
}