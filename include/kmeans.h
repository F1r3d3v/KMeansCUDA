#ifndef KMEANS_H
#define KMEANS_H

#include "cuda_runtime.h"

void KMeansCPU(float* data, int numPoints, int dimensions, int numClusters, int maxIterations, float* centroids, int* assignments);
cudaError_t KMeansGPU1(float* data, int numPoints, int dimensions, int numClusters, int maxIterations, float* centroids, int* assignments);
cudaError_t KMeansGPU2(float* data, int numPoints, int dimensions, int numClusters, int maxIterations, float* centroids, int* assignments);
void EmptyCUDACall();

#endif // !KMEANS_H

