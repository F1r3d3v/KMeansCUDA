#ifndef KMEANS_H
#define KMEANS_H

#include "cuda_runtime.h"

void KMeansCPU_Runner(float* data, int numPoints, int dimensions, int numClusters, int maxIterations, float* centroids, char* assignments);
cudaError_t KMeansGPU1_Runner(float* data, int numPoints, int dimensions, int numClusters, int maxIterations, float* centroids, char* assignments);
cudaError_t KMeansGPU2_Runner(float* data, int numPoints, int dimensions, int numClusters, int maxIterations, float* centroids, char* assignments);
void EmptyCUDACall();

#endif // !KMEANS_H

