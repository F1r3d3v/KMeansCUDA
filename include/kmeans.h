#ifndef KMEANS_H
#define KMEANS_H

#include "cuda_runtime.h"

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);
void KMeansCPU(float* data, int numPoints, int dimensions, int numClusters, int maxIterations, float* centroids, int* assignments);

#endif // !KMEANS_H

