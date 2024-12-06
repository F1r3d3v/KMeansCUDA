#ifndef RENDERER_H
#define RENDERER_H

#include "cuda_runtime.h"

cudaError_t DrawVisualization(float* data, char* assignments, int numPoints, unsigned char* buffer, int width, int height, int axisMax, int margin, int pointSize);

#endif // !RENDERER_H
