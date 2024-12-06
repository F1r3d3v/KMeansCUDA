#include "renderer.h"
#include "device_launch_parameters.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "cuda_helper.cuh"

#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#define _USE_MATH_DEFINES
#include <math.h>

__device__
glm::i8vec3 GetColor(char assignment)
{
	glm::ivec3 col[] = {
		{255, 0, 0}, // Red
		{0, 255, 0}, // Green
		{0, 0, 255}, // Blue
		{255, 255, 0}, // Yellow
		{255, 0, 255}, // Magenta
		{0, 255, 255}, // Cyan
		{255, 255, 255}, // White
		{128, 0, 0}, // Maroon
		{0, 128, 0}, // Green
		{0, 0, 128}, // Navy
		{128, 128, 0}, // Olive
		{128, 0, 128}, // Purple
		{0, 128, 128}, // Teal
		{128, 128, 128}, // Gray
		{64, 0, 0}, // Brown
		{0, 64, 0}, // Dark Green
		{0, 0, 64}, // Dark Blue
		{64, 64, 0}, // Dark Yellow
		{64, 0, 64}, // Dark Magenta
		{0, 64, 64} // Dark Cyan
	};

	if (assignment >= 0 && assignment < 20)
		return col[assignment];

	return { 0, 0, 0 };
}

__device__ __host__
inline glm::vec3 MakeProjection3D(glm::vec3 p)
{
	glm::mat3 projection(
		0.7071, -0.4082, 0.5774,
		0, 0.8165, 0.5774,
		-0.7071, -0.4082, 0.5774
	);

	return projection * p;
}

__device__ __host__
inline glm::vec2 MakeProjection2D(glm::vec3 p)
{
	glm::vec3 projectedPoint = MakeProjection3D(p);
	return glm::vec2(projectedPoint.x, projectedPoint.y);
}

__global__
void IsometricProjectionKernel(float* data, float* projectedData, int numPoints)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	while (tid < numPoints)
	{
		glm::vec3 projectedPoint = MakeProjection3D({ data[tid], data[tid + numPoints], data[tid + 2 * numPoints] });
		projectedData[tid] = projectedPoint.x;
		projectedData[tid + numPoints] = projectedPoint.y;
		projectedData[tid + 2 * numPoints] = projectedPoint.z;
		tid += stride;
	}
}

__global__
void ApplyScaleKernel(float* data, int numPoints, int height, int margin, glm::vec2 min, glm::vec2 scale)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	while (tid < numPoints)
	{
		// Apply scale
		data[tid] = scale.x * (data[tid] - min.x) + margin;
		data[tid + numPoints] = scale.y * (data[tid + numPoints] - min.y) + margin;
		tid += stride;
	}
}

__global__
void RenderBufferKernel(float* data, char* assignments, int numPoints, unsigned char* buffer, float* zbuffer, int width, int height, int pointSize)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	while (tid < numPoints)
	{
		int x = (int)data[tid];
		int y = (int)data[tid + numPoints];
		float z = data[tid + 2 * numPoints];
		
		// Check z-buffer
		if (zbuffer[y * width + x] > z)
		{
			tid += stride;
			continue;
		}

		// Update z-buffer
		zbuffer[y * width + x] = z;

		// Draw point
		if (x >= 0 && x < width && y >= 0 && y < height)
		{
			glm::i8vec3 color = GetColor(assignments[tid]);

			for (int i = x - pointSize / 2; i < x + pointSize / 2; ++i)
			{
				for (int j = y - pointSize / 2; j < y + pointSize / 2; ++j)
				{
					if (i >= 0 && i < width && j >= 0 && j < height)
					{
						int index = 3 * (j * width + i);
						buffer[index] = color.b;
						buffer[index + 1] = color.g;
						buffer[index + 2] = color.r;
					}
				}
			}
		}

		tid += stride;
	}
}

__host__
void BresenhamLine(unsigned char* buffer, int width, int height, glm::vec2 start, glm::vec2 end, glm::ivec3 color, int thickness)
{
	int x0 = (int)start.x;
	int y0 = (int)start.y;
	int x1 = (int)end.x;
	int y1 = (int)end.y;

	int dx = abs(x1 - x0);
	int dy = abs(y1 - y0);
	int sx = x0 < x1 ? 1 : -1;
	int sy = y0 < y1 ? 1 : -1;
	int err = dx - dy;

	while (true)
	{
		for (int t = -thickness / 2; t <= thickness / 2; ++t)
		{
			int offsetX = (dy > dx) ? t : 0;
			int offsetY = (dx >= dy) ? t : 0;

			int px = x0 + offsetX;
			int py = y0 + offsetY;

			if (px >= 0 && px < width && py >= 0 && py < height)
			{
				int index = 3 * (py * width + px);
				buffer[index] = color.b;
				buffer[index + 1] = color.g;
				buffer[index + 2] = color.r;
			}
		}

		if (x0 == x1 && y0 == y1)
			break;

		int e2 = 2 * err;
		if (e2 > -dy)
		{
			err -= dy;
			x0 += sx;
		}

		if (e2 < dx)
		{
			err += dx;
			y0 += sy;
		}
	}
}

__host__
void ArrowedLine(unsigned char* buffer, int width, int height, glm::vec2 start, glm::vec2 end, glm::ivec3 color, int thickness) {
	// Draw the main line
	BresenhamLine(buffer, width, height, start, end, color, thickness);

	// Calculate the tip size based on the line length
	double tipSize = glm::length(start - end) * 0.025;

	// Calculate the angle of the line
	double angle = atan2(start.y - end.y, start.x - end.x);

	// Draw the first arrow tip
	glm::vec2 tip1 = end + glm::vec2(tipSize * cos(angle + M_PI / 4), tipSize * sin(angle + M_PI / 4));
	BresenhamLine(buffer, width, height, tip1, end, color, thickness);

	// Draw the second arrow tip
	glm::vec2 tip2 = end + glm::vec2(tipSize * cos(angle - M_PI / 4), tipSize * sin(angle - M_PI / 4));
	BresenhamLine(buffer, width, height, tip2, end, color, thickness);
}

cudaError_t DrawVisualization(float* data, char* assignments, int numPoints, unsigned char* buffer, int width, int height, int axisMax, int margin, int pointSize)
{
	cudaError_t cudaStatus;

	// Set cuda device
	cudaStatus = cudaSetDevice(0);
	CUDACHECK(cudaStatus);

	// Allocate device memory
	thrust::device_vector<float> d_data(data, data + numPoints * 3);
	thrust::device_vector<float> d_projectedData(numPoints * 3);
	thrust::device_vector<char> d_assignments(assignments, assignments + numPoints);

	// Project points
	IsometricProjectionKernel << <BLOCKS_PER_GRID(numPoints), THREADS_PER_BLOCK >> > (
		thrust::raw_pointer_cast(d_data.data()),
		thrust::raw_pointer_cast(d_projectedData.data()),
		numPoints);

	cudaStatus = cudaDeviceSynchronize();
	CUDACHECK(cudaStatus);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	CUDACHECK(cudaStatus);

	// Apply scale
	const float MinX = *thrust::min_element(d_projectedData.begin(), d_projectedData.begin() + numPoints);
	const float MaxX = *thrust::max_element(d_projectedData.begin(), d_projectedData.begin() + numPoints);
	const float MinY = *thrust::min_element(d_projectedData.begin() + numPoints, d_projectedData.begin() + 2 * numPoints);
	const float MaxY = *thrust::max_element(d_projectedData.begin() + numPoints, d_projectedData.begin() + 2 * numPoints);
	const float ScaleX = (width - 2 * margin) / (MaxX - MinX);
	const float ScaleY = (height - 2 * margin) / (MaxY - MinY);

	glm::vec2 min(MinX, MinY);
	glm::vec2 scale(ScaleX, ScaleY);

	ApplyScaleKernel << <BLOCKS_PER_GRID(numPoints), THREADS_PER_BLOCK >> > (
		thrust::raw_pointer_cast(d_projectedData.data()),
		numPoints,
		height,
		margin,
		min,
		scale);

	cudaStatus = cudaDeviceSynchronize();
	CUDACHECK(cudaStatus);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	CUDACHECK(cudaStatus);

	// Draw axes
	glm::vec2 center(width / 2, height / 2);
	glm::vec3 XAxis(-axisMax, 0, 0);
	glm::vec3 YAxis(0, -axisMax + 2 * margin, 0);
	glm::vec3 ZAxis(0, 0, -axisMax);

	glm::vec2 XAxisStart = MakeProjection2D(XAxis) + center;
	glm::vec2 YAxisStart = MakeProjection2D(YAxis) + center;
	glm::vec2 ZAxisStart = MakeProjection2D(ZAxis) + center;
	glm::vec2 XAxisEnd = MakeProjection2D(-XAxis) + center;
	glm::vec2 YAxisEnd = MakeProjection2D(-YAxis) + center;
	glm::vec2 ZAxisEnd = MakeProjection2D(-ZAxis) + center;

	ArrowedLine(buffer, width, height, XAxisStart, XAxisEnd, { 255, 0, 0 }, 4);
	ArrowedLine(buffer, width, height, YAxisStart, YAxisEnd, { 0, 255, 0 }, 4);
	ArrowedLine(buffer, width, height, ZAxisStart, ZAxisEnd, { 0, 0, 255 }, 4);

	// Initialize z-buffer
	thrust::device_vector<float> d_zbuffer(width * height);
	thrust::fill(d_zbuffer.begin(), d_zbuffer.end(), FLT_MIN);

	// Render points
	thrust::device_vector<unsigned char> d_buffer(buffer, buffer + width * height * 3);
	RenderBufferKernel << <BLOCKS_PER_GRID(numPoints), THREADS_PER_BLOCK >> > (
		thrust::raw_pointer_cast(d_projectedData.data()),
		thrust::raw_pointer_cast(d_assignments.data()),
		numPoints,
		thrust::raw_pointer_cast(d_buffer.data()),
		thrust::raw_pointer_cast(d_zbuffer.data()),
		width,
		height,
		pointSize);

	cudaStatus = cudaDeviceSynchronize();
	CUDACHECK(cudaStatus);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	CUDACHECK(cudaStatus);

	// Copy image from device to host
	thrust::copy(d_buffer.begin(), d_buffer.end(), buffer);

	return cudaStatus;
}