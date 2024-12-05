#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kmeans.h"
#include "io_helpers.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <filesystem>
#include <chrono>

namespace fs = std::filesystem;

constexpr int MAX_ITER = 100;

void printUsage(const char* argv0) {
	const char* help =
		"Usage: %s data_format computation_method input_file output_file\n\n"
		"Where:\n"
		"       data_format: txt or bin\n"
		"       computation_method: gpu1, gpu2, or cpu\n"
		"       input_file: path to input data file\n"
		"       output_file: path to output result file\n";
	fprintf(stderr, help, argv0);
}

int main(int argc, char** argv) {
	if (argc != 5) {
		printUsage(argv[0]);
		return 1;
	}

	// Parse command-line arguments
	const char* dataFormat = argv[1];         // txt or bin
	const char* computationMethod = argv[2];  // gpu1, gpu2, or cpu
	fs::path inputFile = argv[3];
	fs::path outputFile = argv[4];

	// Validation
	if (strcmp(dataFormat, "txt") && strcmp(dataFormat, "bin")) {
		printUsage(argv[0]);
		return 1;
	}
	if (strcmp(computationMethod, "gpu1") && strcmp(computationMethod, "gpu2") && strcmp(computationMethod, "cpu")) {
		printUsage(argv[0]);
		return 1;
	}
	if (!fs::exists(inputFile)) {
		printf("Input file does not exist\n");
		return 1;
	}

	printf("Running K-means with configuration:\n");
	printf("Data format: %s\n", dataFormat);
	printf("Computation method: %s\n", computationMethod);
	printf("Input file: %s\n", inputFile.filename().string().c_str());
	printf("Output file: %s\n", outputFile.filename().string().c_str());
	printf("\n");

	std::chrono::steady_clock::time_point start, end;
	int numPoints = 0, dimensions = 0, numClusters = 0;
	float* data = nullptr;

	// Load data
	printf("Loading data from file...\n");
	start = std::chrono::high_resolution_clock::now();
	if (!strcmp(dataFormat, "txt"))
	{
		if (!loadFileTxt(inputFile.string().c_str(), &numPoints, &dimensions, &numClusters, &data))
		{
			fprintf(stderr, "Failed to load data from file: %s\n", inputFile.string().c_str());
			return 1;
		}
	}
	else
	{
		if (!loadFileBin(inputFile.string().c_str(), &numPoints, &dimensions, &numClusters, &data))
		{
			fprintf(stderr, "Failed to load data from file: %s\n", inputFile.string().c_str());
			return 1;
		}
	}
	end = std::chrono::high_resolution_clock::now();
	printf("Data loaded: %d points, %d dimensions, %d clusters\n", numPoints, dimensions, numClusters);
	printf("Data loading time: %lld ms\n\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

	// Allocate memory for centroids and assignments
	float* centroids = (float*)malloc(numClusters * dimensions * sizeof(float));
	int* assignments = (int*)malloc(numPoints * sizeof(int));
	if (!centroids || !assignments)
	{
		fprintf(stderr, "Memory allocation failed.\n");
		return 1;
	}
	memset(assignments, 0, numPoints * sizeof(int));

	// Empty CUDA call to initialize CUDA context
	if (!strcmp(computationMethod, "gpu1") || !strcmp(computationMethod, "gpu2"))
	{
		printf("Initializing CUDA context...\n\n");
		EmptyCUDACall();
	}

	// Run K-means
	printf("Starting K-means computation...\n");
	start = std::chrono::high_resolution_clock::now();
	cudaError_t cudaStatus = cudaSuccess;
	if (!strcmp(computationMethod, "cpu"))
	{
		KMeansCPU(data, numPoints, dimensions, numClusters, MAX_ITER, centroids, assignments);
	}
	else if (!strcmp(computationMethod, "gpu1"))
	{
		cudaStatus = KMeansGPU1(data, numPoints, dimensions, numClusters, MAX_ITER, centroids, assignments);
	}
	else if (!strcmp(computationMethod, "gpu2"))
	{
		cudaStatus = KMeansGPU2(data, numPoints, dimensions, numClusters, MAX_ITER, centroids, assignments);
	}
	end = std::chrono::high_resolution_clock::now();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "K-means computation failed\n");
		return 1;
	}
	printf("K-means computation completed\n");
	printf("Computation time: %lld ms\n\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

	// Create output directory if it does not exist
	fs::create_directories(outputFile.parent_path());

	// Write results
	printf("Writing results to file...\n");
	start = std::chrono::high_resolution_clock::now();
	if (!writeFileTxt(outputFile.string().c_str(), centroids, numClusters, dimensions, assignments, numPoints))
	{
		fprintf(stderr, "Failed to write results to file: %s\n", outputFile.string().c_str());
		return 1;
	}
	end = std::chrono::high_resolution_clock::now();
	printf("Results written to file\n");
	printf("Writing time: %lld ms\n\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

	// Cleanup
	free(data);
	free(centroids);
	free(assignments);

	return 0;
}