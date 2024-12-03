#ifndef IO_HELPERS_H
#define IO_HELPERS_H

int loadFileTxt(const char* filename, int* n, int* dim, int* k, float** data);
int loadFileBin(const char* filename, int* n, int* dim, int* k, float** data);
int writeFileTxt(const char* filename, float* centroids, int k, int dim, int* assignments, int n);

#endif // !IO_HELPERS_H