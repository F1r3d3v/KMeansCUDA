#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#include <memory>

// Custom deleter for CUDA memory management
template <typename T>
struct CudaDeleter {
    void operator()(T* ptr) const {
        if (ptr)
        {
            cudaError_t err = cudaFree(ptr);
            if (err != cudaSuccess)
            {
				fprintf(stderr, "CUDA free failed: %s. In file '%s' on line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
            }
        }
    }
};

// Define a unique_ptr type with the custom deleter for CUDA
template <typename T>
using CudaUniquePtr = std::unique_ptr<T, CudaDeleter<T>>;

// Helper function to allocate CUDA memory and wrap it in unique_ptr
template <typename T>
CudaUniquePtr<T> allocateCudaMemory(size_t size) {
    T* rawPtr = nullptr;
    cudaError_t err = cudaMalloc(&rawPtr, size * sizeof(T));
    if (err != cudaSuccess) return nullptr;
    return CudaUniquePtr<T>(rawPtr);
}

#define CUDACHECK(err)                                                                                                        \
do                                                                                                                            \
{                                                                                                                             \
    if ((err) != cudaSuccess)                                                                                                 \
    {                                                                                                                         \
        fprintf(stderr, "CUDA Error %d: %s. In file '%s' on line %d\n", (err), cudaGetErrorString((err)), __FILE__, __LINE__);\
        return (err);                                                                                                         \
    }                                                                                                                         \
} while (false)

//https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/
template<typename index_t>
__device__ index_t atomicAggInc(index_t* ctr)
{
    int lane = threadIdx.x % 32;
    //check if thread is active
    int mask = __ballot_sync(0xffffffff, 1);
    //determine first active lane for atomic add
    int leader = __ffs(mask) - 1;
    index_t res;
    if (lane == leader) res = atomicAdd(ctr, __popc(mask));
    //broadcast to warp
    res = __shfl_sync(0xffffffff, res, leader);
    //compute index for each thread
    return res + __popc(mask & ((1 << lane) - 1));
}

#endif // !CUDA_HELPER_H
