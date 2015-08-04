#ifndef CARRAY
#define CARRAY

#include "cuda_runtime.h"

template <typename T> class CArray
{
private:
	size_t deviceStride;
public:
	T* cudaPtr;
	size_t Height;
	size_t Width;
	size_t Stride;

	CArray(T* cpuPtr, int width, int height)  
	{
		Width = width;
		Height = height;
		cudaError_t error = cudaMallocPitch((void**)&cudaPtr, &Stride, Width*sizeof(T), Height);
		error = cudaDeviceSynchronize();
		deviceStride = Stride / sizeof(T);
		error = cudaMemcpy2D(cudaPtr, Stride, cpuPtr, Width*sizeof(T),
			Width*sizeof(T), Height, cudaMemcpyHostToDevice);
		error = cudaDeviceSynchronize();
		error = cudaGetLastError();
	}

	void GetData(T* arr)
	{
		cudaError_t error = cudaMemcpy2D(arr, Width*sizeof(T), cudaPtr, Stride, Width*sizeof(T), Height, cudaMemcpyDeviceToHost);
		error = cudaDeviceSynchronize();
	}

	__device__ T At(int row, int column)
	{
		return cudaPtr[row*deviceStride + column];
	}

	__device__ void SetAt(int row, int column, T value)
	{
		cudaPtr[row*deviceStride + column] = value;
	}

	void Dispose()
	{
		cudaFree(cudaPtr);
	}
};

template class CArray<int>;

#endif