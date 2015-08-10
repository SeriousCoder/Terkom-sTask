
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CArray.cuh"
#include "ImageLoading.cuh"

#include <stdio.h>
#include <stdlib.h>

extern "C"
{
	__declspec(dllexport) void CalculateHistograme(int* source, int imgWidth, int imgHeight, int* res);
	__declspec(dllexport) void Normalize(int* image, int width, int height, int histMax, int histMin);
}

__global__ void cudaCalculateGistograme(CArray<int> image, int* histArray)
{
	int column = blockIdx.x * blockDim.x + threadIdx.x;

	int tempIndex = threadIdx.x;
	int temp[32 * 256];
	__shared__ int height;
	__shared__ int width;

	height = image.Height;
	width = image.Width;

	if (width > column)
	{
		for (int j = 0; j < height; j++)
		{
			int val = image.At(j, column);
			temp[tempIndex * 256 + val]++;
		}
	}

	__syncthreads();

	int i = blockDim.x / 2;
	while (i != 0)
	{
		if (tempIndex < i)
		{
			for (int j = 0; j < 256; j++)
			{
				temp[tempIndex * 256 + j] += temp[(tempIndex + i) * 256 + j];
			}
		}
		i /= 2;
	}
	if (tempIndex == 0)
	{
		for (int j = 0; j < 256; j++)
		{
			histArray[blockIdx.x * 256 + j] = temp[j];
		}
	}

}

void CalculateHistograme(int* image, int width, int height, int* res)
{
	CArray<int> source = CArray<int>(image, width, height);

	dim3 blockSize = dim3(32);
	dim3 gridSize = dim3((height + 31) / 32);

	int* hist = (int*)malloc(sizeof(int) * gridSize.x * 256);

	int* gpu_hist;
	cudaMalloc((void**)&gpu_hist, sizeof(int) * gridSize.x * 256);

	cudaCalculateGistograme << <gridSize, blockSize >> > (source, gpu_hist);

	cudaMemcpy(hist, gpu_hist, sizeof(int) * gridSize.x * 256, cudaMemcpyDeviceToHost);

	for (int i = 0; i < gridSize.x; i++)
	{
		for (int j = 0; j < 256; j++)
		{
			if (i == 0)
			{
				res[j] = hist[i * 256 + j];
			}
			else
			{
				res[j] += hist[i * 256 + j];
			}
		}
	}

	free(hist);
}

__global__ void cudaNormalize(CArray<int> image, int* normalizeTable)
{
	int column = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;

	__shared__ int height;
	__shared__ int width;
	__shared__ int diffHist;

	height = image.Height;
	width = image.Width;

	if (width > column && height > row)
	{
		int valImg = image.At(row, column);
		image.SetAt(row, column, normalizeTable[valImg]);
	}
}

void Normalize(int* image, int width, int height, int* normalizeTable)
{
	CArray<int> source = CArray<int>(image, width, height);

	dim3 blockSize = dim3(32);
	dim3 gridSize = dim3((height + 31) / 32);

	cudaNormalize << <gridSize, blockSize >> > (source, normalizeTable);

	source.GetData(image);
}

//void main()
//{
//	int width;
//		int height;
//		char* filename = "Sample.bmp";  //Write your way to bmp file
//		int* img = loadBmp(filename, &width, &height);
//		int res[256];
//		CalculateHistograme(img, width, height, res);
//	
//		free(res);
//		free(img);
//}