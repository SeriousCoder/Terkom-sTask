
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CArray.cuh"

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
	int temp[32][256];
	__shared__ int height;
	__shared__ int width;

	height = image.Height;
	width = image.Width;

	if (width > column)
	{
		for (int j = 0; j < height; j++)
		{
			int val = image.At(j, column);
			temp[tempIndex][val]++;
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
				temp[tempIndex][j] += temp[tempIndex + i][j];
			}
		}
		i /= 2;
	}
	if (tempIndex == 0)
	{
		for (int j = 0; j < 256; j++)
		{
			histArray[blockIdx.x * 256 + j] = temp[0][j];
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

	int* resHist = (int*)malloc(sizeof(int) * 256);

	for (int i = 1; i < gridSize.x; i++)
	{
		for (int j = 0; j < 256; j++)
		{
			resHist[j] = gpu_hist[i * 256 + j];
		}
	}

	res = resHist;

	free(hist);
	free(resHist);
}

__global__ void cudaNormalize(CArray<int> image, int histMax, int histMin)
{
	int column = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;

	__shared__ int height;
	__shared__ int width;
	__shared__ int diffHist;

	height = image.Height;
	width = image.Width;
	diffHist = histMax - histMin;

	if (width > column && height > row)
	{
		int valImg = image.At(row, column);
		int val = valImg - histMin;
		int div = val / diffHist;
		image.SetAt(row, column, 255 * div);
	}
}

void Normalize(int* image, int width, int height, int histMax, int histMin)
{
	CArray<int> source = CArray<int>(image, width, height);

	dim3 blockSize = dim3(32);
	dim3 gridSize = dim3((height + 31) / 32);

	cudaNormalize << <gridSize, blockSize >> > (source, histMax, histMin);

	source.GetData(image);
}