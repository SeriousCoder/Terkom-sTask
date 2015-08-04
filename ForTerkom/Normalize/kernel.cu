
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CArray.cuh"

#include <stdio.h>
#include <stdlib.h>

extern "C"
{
	__declspec(dllexport) int* CalculateHistograme(int* source, int imgWidth, int imgHeight, int* res);
}

__global__ void cudaCalculateGistograme(CArray<int> image, int* histArray)
{
	int column = blockIdx.x * blockDim.x + threadIdx.x;

	int tempIndex = threadIdx.x;
	__shared__ int temp[32][256];
	__shared__ int height;
	__shared__ int width;

	height = image.Height;
	width = image.Width;

	if (width > column)
	{
		for (int j = 0; j < height; j++)
		{
			temp[tempIndex][image.At(j, column)]++;
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

int* CalculateHistograme(int* image, int width, int height, int* res)
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
		for (int j = 0; j < 255; j++)
		{
			resHist[j] = gpu_hist[i * 256 + j];
		}
	}

	free(hist);
	free(resHist);

	return resHist;
}