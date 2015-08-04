
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CArray.cuh"

#include <stdio.h>

void CalculateGistograme(int* image, int width, int height, int* res)
{
	CArray<int> source = CArray<int>(image, width, height);
}