#include <stdio.h>
#include <stdlib.h>
#include "lodepng.h"
#include <cuda_runtime.h>


__device__ __forceinline__ unsigned char clamp255(int v) {
    if (v < 0) return 0;
    if (v > 255) return 255;
    return (unsigned char)v;
}

//each pixel has 4 values (0 - 255)
//R - RED
//G - GREEN
//B - BLUE
//Transparency (Alpha)


__global__ void Sobel(unsigned char *inputImage, unsigned char *outputImage,
                      unsigned int width, unsigned int height) {

	int threadID = blockDim.x * blockIdx.x + threadIdx.x;
	int pixel = threadID * 4;
	int x = threadIdx.x;
    int y = blockIdx.x;  
	if (x >= (int)width || y >= (int)height) return;
	int a = inputImage[pixel + 3];

	auto grayAt = [&](int nx, int ny) -> int {
		if (nx < 0 || nx >= (int)width || ny < 0 || ny >= (int)height) return 0;
		int id = ny * (int)width + nx;
        int p = id * 4;
        int r = inputImage[p];
		int g = inputImage[p+1];
        int b = inputImage[p+2];
		return (299*r + 587*g + 114*b) / 1000;
	};

	int p00 = grayAt(x-1, y-1), p01 = grayAt(x, y-1), p02 = grayAt(x+1, y-1);
    int p10 = grayAt(x-1, y  ), p11 = grayAt(x, y  ), p12 = grayAt(x+1, y  );
    int p20 = grayAt(x-1, y+1), p21 = grayAt(x, y+1), p22 = grayAt(x+1, y+1);


	int gx =
        (-1 * p00) + (0 * p01) + (1 * p02) +
        (-2 * p10) + (0 * p11) + (2 * p12) +
        (-1 * p20) + (0 * p21) + (1 * p22);

    int gy =
        (-1 * p00) + (-2 * p01) + (-1 * p02) +
        ( 0 * p10) + ( 0 * p11) + ( 0 * p12) +
        ( 1 * p20) + ( 2 * p21) + ( 1 * p22);

    int mag = abs(gx) + abs(gy);
    unsigned char edge = clamp255(mag);


	outputImage[pixel] = edge; 		//R
	outputImage[pixel+1] = edge; 	//G
	outputImage[pixel+2] = edge; 	//B
	outputImage[pixel+3] = a; 		//A
	}

	int main(int argc, char ** argv){
	
	//decode
	//process
	//encode

	unsigned int errorDecode;
	unsigned char* cpuImage; //hold image values
	unsigned int width, height;

	char * filename = argv[1];
	char *newFilename = argv[2];
	
	errorDecode = lodepng_decode32_file(&cpuImage, &width, &height, filename);
	
	if(errorDecode){
		printf("error %u: %s\n", errorDecode, lodepng_error_text(errorDecode));
	}
	
	int arraySize = width*height*4;
	int memorySize = arraySize * sizeof(unsigned char);
	unsigned char* gpuInput;
	unsigned char* gpuOutput;
	
	unsigned char *cpuOutput = (unsigned char*)malloc(memorySize);
	if (!cpuOutput) return 1;
	cudaMalloc( (void**) &gpuInput, memorySize);
	cudaMalloc( (void**) &gpuOutput, memorySize);
	
	cudaMemcpy(gpuInput, cpuImage, memorySize, cudaMemcpyHostToDevice);

	Sobel<<< dim3(height,1,1), dim3(width,1,1) >>>(gpuInput, gpuOutput, width, height);
	cudaDeviceSynchronize();
	
	cudaMemcpy(cpuOutput, gpuOutput, memorySize, cudaMemcpyDeviceToHost);
	
	lodepng_encode32_file(newFilename, cpuOutput, width, height);
	
	cudaFree(gpuInput);
	cudaFree(gpuOutput);
	free(cpuOutput);
	free(cpuImage);

	return 0;
}
