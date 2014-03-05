#include <assert.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <cstring>

#ifndef gpuAssert
#include <stdio.h>
#define gpuAssert( condition ) { \
	if( (condition) != 0 ) { \
		fprintf( stderr, "\n FAILURE %s in %s, line %d\n", \
			cudaGetErrorString(condition), __FILE__, __LINE__ ) ; \
		exit( 1 ); } }
#endif

enum boid_type {PREY_BOID, PREDATOR_BOID, FOOD_BOID};

__device__ float* data;

__global__ void test_kernel()
{
	const int tid = threadIdx.x;
	data[tid]=tid;
}

int main1(void)
{
	boid_type type = PREY_BOID;
	printf("enum: %d\n", type);
	const int NUMBER = 1024;
	const size_t size = NUMBER*sizeof(float);
	
	float *data_h = (float*)malloc(size);

	float *data_d;
	float *data_symbol;
	cudaMalloc(&data_d, size);

	//gpuAssert( cudaGetSymbolAddress((void**)&data_symbol, "data"));
	gpuAssert( cudaMemcpyToSymbol( data, &data_d, sizeof(data_d), 0, cudaMemcpyHostToDevice ) );

	test_kernel<<<1, 1024>>>();

	gpuAssert( cudaMemcpy( data_h, data_d, size, cudaMemcpyDeviceToHost ) );
	for (int i = 0; i < NUMBER; i++)
		printf("%f ", data_h[i]);
	
	return (int)cudaThreadExit();
	
}