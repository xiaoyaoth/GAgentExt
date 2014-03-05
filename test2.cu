#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"
#include "thrust\scan.h"
#include "thrust\sort.h"
#include "thrust\device_ptr.h"
#include "thrust\device_vector.h"
#include <stdio.h>

#define AGENTNO 102400
#define BUFFERSIZE 204800
#define BLOCK_SIZE 128
#define DICE 0.9
#define VERBOSE 0

#define checkCudaErrors(err)	__checkCudaErrors(err, __FILE__, __LINE__)
inline void __checkCudaErrors( cudaError err, const char *file, const int line )
{
	if( cudaSuccess != err) {
		fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n", file, line, (int)err, cudaGetErrorString( err ) );
		exit(-1);
	}
}

#define getLastCudaError(msg)	__getLastCudaError (msg, __FILE__, __LINE__)
inline void __getLastCudaError( const char *errorMessage, const char *file, const int line )
{
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) {
		fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n", file, line, errorMessage, (int)err, cudaGetErrorString( err ) );
		system("PAUSE");
		exit(-1);
	}
}

__device__ int numAg;

class GAgent{
public:
	int id;
	curandState rndState;
	__device__ GAgent(){
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		curand_init(2345, idx, 0, &rndState);
	}
};

__global__ void init(GAgent **alist){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	GAgent *ag = new GAgent();
	ag->id = idx;
	alist[idx] = ag;
}

__device__ int atomicAdd1(int* address, int val)
{
	unsigned int old = *address, assumed;
	do {
		assumed = old;
		old = atomicCAS(address, assumed, (val + assumed));
	} while (assumed != old);
	return old;
}

__global__ void insert(GAgent **alist){
	//int idx = threadIdx.x + blockIdx.x * blockDim.x;
	//GAgent *ag = alist[idx];
	//float dice = curand_uniform(&ag->rndState);
	//if (dice < DICE) {
	//	GAgent *newAg = new GAgent();
		int newIdx = atomicInc((unsigned int *)&numAg, BUFFERSIZE+1);
	//	newAg->id = newIdx;

	//	alist[newIdx] = newAg;
	//}
	//float test = dice;
}

__global__ void check(GAgent **alist) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < numAg) {
		GAgent *ag = alist[idx];
		float dice = curand_uniform(&ag->rndState);
	}
}

int main1()
{
	int num_h = AGENTNO;
	int GRID_SIZE = (int)(AGENTNO/BLOCK_SIZE);
	cudaMemcpyToSymbol(numAg, &num_h, sizeof(int), 0, cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	GAgent **a_dev;
	cudaMalloc((void**)&a_dev, BUFFERSIZE*sizeof(GAgent*));
	cudaMemset(a_dev, 0, BUFFERSIZE*sizeof(GAgent*));

	//init<<<GRID_SIZE, BLOCK_SIZE>>>(a_dev);

	cudaEventRecord(start, 0);
	insert<<<GRID_SIZE, BLOCK_SIZE>>>(a_dev);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float time = 1.0;
	cudaEventElapsedTime(&time, start, stop);
	printf("time: %f\n", time);
	cudaMemcpyFromSymbol(&num_h, numAg, sizeof(int), 0, cudaMemcpyDeviceToHost);
	printf("numAg: %d\n", num_h);

	//GRID_SIZE = (int)(BUFFERSIZE/BLOCK_SIZE);
	//check<<<GRID_SIZE, BLOCK_SIZE>>>(a_dev);

	system("PAUSE");
	return 0;
}

__global__ void scanInit(int *a_dev){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	a_dev[idx] = 1;
}

__global__ void scanInsert(){
	GAgent *ag = new GAgent();
}

int main2(){
	int *a_dev;
	checkCudaErrors(cudaMalloc((void**)&a_dev, AGENTNO * sizeof(int)));
	thrust::device_ptr<int> a_ptr(a_dev);
	thrust::device_vector<int>::iterator key_begin(a_ptr);
	thrust::device_vector<int>::iterator key_end(a_ptr + AGENTNO);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int GRID_SIZE = AGENTNO/BLOCK_SIZE;
	int SMEM_SIZE = BLOCK_SIZE * sizeof(int);
	scanInit<<<GRID_SIZE, BLOCK_SIZE>>>(a_dev);

	cudaEventRecord(start, 0);
	//scanInsert<<<GRID_SIZE, BLOCK_SIZE>>>();
	thrust::sort(key_begin, key_end);
	thrust::inclusive_scan(key_begin, key_end, a_ptr);
	cudaEventRecord(stop, 0);  
	cudaEventSynchronize(stop);

	float insertTime = 0;
	cudaEventElapsedTime(&insertTime, start, stop);
	printf("insert time: %f\n", insertTime);

	int *a_host = (int*)malloc(AGENTNO * sizeof(int));
	cudaMemcpy(a_host, a_dev, sizeof(int) * AGENTNO, cudaMemcpyDeviceToHost);
	printf("%d ", a_host[AGENTNO-1]);
	system("PAUSE");
	return 0;
}