#ifndef LIB_HEADER_H
#define LIB_HEADER_H

#include <stdio.h>
#include <fstream>
#include <string>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include "float.h"

__constant__ int MAX_AGENT_NO_D;//copied from host
__device__ unsigned int AGENT_NO_D;		//copied from host
__device__ unsigned int AGENT_NO_INC_D;	//manipulated on device
__device__ unsigned int AGENT_NO_DEC_D;	//manipulated on device
__constant__ int CELL_NO_D;		//copied from host
__constant__ int BOARDER_L_D;	//copied from host
__constant__ int BOARDER_R_D;	//copied from host
__constant__ int BOARDER_U_D;	//copied from host
__constant__ int BOARDER_D_D;	//copied from host
__constant__ int CNO_PER_DIM;	//(int)pow((float)2, DISCRETI)
__constant__ float CLEN_X;		//(float)(BOARDER_R-BOARDER_L)/CNO_PER_DIM;
__constant__ float CLEN_Y;		//(float)(BOARDER_D-BOARDER_U)/CNO_PER_DIM;
__constant__ float RANGE;		//read from config

int CELL_NO;		//CNO_PER_DIM * CNO_PER_DIM;
int DISCRETI;		//read from config
float RANGE_H;		//read from config
size_t HEAP_SIZE;	//read from config
size_t STACK_SIZE;	//read from config

int BOARDER_L_H;	//read from config
int BOARDER_R_H;	//read from config
int BOARDER_U_H;	//read from config
int BOARDER_D_H;	//read from config
int AGENT_NO;		//read from config
int MAX_AGENT_NO;	//read from config
int STEPS;			//read from config
int SELECTION;		//read from config
bool VISUALIZE;		//read from config
int VERBOSE;		//read from config
int FILE_GEN;		//read from config
char* dataFileName; //read from config

int BLOCK_SIZE;		//read from config
int GRID_SIZE;		//calc with BLOCK_SIZE and AGENT_NO

typedef struct int_2d
{
	int x;
	int y;

	__device__ int cell_id(){
		return y * CNO_PER_DIM + x;
	}
	__device__ __host__ int zcode(){
		int xt = x;
		int yt = y;
		xt &= 0x0000ffff;					// x = ---- ---- ---- ---- fedc ba98 7654 3210
		yt &= 0x0000ffff;					// x = ---- ---- ---- ---- fedc ba98 7654 3210
		xt = (xt ^ (xt << 8)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
		yt = (yt ^ (yt << 8)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
		yt = (yt ^ (yt << 4)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
		xt = (xt ^ (xt << 4)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
		yt = (yt ^ (yt << 2)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
		xt = (xt ^ (xt << 2)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
		yt = (yt ^ (yt << 1)) & 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
		xt = (xt ^ (xt << 1)) & 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0

		return xt | (yt << 1);
	}
	__device__ __host__ void print(){
		printf("(%d, %d)", x, y);
	}
} int2d_t;
typedef struct float_2d
{
	float x;
	float y;

	__device__ __host__ float distance (float_2d p){
		return sqrt((p.x-x)*(p.x-x)+(p.y-y)*(p.y-y));
	}
	__device__ __host__ void print(){
		printf("(%f, %f)", x, y);
	}
} float2d_t;
typedef struct GAgentData{
	float2d_t loc;
	__device__ virtual ~GAgentData(){}
} GAgentData_t;

namespace SCHEDULE_CONSTANT{
	static const float EPOCH = 0.0;
	static const float BEFORE_SIMULATION = EPOCH - 1.0;
	static const float AFTER_SIMULTION = FLT_MAX;
	static const float EPSILON = 1.0;
}

#define checkCudaErrors(err)	__checkCudaErrors(err, __FILE__, __LINE__)
inline void __checkCudaErrors( cudaError err, const char *file, const int line )
{
	if( cudaSuccess != err) {
		fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
			file, line, (int)err, cudaGetErrorString( err ) );
		exit(-1);
	}
}
// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg)	__getLastCudaError (msg, __FILE__, __LINE__)
inline void __getLastCudaError( const char *errorMessage, const char *file, const int line )
{
	cudaError_t err = cudaGetLastError();
	if( VERBOSE == 1) 
		printf("getLastCudaError: %s: \n\t%d-%s in line %i.\n", errorMessage,
		err, cudaGetErrorString(err), line);
	if( cudaSuccess != err) {
		fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
			file, line, errorMessage, (int)err, cudaGetErrorString( err ) );
		system("PAUSE");
		exit(-1);
	}
}

__device__ float *randDebug;

template<class Obj> class Pool;

namespace poolUtil{
	template<class Obj> __global__ void initPool(Pool<Obj> *pDev);
	template<class Obj> __global__ void cleanupDevice(Pool<Obj> *pDev);
	template<class Obj> void cleanup(Pool<Obj> *pHost, Pool<Obj> *pDev);
};
template<class Obj> class Pool 
{
public:
	/* pointer array, elements are pointers points to elements in data array
	Since pointers are light weighted, it will be manipulated in sorting, etc. */
	int *idxArray;

	/* keeping the actual Objects data */
	Obj *dataArray;

	/* objects to be deleted will be marked as delete */
	bool *delMark;

	/*the pointers in the ptrArray are one-to-one associated with the elem in dataArray.
	No matter what the data is, the pointer points to the same data.
	However, the ptrArray and delMark are one-to-one corresponding, i.e., ptrArray is sorted
	with delMark*/

	unsigned int numElem;
	unsigned int numElemMax;
	unsigned int incCount;
	unsigned int decCount;
public:
	__device__ int assign();
	__device__ void remove(int idx);
	__device__ void link();
	__host__ void alloc(int nElem, int nElemMax);
	__host__ Pool(int nElem, int nElemMax);

	friend __global__ void poolUtil::initPool(Pool<Obj> *pDev);
	friend __global__ void poolUtil::cleanupDevice(Pool<Obj> *pDev);
	friend void poolUtil::cleanup(Pool<Obj> *pHost, Pool<Obj> *pDev);
};

//Pool implementation
template<class Obj> __device__ int Pool<Obj>::assign()
{
	int idx = atomicInc(&incCount, numElemMax-numElem) + numElem;
	this->delMark[idx] = false;
	return this->idxArray[idx];
}
template<class Obj> __device__ void Pool<Obj>::remove(int idx)
{
	delMark[idx] = true;
	atomicInc(&decCount, numElem);
}
template<class Obj> __host__ void Pool<Obj>::alloc(int nElem, int nElemMax){
	printf("sizeof obj in Pool<Obj>::alloc: %d\n", sizeof(Obj));
	this->numElem = nElem;
	this->numElemMax = nElemMax;
	this->incCount = 0;
	this->decCount = 0;
	cudaMalloc((void**)&this->delMark, nElemMax * sizeof(bool));
	cudaMalloc((void**)&this->dataArray, nElemMax * sizeof(Obj));
	cudaMalloc((void**)&this->idxArray, nElemMax * sizeof(int));
	cudaMemset(this->dataArray, 0xff, nElemMax * sizeof(Obj));
	cudaMemset(this->idxArray, 0x00, nElemMax * sizeof(int));
	cudaMemset(this->delMark, 1, nElemMax * sizeof(bool));
}
template<class Obj> __host__ Pool<Obj>::Pool(int nElem, int nElemMax){
	this->alloc(nElem, nElemMax);
}

//poolUtil implementation
template<class Obj> __global__ void poolUtil::initPool(Pool<Obj> *pDev) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < pDev->numElemMax)
		pDev->idxArray[idx] = idx;
}
template<class Obj> __global__ void poolUtil::cleanupDevice(Pool<Obj> *pDev)
{
	pDev->numElem = pDev->numElem + pDev->incCount - pDev->decCount;
	pDev->incCount = 0;
	pDev->decCount = 0;
}
template<class Obj> void poolUtil::cleanup(Pool<Obj> *pHost, Pool<Obj> *pDev)
{
	/**/
	int *ptrArrayLocal = (int*)pHost->idxArray;
	bool *delMarkLocal = pHost->delMark;

	thrust::device_ptr<int> objPtr(ptrArrayLocal);
	thrust::device_ptr<bool> dMarkPtr(delMarkLocal);
	typedef thrust::device_vector<int>::iterator idxIter;
	typedef thrust::device_vector<bool>::iterator dMarkIter;
	dMarkIter keyBegin(dMarkPtr);
	dMarkIter keyEnd(dMarkPtr + pHost->numElemMax);
	idxIter valBegin(objPtr);
	thrust::sort_by_key(keyBegin, keyEnd, valBegin, thrust::less<int>());

	cleanupDevice<<<1, 1>>>(pDev);
	cudaMemcpy(pHost, pDev, sizeof(Pool<Obj>), cudaMemcpyDeviceToHost);
}
#endif