#ifndef LIB_HEADER_H
#define LIB_HEADER_H

#include <stdio.h>
#include <fstream>
#include <string>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "float.h"

__constant__ int AGENT_NO_D;	//copied from host
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

int BOARDER_L_H;	//read from config
int BOARDER_R_H;	//read from config
int BOARDER_U_H;	//read from config
int BOARDER_D_H;	//read from config
int AGENT_NO;		//read from config
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
	int id;
	float2d_t loc;
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
#define BOID_DEBUG

#endif