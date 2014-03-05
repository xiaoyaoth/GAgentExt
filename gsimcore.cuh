#ifndef GSIMCORE_H
#define GSIMCORE_H

#include "gsimlib_header.cuh"
#include "gsimapp_header.cuh"
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <curand_kernel.h>

//class delaration
class GSteppalbe;
class GAgent;
class GIterativeAgent;
class Continuous2D;
class GScheduler;
class GModel;
class GRandomGen;

typedef struct iter_info_per_thread
{
	int2d_t cellCur;
	int2d_t cellUL;
	int2d_t cellDR;

	int ptr;
	int boarder;
	int count;
	float2d_t myLoc;
	int ptrInSmem;
	int id;

	float range;
} iterInfo;
extern __shared__ int smem[];

namespace c2dUtil{
	__global__ void gen_hash_kernel(int *hash, Continuous2D *c2d);
	void sort_hash_kernel(int *hash, int *neighborIdx);
	__global__ void gen_cellIdx_kernel(int *hash, Continuous2D *c2d);
	void queryNeighbor(Continuous2D *c2d);
	void genNeighbor(Continuous2D *world, Continuous2D *world_h);
	__global__ void swapAgentsInWorld(Continuous2D *world);
};
namespace schUtil{
	void sortWithKey(GModel *model);
	__global__ void scheduleRepeatingAllAgents(GModel *gm);
	__global__ void swapAgentsInScheduler(GModel *gm);
	__global__ void step(GModel *gm);

}
namespace rgenUtil{
	__global__ void initStates(GRandomGen *rgen, int seed);
}

class GSteppable{
public:
	float time;
	int rank;
	virtual __device__ void step(GModel *model) = 0;
};
class GAgent : public GSteppable{
protected:
	GAgent *dummy;
	GAgentData_t *data;
	__device__ int initId();
public:
	__device__ void allocOnDevice();
	__device__ int getId() const;
	__device__ void setDummy(GAgent *dummy);
	__device__ GAgent* getDummy() const;
	__device__ GAgentData_t *getData();
	__device__ float2d_t getLoc() const;
	__device__ virtual void step(GModel *model) = 0;
	__device__ virtual void putDataInSmem(dataUnion &dataElem) = 0;
};
class GIterativeAgent : public GAgent{
private:
	GAgent *realAg;
	float interval;
public:
	__device__ GIterativeAgent(float time, int rank, 
		GAgent *ag, float interval){
			this->realAg = ag;
			this->time = time;
			this->rank = rank;
			this->interval = interval;
	}
	__device__ void step(GModel *model);
	__device__ void putDataInSmem(dataUnion &dataElem){}
};
class Continuous2D{
public:
	float width;
	float height;
	float discretization;
private:
	GAgent **allAgents;
	int *neighborIdx;
	int *cellIdxStart;
	int *cellIdxEnd;
public:
	Continuous2D(float w, float h, float disc){
		this->width = w;
		this->height = h;
		this->discretization = disc;
	}
	void allocOnDevice();
	void allocOnHost();
	//GScheduler helper function
	__device__ const int* getNeighborIdx() const;
	//agent list manipulation
	__device__ bool add(GAgent *ag, int idx);
	__device__ bool remove(GAgent *ag);
	__device__ void swap();
	__device__ GAgent* obtainAgentPerThread() const;
	__device__ GAgent* obtainAgentByInfoPtr(int ptr) const;
	__device__ GAgent* obtainAgent(int agIdx) const;
	//distance utility
	__device__ float stx(float x) const;
	__device__ float sty(float y) const;
	__device__ float tdx(float ax, float bx) const;
	__device__ float tdy(float ay, float by) const;
	__device__ float tds(float2d_t aloc, float2d_t bloc) const;
	//Neighbors related
	__device__ dataUnion* nextNeighborInit2(int agId, float2d_t loc, float range, iterInfo &info) const;
	__device__ void resetNeighborInit(iterInfo &info) const;
	__device__ void calcPtrAndBoarder(iterInfo &info) const;
	__device__ void putAgentDataIntoSharedMem(const iterInfo &info, dataUnion *elem, int tid, int lane) const;
	__device__ dataUnion *nextAgentDataIntoSharedMem(iterInfo &info) const;
	__device__ GAgentData_t *nextAgentData(iterInfo &info) const;
	//__global__ functions
	friend __global__ void c2dUtil::gen_hash_kernel(int *hash, Continuous2D *c2d);
	friend __global__ void c2dUtil::gen_cellIdx_kernel(int *hash, Continuous2D *c2d);
	friend void c2dUtil::genNeighbor(Continuous2D *world, Continuous2D *world_h);
	friend void c2dUtil::queryNeighbor(Continuous2D *c2d);
	//friend class GModel;
};
class GScheduler{
private:
	GAgent **allAgents;
	const int *assignments;
	float time;
	int steps;
public:
	__device__ bool ScheduleOnce(const float time, const int rank,
		GAgent *ag);
	__device__ bool scheduleRepeating(const float time, const int rank, 
		GAgent *ag, const float interval);
	__device__ void setAssignments(const int *newAssignments);
	__device__ GAgent* obtainAgentPerThread() const;
	__device__ GAgent* obtainAgentById(int idx) const;
	__device__ bool add(GAgent* ag, int idx);
	__device__ bool swap();
	void allocOnHost();
	void allocOnDevice();
	friend void schUtil::sortWithKey(GModel *model);
	//friend class GModel;
};
class GRandomGen{
public:
	curandState *states;
public:
	void allocOnDevice();
	__device__ float nextFloat();
	__device__ float nextGaussian();
	__device__ float nextFloat(curandState *state);
	friend __global__ void rgenUtil::initStates(GRandomGen *rgen, int seed);
};
class GModel{
protected:
	GScheduler *scheduler, *schedulerH;
public:
	void allocOnHost();
	void allocOnDevice();
	__device__ GScheduler* getScheduler() const;
	__device__ void addToScheduler(GAgent *ag, int idx);
	__device__ void foo();
	friend void schUtil::sortWithKey(GModel *model);
};

//Continuous2D
void Continuous2D::allocOnDevice(){
	size_t sizeAgArray = AGENT_NO*sizeof(int);
	size_t sizeCellArray = CELL_NO*sizeof(int);

	cudaMalloc((void**)&this->allAgents, AGENT_NO*sizeof(GAgent*));
	getLastCudaError("Continuous2D():cudaMalloc:allAgents");
	cudaMalloc((void**)&neighborIdx, sizeAgArray);
	getLastCudaError("Continuous2D():cudaMalloc:neighborIdx");
	cudaMalloc((void**)&cellIdxStart, sizeCellArray);
	getLastCudaError("Continuous2D():cudaMalloc:cellIdxStart");
	cudaMalloc((void**)&cellIdxEnd, sizeCellArray);
	getLastCudaError("Continuous2D():cudaMalloc:cellIdxEnd");
}
void Continuous2D::allocOnHost(){
	size_t sizeAgArray = AGENT_NO*sizeof(int);
	size_t sizeCellArray = CELL_NO*sizeof(int);
	neighborIdx = (int*)malloc(sizeAgArray);
	cellIdxStart = (int*)malloc(sizeCellArray);
}
__device__ const int* Continuous2D::getNeighborIdx() const{
	return this->neighborIdx;
}
__device__ bool Continuous2D::add(GAgent *ag, int idx) {
	if(idx>=AGENT_NO_D)
		return false;
	this->allAgents[idx]=ag;
	return true;
}
__device__ bool Continuous2D::remove(GAgent *ag){return true;}
__device__ void Continuous2D::swap(){
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx<AGENT_NO_D){
		GAgent *ag = this->allAgents[idx];
		GAgent *dummy = ag->getDummy();
		this->allAgents[idx] = dummy;
	}
}
__device__ GAgent* Continuous2D::obtainAgentPerThread() const {
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	GAgent *ag;
	if (idx < AGENT_NO_D)
		ag = this->allAgents[idx];
	else
		ag = NULL;
	return ag;
}
__device__ GAgent* Continuous2D::obtainAgentByInfoPtr(int ptr) const {
	GAgent *ag;
	if (ptr < AGENT_NO_D && ptr >= 0){
		int agIdx = this->neighborIdx[ptr];
		if (agIdx < AGENT_NO_D && agIdx >=0)
			ag = this->allAgents[agIdx];
		else 
			printf("Continuous2D::obtainAgentByInfoPtr:ptr:%d\n", ptr);
	} else
		ag = NULL;
	return ag;
}
__device__ float Continuous2D::stx(const float x) const{
	float res = x;
	if (x >= 0) {
		if (x >= this->width)
			res = x - this->width;
	} else
		res = x + this->width;
	if (res == this->width)
		res = 0;
	return res;
}
__device__ float Continuous2D::sty(const float y) const {
	float res = y;
	if (y >= 0) {
		if (y >= this->height)
			res = y - this->height;
	} else
		res = y + this->height;
	if (res == this->height)
		res = 0;
	return res;

}
__device__ float Continuous2D::tdx(float ax, float bx) const {
	float dx = abs(ax-bx);
	if (dx < BOARDER_R_D/2)
		return dx;
	else
		return BOARDER_R_D-dx;
}
__device__ float Continuous2D::tdy(float ay, float by) const {
	float dy = abs(ay-by);
	if (dy < BOARDER_D_D/2)
		return dy;
	else
		return BOARDER_D_D-dy;
}
#define SQRT_MAGIC_F 0x5f3759df 
__device__ float Continuous2D::tds(const float2d_t loc1, const float2d_t loc2) const {
	float dx = loc1.x - loc2.x;
	float dxsq = dx*dx;
	float dy = loc1.y - loc2.y;
	float dysq = dy*dy;
	float x = dxsq+dysq;
	return sqrt(x);
}
__device__ dataUnion* Continuous2D::nextNeighborInit2(int agId, float2d_t agLoc, float range, iterInfo &info) const {
	const unsigned int tid	= threadIdx.x;
	const unsigned int idx	= threadIdx.x + blockIdx.x * blockDim.x;

	info.myLoc = agLoc;
	info.ptr = -1;
	info.boarder = -1;
	info.count = 0;
	info.range = range;
	info.ptrInSmem = 0;
	info.id = agId;

	if ((agLoc.x-range)>BOARDER_L_D)	info.cellUL.x = (int)((agLoc.x-range)/CLEN_X);
								else	info.cellUL.x = (int)BOARDER_L_D/CLEN_X;
	if ((agLoc.x+range)<BOARDER_R_D)	info.cellDR.x = (int)((agLoc.x+range)/CLEN_X);
								else	info.cellDR.x = (int)BOARDER_R_D/CLEN_X - 1;
	if ((agLoc.y-range)>BOARDER_U_D)	info.cellUL.y = (int)((agLoc.y-range)/CLEN_Y);
								else	info.cellUL.y = (int)BOARDER_U_D/CLEN_Y;
	if ((agLoc.y+range)<BOARDER_D_D)	info.cellDR.y = (int)((agLoc.y+range)/CLEN_Y);
								else	info.cellDR.y = (int)BOARDER_D_D/CLEN_Y - 1;

	int *cellulx = (int*)smem;
	int *celluly = (int*)&(cellulx[blockDim.x]);
	int *celldrx = (int*)&(celluly[blockDim.x]);
	int *celldry = (int*)&(celldrx[blockDim.x]);

	cellulx[tid]=info.cellUL.x;
	celluly[tid]=info.cellUL.y;
	celldrx[tid]=info.cellDR.x;
	celldry[tid]=info.cellDR.y;

	const unsigned int lane = tid&31;

	for (int i=0; i<32; i++){
#ifdef BOID_DEBUG
		;if (celluly[tid-lane+i] < 0) printf("zhongjian: y: %d, tid-lane+i: %d\n", celluly[tid-lane+i], tid-lane+i);
#endif
		info.cellUL.x = min(info.cellUL.x, cellulx[tid-lane+i]);
		info.cellUL.y = min(info.cellUL.y, celluly[tid-lane+i]);
		info.cellDR.x = max(info.cellDR.x, celldrx[tid-lane+i]);
		info.cellDR.y = max(info.cellDR.y, celldry[tid-lane+i]);
	}

	info.cellCur.x = info.cellUL.x;
	info.cellCur.y = info.cellUL.y;

#ifdef BOID_DEBUG
	if (info.cellCur.x < 0 || info.cellCur.y < 0) {
		printf("xiamian[agId :%d, loc.x: %f, loc.y: %f][xiamian: x: %d, y: %d]\n", agId, agLoc.x, agLoc.y,info.cellUL.x, info.cellUL.y);
	}
#endif

	this->calcPtrAndBoarder(info);
	return NULL;
}
__device__ void Continuous2D::resetNeighborInit(iterInfo &info) const{
	info.ptr = -1;
	info.boarder = -1;
	info.count = 0;
	info.ptrInSmem = 0;
	info.cellCur.x = info.cellUL.x;
	info.cellCur.y = info.cellUL.y;
	this->calcPtrAndBoarder(info);
}
__device__ void Continuous2D::calcPtrAndBoarder(iterInfo &info) const {
	int hash = info.cellCur.zcode();
	if(hash < CELL_NO_D && hash>=0){
		info.ptr = this->cellIdxStart[hash];
		info.boarder = this->cellIdxEnd[hash];
	}
#ifdef BOID_DEBUG
	else {
		printf("x: %d, y: %d, hash: %d\n", info.cellCur.x, info.cellCur.y, hash);
	}
#endif
}
__device__ void Continuous2D::putAgentDataIntoSharedMem(const iterInfo &info, dataUnion *elem, int tid, int lane) const{
	int agPtr = info.ptr + lane;
	if (agPtr <= info.boarder && agPtr >=0) {
		GAgent *ag = this->obtainAgentByInfoPtr(agPtr);
		elem->addValue(ag->getData());
	} else
		elem->id = -1;	
#ifdef BOID_DEBUG
	if (agPtr < -1 || agPtr > AGENT_NO_D + 32){
		printf("Continuous2D::putAgentDataIntoSharedMem: ptr is %d, info.ptr is %d, lane is %d\n", agPtr, info.ptr, lane);
	}
#endif
}
__device__ dataUnion *Continuous2D::nextAgentDataIntoSharedMem(iterInfo &info) const {
	dataUnion *unionArray = (dataUnion*)&smem[4*blockDim.x];
	const int tid = threadIdx.x;
	const int lane = tid & 31;

	if (info.ptr>info.boarder) {
		info.ptrInSmem = 0;
		info.cellCur.x++;
		if(info.cellCur.x>info.cellDR.x){
			info.cellCur.x = info.cellUL.x;
			info.cellCur.y++;
			if(info.cellCur.y>info.cellDR.y)
				return NULL;
		}
		this->calcPtrAndBoarder(info);
	}

	if (info.ptrInSmem == 32)
		info.ptrInSmem = 0;

	if (info.ptrInSmem == 0) {
		dataUnion *elem = &unionArray[tid];
		this->putAgentDataIntoSharedMem(info, elem, tid, lane);
	}
	dataUnion *elem = &unionArray[tid-lane+info.ptrInSmem];
	info.ptrInSmem++;
	info.ptr++;

	while (elem->id == -1 && info.cellCur.y <= info.cellDR.y) {
		info.ptrInSmem = 0;
		info.cellCur.x++;
		if(info.cellCur.x>info.cellDR.x){
			info.cellCur.x = info.cellUL.x;
			info.cellCur.y++;
			if(info.cellCur.y>info.cellDR.y)
				return NULL;
		}
		this->calcPtrAndBoarder(info);
		this->putAgentDataIntoSharedMem(info, &unionArray[tid], tid, lane);
		elem = &unionArray[tid-lane+info.ptrInSmem];
		info.ptrInSmem++;
		info.ptr++;
	}

	if (elem->id == -1) {
		elem = NULL;
	}
	return elem;
}
__device__ GAgentData_t *Continuous2D::nextAgentData(iterInfo &info) const {
	const int tid = threadIdx.x;
	const int lane = tid & 31;

	if (info.ptr>info.boarder) {
		info.ptrInSmem = 0;
		info.cellCur.x++;
		if(info.cellCur.x>info.cellDR.x){
			info.cellCur.x = info.cellUL.x;
			info.cellCur.y++;
			if(info.cellCur.y>info.cellDR.y)
				return NULL;
		}
		this->calcPtrAndBoarder(info);
	}

	while (info.ptr == -1) {
		info.cellCur.x++;
		if(info.cellCur.x>info.cellDR.x){
			info.cellCur.x = info.cellUL.x;
			info.cellCur.y++;
			if(info.cellCur.y>info.cellDR.y)
				return NULL;
		}
		this->calcPtrAndBoarder(info);
	}

	GAgent *ag = this->obtainAgentByInfoPtr(info.ptr);
	info.ptr++;
	return ag->getData();
}
//GAgent
__device__ int GAgent::initId() {
	return threadIdx.x + blockIdx.x * blockDim.x;
}
__device__ void GAgent::allocOnDevice(){
	this->data->id = threadIdx.x + blockIdx.x * blockDim.x;
}
__device__ void GAgent::setDummy(GAgent* dummy){
	this->dummy = dummy;
	dummy->dummy = this;
}
__device__ GAgent* GAgent::getDummy() const {
	return this->dummy;
}
__device__ GAgentData_t *GAgent::getData(){
	return this->data;
}
__device__ float2d_t GAgent::getLoc() const{
	return this->data->loc;
}
__device__ int	GAgent::getId() const {
	return this->data->id;
}

//GIterativeAgent
__device__ void GIterativeAgent::step(GModel *model){
	this->time += this->interval;
	this->realAg->step(model);
	model->getScheduler()->ScheduleOnce(this->time, this->rank, this);
}

//GScheduler
__device__ bool GScheduler::ScheduleOnce(const float time, 	const int rank, 
	GAgent *ag){
		const int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx < AGENT_NO_D){
			ag->time = time;
			ag->rank = rank;
			allAgents[idx] = ag;
		}
		return true;
}
__device__ bool GScheduler::scheduleRepeating(const float time, const int rank, 
	GAgent *ag, const float interval){
		const int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx < AGENT_NO_D){
			GIterativeAgent *iterAg = new GIterativeAgent(time, rank, ag, interval);
			allAgents[idx] = iterAg;
		}
		return true;
}
__device__ void GScheduler::setAssignments(const int* newAs){
	this->assignments = newAs;
}
__device__ GAgent* GScheduler::obtainAgentPerThread() const {
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < AGENT_NO_D) {
		if (this->assignments == NULL) 
			return this->allAgents[idx];
		else
			return this->allAgents[this->assignments[idx]];
	}
	return NULL;
}
__device__ GAgent* GScheduler::obtainAgentById(int idx) const {
	if (idx <AGENT_NO_D)
		return this->allAgents[idx];
	else
		return NULL;
}
__device__ bool GScheduler::add(GAgent *ag, int idx){
	if(idx>=AGENT_NO_D)
		return false;
	this->allAgents[idx] = ag;
	return true;
}
__device__ bool GScheduler::swap(){
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx<AGENT_NO_D){
		GAgent *ag = this->allAgents[idx];
		this->allAgents[idx] = ag->getDummy();
	}
	return true;
}
void GScheduler::allocOnHost(){
}
void GScheduler::allocOnDevice(){
	this->assignments = NULL;
	cudaMalloc((void**)&allAgents, AGENT_NO*sizeof(GAgent*));
	cudaMalloc((void**)&time, sizeof(int));
	cudaMalloc((void**)&steps, sizeof(int));
	getLastCudaError("Scheduler::allocOnDevice:cudaMalloc");
}

//GModel
void GModel::allocOnDevice(){
	schedulerH = new GScheduler();
	schedulerH->allocOnDevice();
	cudaMalloc((void**)&scheduler, sizeof(GScheduler));
	cudaMemcpy(scheduler, schedulerH, sizeof(GScheduler), cudaMemcpyHostToDevice);
	getLastCudaError("GModel()");
}
void GModel::allocOnHost(){

}
__device__ GScheduler* GModel::getScheduler() const {
	return this->scheduler;
}
__device__ void GModel::addToScheduler(GAgent *ag, int idx){
	this->scheduler->add(ag, idx);
}

//GRandomGen
void GRandomGen::allocOnDevice(){
	size_t genRandStatesSize = GRID_SIZE*BLOCK_SIZE*sizeof(curandState);
	printf("curandStateSize: %d\n", genRandStatesSize);
	cudaMalloc((void**)&states,genRandStatesSize);
	getLastCudaError("GRandomGen::allocOnDevice");
}
__device__ float GRandomGen::nextFloat(){
	//int idx = threadIdx.x + blockIdx.x * blockDim.x;
	return 0;
	//return curand_uniform(&states[idx]);
}
__device__ float GRandomGen::nextFloat(curandState *state){
	return curand_uniform(state);
}
__device__ float GRandomGen::nextGaussian(){return 0;}

//namespace continuous2D Utility
__device__ int zcode(int x, int y){
	y &= 0x0000ffff;                  // x = ---- ---- ---- ---- fedc ba98 7654 3210
	x &= 0x0000ffff;                 // x = ---- ---- ---- ---- fedc ba98 7654 3210
	y = (y ^ (y << 8)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
	x = (x ^ (x << 8)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
	y = (y ^ (y << 4)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
	x = (x ^ (x << 4)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
	y = (y ^ (y << 2)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
	x = (x ^ (x << 2)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
	y = (y ^ (y << 1)) & 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
	x = (x ^ (x << 1)) & 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
	return x | (y << 1);
}
__global__ void c2dUtil::gen_hash_kernel(int *hash, Continuous2D *c2d)
{
	__syncthreads();
	GAgent *ag = c2d->obtainAgentPerThread();
	int idx = ag->getId();
	float2d_t myLoc = ag->getLoc();
	int xhash = (int)(myLoc.x/CLEN_X);
	int yhash = (int)(myLoc.y/CLEN_Y);
	hash[idx] = zcode(xhash, yhash);
	c2d->neighborIdx[idx] = idx;
	//printf("id: %d, hash: %d, neiIdx: %d\n", idx, hash[idx], c2d->neighborIdx[idx]);
}
__global__ void c2dUtil::gen_cellIdx_kernel(int *hash, Continuous2D *c2d)
{
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < AGENT_NO_D && idx > 0) {
		if (hash[idx] != hash[idx-1]) {
			c2d->cellIdxStart[hash[idx]] = idx;
			c2d->cellIdxEnd[hash[idx-1]] = idx-1;
		}
	}
	if (idx == 0) {
		c2d->cellIdxStart[hash[0]] = idx;
		c2d->cellIdxEnd[hash[AGENT_NO_D-1]] = AGENT_NO_D-1;
	}
}
void c2dUtil::sort_hash_kernel(int *hash, int *neighborIdx)
{
	thrust::device_ptr<int> id_ptr(neighborIdx);
	thrust::device_ptr<int> hash_ptr(hash);
	typedef thrust::device_vector<int>::iterator Iter;
	Iter key_begin(hash_ptr);
	Iter key_end(hash_ptr + AGENT_NO);
	Iter val_begin(id_ptr);
	thrust::sort_by_key(key_begin, key_end, val_begin);
	getLastCudaError("sort_hash_kernel");
}
void c2dUtil::genNeighbor(Continuous2D *world, Continuous2D *world_h)
{
	static int iterCount = 0;
	int bSize = BLOCK_SIZE;
	int gSize = GRID_SIZE;

	int *hash;
	cudaMalloc((void**)&hash, AGENT_NO*sizeof(int));
	cudaMemset(world_h->cellIdxStart, 0xff, CELL_NO*sizeof(int));
	cudaMemset(world_h->cellIdxEnd, 0xff, CELL_NO*sizeof(int));

	gen_hash_kernel<<<gSize, bSize>>>(hash, world);
	sort_hash_kernel(hash, world_h->neighborIdx);
	gen_cellIdx_kernel<<<gSize, bSize>>>(hash, world);

	//debug
	if (iterCount == SELECTION && FILE_GEN == 1){
		int *id_h, *hash_h, *cidx_h;
		id_h = new int[AGENT_NO];
		hash_h = new int[AGENT_NO];
		cidx_h = new int[CELL_NO];
		cudaMemcpy(id_h, world_h->neighborIdx, AGENT_NO * sizeof(int), cudaMemcpyDeviceToHost);
		getLastCudaError("genNeighbor:cudaMemcpy(id_h");
		cudaMemcpy(hash_h, hash, AGENT_NO * sizeof(int), cudaMemcpyDeviceToHost);
		getLastCudaError("genNeighbor:cudaMemcpy(hash_h");
		cudaMemcpy(cidx_h, world_h->cellIdxStart, CELL_NO * sizeof(int), cudaMemcpyDeviceToHost);
		getLastCudaError("genNeighbor:cudaMemcpy(cidx_h");
		std::fstream fout;
		char *outfname = new char[30];
		sprintf(outfname, "out_genNeighbor_%d_neighborIdx.txt", iterCount);
		fout.open(outfname, std::ios::out);
		for (int i = 0; i < AGENT_NO; i++){
			fout << id_h[i] << " " << hash_h[i] <<std::endl;
			fout.flush();
		}
		fout.close();
		sprintf(outfname, "out_genNeighbor_%d_cellIdx.txt", iterCount);
		fout.open(outfname, std::ios::out);
		for (int i = 0; i < CELL_NO; i++){
			fout << cidx_h[i] <<std::endl;
			fout.flush();
		}
		fout.close();
	}
	//~debug

	iterCount++;
	cudaFree(hash);
	getLastCudaError("genNeighbor:cudaFree:hash");
}
__global__ void c2dUtil::swapAgentsInWorld(Continuous2D *world){
	world->swap();
}

//namespace Scheduler Utility
struct SchdulerComp : public thrust::binary_function<GAgent, GAgent, bool> {
	__host__ __device__
		bool operator () (const GAgent *a, const GAgent *b) const
	{
		if (a->time != b->time)
			return a->time < b->time;
		else
			return a->rank < b->rank;
	}
};
void schUtil::sortWithKey(GModel *model){
	//it is the Scheduler in GModel that we are interested
	GModel *model_h = new GModel();
	GScheduler *sch_h = new GScheduler();
	cudaMemcpy(model_h, model, sizeof(GModel), cudaMemcpyDeviceToHost);
	cudaMemcpy(sch_h, model_h->scheduler, sizeof(GScheduler), cudaMemcpyDeviceToHost);
	thrust::device_ptr<GAgent*> ptr(sch_h->allAgents);
	thrust::sort(ptr, ptr+AGENT_NO, SchdulerComp());
}
__global__ void schUtil::scheduleRepeatingAllAgents(GModel *gm){
	GScheduler *sch = gm->getScheduler();
	GAgent *ag = sch->obtainAgentPerThread();
	if (ag != NULL){
		sch->scheduleRepeating(0,0,ag,1);
	}
}
__global__ void schUtil::swapAgentsInScheduler(GModel *model) {
	model->getScheduler()->swap();
}
__global__ void schUtil::step(GModel *gm){
	GScheduler *sch = gm->getScheduler();
	GAgent *ag = sch->obtainAgentPerThread();
#ifdef BOID_DEBUG
	if (ag == NULL)
		printf("schUtil::step: ag is NULL\n");
#endif
	//__syncthreads();
	ag->step(gm);
}

//namespace GRandomGen Utility
__global__ void rgenUtil::initStates(GRandomGen *rgen, int seed){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(seed, idx, 0, &rgen->states[idx]);
}

namespace gsim{

};
#endif