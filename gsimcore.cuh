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
class Continuous2D;
class GScheduler;
class GModel;
class GRandom;

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

namespace util{
	__global__ void gen_hash_kernel(int *hash, Continuous2D *c2d);
	void sort_hash_kernel(int *hash, int *neighborIdx);
	__global__ void gen_cellIdx_kernel(int *hash, Continuous2D *c2d);
	void queryNeighbor(Continuous2D *c2d);
	void genNeighbor(Continuous2D *world, Continuous2D *world_h);
};

class GSteppable{
public:
	float time;
	int rank;
	virtual __device__ void step(GModel *model) = 0;
};
class GAgent : public GSteppable{	
protected:
	bool delMark;
	int id;
	GAgentData_t *data;
	GAgentData_t *dataCopy;
	__device__ int initId();
public:
	__device__ void allocOnDevice();
	__device__ int getId() const;
	__device__ bool getDelMark() const;
	__device__ GAgentData_t *getData();
	__device__ float2d_t getLoc() const;
	__device__ void remove();
	__device__ void swapDataAndCopy();
	__device__ virtual void step(GModel *model) = 0;
	__device__ virtual ~GAgent() {}
};
class Continuous2D{
public:
	float width;
	float height;
	float discretization;
private:
	Pool<GAgent> *allAgents;
	Pool<GAgent> *allAgentsHostCopy;
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
	__device__ GAgent* obtainAgentPerThread() const;
	__device__ GAgent* obtainAgentByInfoPtr(int ptr) const;
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
	friend __global__ void util::gen_hash_kernel(int *hash, Continuous2D *c2d);
	friend __global__ void util::gen_cellIdx_kernel(int *hash, Continuous2D *c2d);
	friend void util::genNeighbor(Continuous2D *world, Continuous2D *world_h);
	friend void util::queryNeighbor(Continuous2D *c2d);
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
	__device__ bool remove(GAgent *ag);
	void allocOnHost();
	void allocOnDevice();
};
class GModel{
public:
	GScheduler *scheduler, *schedulerH;
public:
	void allocOnHost();
	void allocOnDevice();
	__device__ GScheduler* getScheduler() const;
	__device__ void addToScheduler(GAgent *ag, int idx);
	__device__ void foo();
	__device__ int incAgentNo();
	__device__ int decAgentNo();
};
class GRandom {
	curandState *rState;
public:
	__device__ GRandom(int seed, int agId) {
		rState = new curandState();
		curand_init(seed, agId, 0, this->rState);
	}

	__device__ float uniform(){
		return curand_uniform(this->rState);
	}
};

//Continuous2D
void Continuous2D::allocOnDevice(){
	size_t sizeAgArray = MAX_AGENT_NO*sizeof(int);
	size_t sizeCellArray = CELL_NO*sizeof(int);

	this->allAgentsHostCopy = new Pool<GAgent>(AGENT_NO, MAX_AGENT_NO);
	cudaMalloc((void**)&this->allAgents, sizeof(Pool<GAgent>));
	cudaMemcpy(this->allAgents, this->allAgentsHostCopy, sizeof(Pool<GAgent>), cudaMemcpyHostToDevice);
	getLastCudaError("Continuous2D():cudaMalloc:allAgents");
	cudaMalloc((void**)&neighborIdx, sizeAgArray);
	getLastCudaError("Continuous2D():cudaMalloc:neighborIdx");
	cudaMalloc((void**)&cellIdxStart, sizeCellArray);
	getLastCudaError("Continuous2D():cudaMalloc:cellIdxStart");
	cudaMalloc((void**)&cellIdxEnd, sizeCellArray);
	getLastCudaError("Continuous2D():cudaMalloc:cellIdxEnd");
}
void Continuous2D::allocOnHost(){
	size_t sizeAgArray = MAX_AGENT_NO*sizeof(int);
	size_t sizeCellArray = CELL_NO*sizeof(int);
	neighborIdx = (int*)malloc(sizeAgArray);
	cellIdxStart = (int*)malloc(sizeCellArray);
}
__device__ const int* Continuous2D::getNeighborIdx() const{
	return this->neighborIdx;
}
__device__ bool Continuous2D::add(GAgent *ag, int idx) {
	int refIdx = this->allAgents->assign();
}
__device__ GAgent* Continuous2D::obtainAgentPerThread() const {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	GAgent *ag = NULL;
	if (idx < AGENT_NO_D) {
		int refIdx = this->allAgents->idxArray[idx];
		ag = &this->allAgents->dataArray[refIdx];
	}
	return ag;
}
__device__ GAgent* Continuous2D::obtainAgentByInfoPtr(int ptr) const {
	GAgent *ag = NULL;
	if (ptr < AGENT_NO_D && ptr >= 0){
		int agIdx = this->neighborIdx[ptr];
		if (agIdx < AGENT_NO_D && agIdx >=0) {
			int refIdx = this->allAgents->idxArray[agIdx];
			ag = &this->allAgents->dataArray[refIdx];
		}
		else 
			printf("Continuous2D::obtainAgentByInfoPtr:ptr:%d\n", ptr);
	} 
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
__device__ float Continuous2D::tds(const float2d_t loc1, const float2d_t loc2) const {
	float dx = loc1.x - loc2.x;
	float dxsq = dx*dx;
	float dy = loc1.y - loc2.y;
	float dysq = dy*dy;
	float x = dxsq+dysq;
	return sqrt(x);
}
__device__ dataUnion* Continuous2D::nextNeighborInit2(int agId, float2d_t agLoc, float range, iterInfo &info) const {
	const unsigned int tid = threadIdx.x;
	const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

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
	int lastFullWarp = AGENT_NO_D / warpSize;
	int totalFullWarpThreads = lastFullWarp * warpSize;
	int temp = 32;
	if (idx >= totalFullWarpThreads)
		temp = AGENT_NO_D - totalFullWarpThreads;

	for (int i=0; i<temp; i++){
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
		elem->loc.x = -1;
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

	while (elem->loc.x == -1 && info.cellCur.y <= info.cellDR.y) {
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

	if (elem->loc.x == -1) {
		elem = NULL;
	}
	return elem;
}
__device__ GAgentData_t *Continuous2D::nextAgentData(iterInfo &info) const {

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
	this->id = threadIdx.x + blockIdx.x * blockDim.x;
}
__device__ GAgentData_t *GAgent::getData(){
	return this->data;
}
__device__ float2d_t GAgent::getLoc() const{
	return this->data->loc;
}
__device__ int	GAgent::getId() const {
	return this->id;
}
__device__ bool GAgent::getDelMark() const {
	return this->delMark;
}
__device__ void GAgent::swapDataAndCopy() {
	GAgentData_t *temp = this->data;
	this->data = this->dataCopy;
	this->dataCopy = temp;
}
__device__ void GAgent::remove(){
	this->delMark = true;
	atomicInc(&AGENT_NO_DEC_D, AGENT_NO_D);
}

//GScheduler
__device__ bool GScheduler::ScheduleOnce(const float time, 	const int rank, GAgent *ag){
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < AGENT_NO_D){
		ag->time = time;
		ag->rank = rank;
		allAgents[idx] = ag;
	}
	return true;
}
__device__ bool GScheduler::scheduleRepeating(const float time, const int rank, GAgent *ag, const float interval){

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
	if(idx>=MAX_AGENT_NO_D)
		return false;
	this->allAgents[idx] = ag;
	return true;
}
void GScheduler::allocOnHost(){
}
void GScheduler::allocOnDevice(){
	this->assignments = NULL;
	cudaMalloc((void**)&this->allAgents, MAX_AGENT_NO*sizeof(GAgent*));
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
__device__ int GModel::incAgentNo(){
	return atomicInc((unsigned int *)&AGENT_NO_INC_D, MAX_AGENT_NO_D);
}

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
__global__ void util::gen_hash_kernel(int *hash, Continuous2D *c2d)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < AGENT_NO_D) {
		GAgent *ag = c2d->obtainAgentPerThread();
		float2d_t myLoc = ag->getLoc();
		int xhash = (int)(myLoc.x/CLEN_X);
		int yhash = (int)(myLoc.y/CLEN_Y);
		hash[idx] = zcode(xhash, yhash);
		c2d->neighborIdx[idx] = idx;
	}
	//printf("id: %d, hash: %d, neiIdx: %d\n", idx, hash[idx], c2d->neighborIdx[idx]);
}
__global__ void util::gen_cellIdx_kernel(int *hash, Continuous2D *c2d)
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
void util::sort_hash_kernel(int *hash, int *neighborIdx)
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
void util::genNeighbor(Continuous2D *world, Continuous2D *world_h)
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

__global__ void step(GModel *gm){
	GScheduler *sch = gm->getScheduler();
	GAgent *ag = sch->obtainAgentPerThread();
	if (ag != NULL) {
		ag->step(gm);
		ag->swapDataAndCopy();
	}
}

#endif