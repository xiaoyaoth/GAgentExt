#ifndef GSIMAPP_BOID_H
#define GSIMAPP_BOID_H

#include "gsimcore.cuh"
#include "gsimlib_header.cuh"
#include "gsimapp_header.cuh"

#define STRIP 5

class BoidModel : public GModel{
public:
	Continuous2D *world, *worldH;
	GRandomGen *rgen, *rgenH;
	PreyBoidData_t *preyBoidDataArray;

	float cohesion;
	float avoidance;
	float randomness;
	float consistency;
	float momentum;
	float deadFlockerProbability;
	float neighborhood;
	float jump;
	BoidModel(){
		cohesion = 1.0;
		avoidance = 1.0;
		randomness = 1.0;
		consistency = 1.0;
		momentum = 1.0;
		deadFlockerProbability = 0.1;
		neighborhood = RANGE;
		jump = 0.7;
	}
	void allocOnDevice();
	void allocOnHost();
	__device__ Continuous2D* getWorld() const;
	__device__ void addToWorld(GAgent *ag, int idx);
};
class BaseBoid : public GAgent {
public:
	BoidModel *model;
	__device__ float getOrientation2D();
	__device__ void  setOrientation2D(float val);
	__device__ float2d_t momentum();
	__device__ virtual void step(GModel* model) = 0;
	__device__ virtual void putDataInSmem(dataUnion &dataElem) = 0;	
};
class FoodBoid : public BaseBoid{
	float scale;
	int amount;
	int respawnCount;
public:
	FoodBoidData_t *data;
	__device__ void reduce();
	__device__ void increase();
	__device__ void step(GModel* model);
	__device__ void putDataInSmem(dataUnion &dataElem){
	}
};
class PreyBoid : public BaseBoid{
public:
	__device__ PreyBoid(){
		PreyBoid(0,0, NULL);
	}
	__device__ PreyBoid(PreyBoid *twin, BoidModel *model){
		PreyBoidData_t *twinData = (PreyBoidData_t*)twin->getData();
		int id = twinData->id;
		PreyBoidData_t *myData = &model->preyBoidDataArray[AGENT_NO_D+id];
		myData->HUNGER_LIMIT = twinData->HUNGER_LIMIT;
		myData->STARVE_LIMIT = twinData->STARVE_LIMIT;
		myData->DEFAULT_SPEED = twinData->DEFAULT_SPEED;
		myData->id = id;
		myData->loc = twinData->loc;
		myData->lastd = twinData->lastd;
		myData->btype = twinData->btype;
		myData->bstate = twinData->bstate;
		myData->dead = twinData->dead;
		this->data = myData;
		this->model = model;
		this->time = twin->time;
		this->rank = twin->rank;
		this->dummy = twin;
	}
	__device__ PreyBoid(float x, float y, BoidModel *model){
		int id = this->initId();
		PreyBoidData_t *myData = &model->preyBoidDataArray[id];
		myData->HUNGER_LIMIT = CONSTANT::PREY_HUNGER_LIMIT;
		myData->STARVE_LIMIT = CONSTANT::PREY_STARVE_LIMIT;
		myData->DEFAULT_SPEED = 0.7;
		myData->id = id;
		myData->loc.x = x;
		myData->loc.y = y;
		myData->lastd.x = 0;
		myData->lastd.y = 0;
		myData->btype = PREY_BOID;
		myData->bstate = SEEKING_MATE;
		myData->dead = false;
		this->data = myData;
		this->model = model;
		this->time = 0;
		this->rank = 0;
		this->dummy = new PreyBoid(this, model);
	}
	__device__ bool hungry();
	__device__ void eat(FoodBoid *food);
	__device__ bool starved();
	__device__ bool readyToMate();
	__device__ void setRandomSpeed();
	__device__ float2d_t randomness(GRandomGen *gen);
	__device__ float2d_t consistency(const Continuous2D *world, iterInfo &info);
	__device__ float2d_t cohesion(const Continuous2D *world, iterInfo &info);
	__device__ float2d_t avoidance(const Continuous2D *world, iterInfo &info);
	__device__ void step(GModel *state);
	__device__ void step1(GModel *state);
	__device__ void putDataInSmem(dataUnion &dataElem);
};
class PredatorBoid : public BaseBoid{
public:
	int hungerCount;
	int accCount;
	int starveCount;

	float neighborRange;
	float surrounding;
	float jump;

	float MAX_SPEED;
	int ACC_DURATION;
	int ACC_COOLDOWN;
	int STARVE_LIMIT;
	int HUNGER_LIMIT;

	PreyBoid *lastTarget;

	__device__ float distanceToOther(BaseBoid* ag);
	__device__ void accelerate();
	__device__ void decelerate();
	__device__ bool hungry();
	__device__ void feast();
	__device__ bool starved();
	__device__ float2d_t randomness(GRandomGen *gen);
	__device__ float2d_t huntPrimitive();
	__device__ float2d_t huntByLockOnNearest();
	__device__ float2d_t huntByLockOnRandom();
	__device__ float2d_t stray();
	__device__ void step(GModel *model);
	__device__ void putDataInSmem(dataUnion &dataElem){
	}
};

//BoidModel
void BoidModel::allocOnDevice(){
	//init scheduler
	GModel::allocOnDevice();
	//init Continuous2D
	worldH = new Continuous2D(BOARDER_R_H, BOARDER_D_H, this->neighborhood/1.5);
	worldH->allocOnDevice();
	cudaMalloc((void**)&world, sizeof(Continuous2D));
	cudaMemcpy(world, worldH, sizeof(Continuous2D), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&preyBoidDataArray, 2*AGENT_NO*sizeof(PreyBoidData_t));
	//cudaMemcpyToSymbol(this->schedulerH->assignments, &this->worldH->neighborIdx,
		//sizeof(int), 0, cudaMemcpyDeviceToDevice);
	//init GRandomGen
	//rgenH = new GRandomGen();
	//rgenH->allocOnDevice();
	//cudaMalloc((void**)&rgen, sizeof(GRandomGen));
	//cudaMemcpy(rgen, rgenH, sizeof(GRandomGen), cudaMemcpyHostToDevice);
	//int gSize = GRID_SIZE;
	//rgenUtil::initStates<<<gSize, BLOCK_SIZE>>>(rgen, 1234);
	getLastCudaError("BoidModel()");
}
void BoidModel::allocOnHost(){
	GModel::allocOnHost();

	world = new Continuous2D(BOARDER_R_H, BOARDER_D_H, this->neighborhood/1.5);
	world->allocOnHost();
	this->scheduler = new GScheduler();
	this->scheduler->allocOnHost();
}
__device__ Continuous2D* BoidModel::getWorld() const {
	return this->world;
}
__device__ void BoidModel::addToWorld(GAgent *ag, int idx){
	this->world->add(ag, idx);
}

//BaseBoid
__device__ float BaseBoid::getOrientation2D(){
	BaseBoidData_t *myData = (BaseBoidData_t*)this->data;
	if (myData->lastd.x = 0 && myData->lastd.y == 0)
		return 0;
	return atan2(myData->lastd.y, myData->lastd.x);
}
__device__ void BaseBoid::setOrientation2D(float val){
	BaseBoidData_t *myData = (BaseBoidData_t*)this->data;
	myData->lastd.x = cos(val);
	myData->lastd.y = sin(val);
}
__device__ float2d_t BaseBoid::momentum(){
	BaseBoidData_t *myData = (BaseBoidData_t*)this->data;
	return myData->lastd;
}

//FoodBoid
__device__ void FoodBoid::reduce(){
	//if (this->amount <= 0)
	//	this->dead = true;
	//else {
	//	this->amount--;
	//	this->scale -= (this->scale/this->amount);
	//}
}
__device__ void FoodBoid::increase(){
	//if (this->amount <= 0)
	//	this->dead = true;
	//else if (this->amount < CONSTANT::FOOD_AMOUNT) {
	//	if (this->model->rgen->nextFloat() < 0.05){
	//		amount++;
	//		this->scale += (this->scale/this->amount);
	//	}
	//}
}
__device__ void FoodBoid::step(GModel *model){
	//if(this->dead) {
	//	this->model->getWorld()->remove(this);
	//	return;
	//}
	//if(this->model->rgen->nextFloat() > 0.3)
	//	this->increase();
}

//PreyBoid
__device__ bool PreyBoid::hungry(){
	PreyBoidData_t *myData = (PreyBoidData_t*)this->data;
	if (myData->hungerCount == myData->HUNGER_LIMIT) {
		myData->bstate = HUNGRY;
		myData->starveCount++;
		return true;
	} else {
		myData->hungerCount++;
		myData->bstate = NORMAL;
		return false;
	}
}
__device__ void PreyBoid::eat(FoodBoid *food){
	//food->reduce();
	//this->hungerCount = 0;
	//this->starveCount = 0;
	//this->mateCount++;
	//float hornyValue = 
	//	(float)(this->mateCount/CONSTANT::PREY_MATING_TIME);
	//if (this->model->rgen->nextFloat() < hornyValue)
	//	this->horny = true;

}
__device__ bool PreyBoid::starved(){
	PreyBoidData_t *myData = (PreyBoidData_t*)this->data;
	return myData->starveCount == myData->STARVE_LIMIT;
}
__device__ bool PreyBoid::readyToMate(){
	PreyBoidData_t *myData = (PreyBoidData_t*)this->data;
	return myData->horny;
}
__device__ void PreyBoid::setRandomSpeed(){
	PreyBoidData_t *myData = (PreyBoidData_t*)this->data;
	myData->DEFAULT_SPEED = CONSTANT::PREY_STD_SPEED + 
		this->model->rgen->nextGaussian() * 0.2;
}

__device__ float2d_t PreyBoid::randomness(GRandomGen *gen){
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	float x = randDebug[STRIP*idx];
	float y = randDebug[STRIP*idx+1];
	float l = sqrt(x*x + y*y);
	float2d_t res;
	res.x = 0.05*x/l;
	res.y = 0.05*y/l;
	return res;
}

#define TRIAL_NEXT_AGENT_DATA 0
__device__ float2d_t PreyBoid::consistency(const Continuous2D *world, iterInfo &info){
	float x=0, y=0, dx=0, dy=0;
	float sqrDist, ds;
	float2d_t m;
	world->resetNeighborInit(info);
	//world->nextNeighborInit2(this->data->id, this->data->loc, RANGE, info);
#if TRIAL_NEXT_AGENT_DATA == 0
	dataUnion otherData;
	dataUnion *elem = world->nextAgentDataIntoSharedMem(info);
#else
	PreyBoidData_t otherData;
	PreyBoidData_t *elem = (PreyBoidData_t*)world->nextAgentData(info);
#endif
	while(elem != NULL){
		otherData = *elem;
		ds = world->tds(info.myLoc, otherData.loc);
		if (ds < RANGE) {
			if (!otherData.dead){
				info.count++;
				m = otherData.lastd;
				x += m.x;
				y += m.y;
			}
		}
#if TRIAL_NEXT_AGENT_DATA == 0
		elem = world->nextAgentDataIntoSharedMem(info);
#else
		elem = (PreyBoidData_t*)world->nextAgentData(info);
#endif
	}

	if (info.count > 0){
		x /= info.count;
		y /= info.count;
	}
	int id = this->data->id;
	randDebug[STRIP*id+4] = info.count;
	float2d_t res;
	res.x = x;
	res.y = y;
	return res;
}
__device__ float2d_t PreyBoid::cohesion(const Continuous2D *world, iterInfo &info){
	float x=0, y=0;
	float ds;
	world->resetNeighborInit(info);
	//world->nextNeighborInit2(this->data->id, this->data->loc, RANGE, info);
#if TRIAL_NEXT_AGENT_DATA == 0
	dataUnion otherData;
	dataUnion *elem = world->nextAgentDataIntoSharedMem(info);
#else
	PreyBoidData_t otherData;
	PreyBoidData_t *elem = (PreyBoidData_t*)world->nextAgentData(info);
#endif
	while(elem != NULL){
		otherData = *elem;
		ds = world->tds(info.myLoc, otherData.loc);
		if (ds < RANGE) {
			if (!otherData.dead){
				info.count++;
				x += world->tdx(info.myLoc.x, otherData.loc.x);
				y += world->tdy(info.myLoc.y, otherData.loc.y);
			}
		}
#if TRIAL_NEXT_AGENT_DATA == 0
		elem = world->nextAgentDataIntoSharedMem(info);
#else
		elem = (PreyBoidData_t*)world->nextAgentData(info);
#endif
	}

	if (info.count > 0){
		x /= info.count;
		y /= info.count;
	}

	int id = this->data->id;
	randDebug[STRIP*id+3] = info.count;

	float2d_t res;
	res.x = -x/10;
	res.y = -y/10;
	return res;
}
__device__ float2d_t PreyBoid::avoidance(const Continuous2D *world, iterInfo &info){
	float x=0, y=0, dx=0, dy=0;
	float sqrDist, ds;
	world->nextNeighborInit2(this->data->id, this->data->loc, RANGE, info);
#if TRIAL_NEXT_AGENT_DATA == 0
	dataUnion otherData;
	dataUnion *elem = world->nextAgentDataIntoSharedMem(info);
#else
	PreyBoidData_t otherData;
	PreyBoidData_t *elem = (PreyBoidData_t*)world->nextAgentData(info);
#endif
	while(elem != NULL){
		otherData = *elem;
		ds = world->tds(info.myLoc, otherData.loc);
		if (ds < RANGE) {
			if (!otherData.dead){
				info.count++;
				dx = world->tdx(info.myLoc.x, otherData.loc.x);
				dy = world->tdy(info.myLoc.y, otherData.loc.y);
				sqrDist = dx*dx + dy*dy;
				x += dx/(sqrDist*sqrDist + 1);
				y += dy/(sqrDist*sqrDist + 1);
			}
		}
#if TRIAL_NEXT_AGENT_DATA == 0
		elem = world->nextAgentDataIntoSharedMem(info);
#else
		elem = (PreyBoidData_t*)world->nextAgentData(info);
#endif
	}

	if (info.count > 0){
		x /= info.count;
		y /= info.count;
	}

	int id = this->data->id;
	randDebug[STRIP*id+2] = info.count;

	float2d_t res;
	res.x = 400*x;
	res.y = 400*y;
	return res;
}

__device__ void PreyBoid::step(GModel *model){
	__syncthreads(); //这个barrier可以放到刚进step，但是不能放到getLoc()之后， 不然sync会出错
	const BoidModel *boidModel = (BoidModel*) model;
	const Continuous2D *world = boidModel->getWorld();
	iterInfo info;
	float2d_t avoid = this->avoidance(world, info);
	float2d_t cohes = this->cohesion(world, info);
	float2d_t consi = this->consistency(world, info);
	//float2d_t rdnes = this->randomness(model->rgen);
	float2d_t momen = this->momentum();
	float dx = 
		cohes.x * boidModel->cohesion +
		avoid.x * boidModel->avoidance +
		consi.x * boidModel->consistency +
		//rdnes.x * boidModel->randomness +
		momen.x * boidModel->momentum;
	float dy = 
		cohes.y * boidModel->cohesion +
		avoid.y * boidModel->avoidance +
		consi.y * boidModel->consistency +
		//rdnes.y * boidModel->randomness +
		momen.y * boidModel->momentum;

	float dist = sqrt(dx*dx + dy*dy);
	if (dist > 0){
		dx = dx / dist * boidModel->jump;
		dy = dy / dist * boidModel->jump;
	}
	 
	PreyBoidData_t *dummyDataPtr = (PreyBoidData_t*)dummy->getData();
	float2d_t myLoc = this->data->loc;
	dummyDataPtr->lastd.x = dx;
	dummyDataPtr->lastd.y = dy;
	dummyDataPtr->loc.x = world->stx(myLoc.x + dx);
	dummyDataPtr->loc.y = world->sty(myLoc.y + dy);

	randDebug[STRIP*this->data->id] = dummyDataPtr->loc.x;
	randDebug[STRIP*this->data->id+1] = dummyDataPtr->loc.y;
}
__device__ void PreyBoid::step1(GModel *model){
	const BoidModel *boidModel = (BoidModel*) model;
	const Continuous2D *world = boidModel->getWorld();

	iterInfo info; 
	float x=0, y=0, dx=0, dy=0;
	float sqrDist, ds;
	int ptrInSmem = 0;


	world->nextNeighborInit2(this->data->id, this->data->loc, RANGE, info);
	dataUnion *elem = NULL;
	while(elem != NULL){
		PreyBoidData_t otherData;
		otherData.dead = elem->dead;
		otherData.loc = elem->loc;
		ds = world->tds(info.myLoc, otherData.loc);
		if (ds < RANGE) {
			if (!otherData.dead){
				info.count++;
				dx = world->tdx(info.myLoc.x, otherData.loc.x);
				dy = world->tdy(info.myLoc.y, otherData.loc.y);
				sqrDist = dx*dx + dy*dy;
				x += dx/(sqrDist*sqrDist + 1);
				y += dy/(sqrDist*sqrDist + 1);
			}
		}
		elem = world->nextAgentDataIntoSharedMem(info);
	}
}
__device__ void PreyBoid::putDataInSmem(dataUnion &dataElem){
	PreyBoidData_t *myData = (PreyBoidData_t*)this->data;
	dataElem.bstate = myData->bstate;
	dataElem.btype = myData->btype;
	dataElem.dead = myData->dead;
	dataElem.id = myData->id;
	dataElem.lastd = myData->lastd;
	dataElem.loc = myData->loc;
}

//PredatorBoid
__device__ float PredatorBoid::distanceToOther(BaseBoid *b){return 0;}
__device__ void PredatorBoid::accelerate(){return;}
__device__ void PredatorBoid::decelerate(){return;}
__device__ bool PredatorBoid::hungry(){return true;}
__device__ void PredatorBoid::feast(){return;}
__device__ bool PredatorBoid::starved(){return true;}
__device__ float2d_t PredatorBoid::randomness(GRandomGen *gen){return float2d_t();}
__device__ float2d_t PredatorBoid::huntPrimitive(){return float2d_t();}
__device__ float2d_t PredatorBoid::huntByLockOnNearest(){return float2d_t();}
__device__ float2d_t PredatorBoid::huntByLockOnRandom(){return float2d_t();}
__device__ float2d_t PredatorBoid::stray(){return float2d_t();}
__device__ void PredatorBoid::step(GModel *model){}

#endif