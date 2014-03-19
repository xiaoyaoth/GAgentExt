#ifndef APP_HEADER_H
#define APP_HEADER_H

#include "gsimlib_header.cuh"

__device__ __host__ enum BoidType {PREDATOR_BOID, FOOD_BOID, PREY_BOID};
__device__ __host__ enum BoidState{HUNGRY, NORMAL, FLEEING, STARVING, SEEKING_MATE};

typedef struct BaseBoidData : public GAgentData_t{
	float2d_t lastd;
	bool dead;
	BoidType btype;
	BoidState bstate;
	float DEFAULT_SPEED;
	__device__ float2d_t momentum(){
		return this->lastd;
	}
	__device__ virtual ~BaseBoidData(){}
} BaseBoidData_t;

typedef struct PreyBoidData : public BaseBoidData_t {
	float fleeBonus;
	int hungerCount;
	int starveCount;
	int mateCount;
	bool horny;

	float SENSE_FOOD_RANGE;
	int HUNGER_LIMIT;
	int STARVE_LIMIT;
	__device__ ~PreyBoidData(){}
} PreyBoidData_t;

typedef struct FoodBoidData : public BaseBoidData_t{
} FoodBoidData_t;

typedef struct dataUnionStruct : public GAgentData_t {
	float2d_t lastd;
	bool dead;
	BoidType btype;
	BoidState bstate;
	__device__ void addValue(GAgentData_t *data){
		loc = data->loc;
		BaseBoidData_t *boidData = (BaseBoidData_t*)data;
		BoidType bt = boidData->btype;
		if (bt == 2){
			PreyBoidData_t *pd = (PreyBoidData*)data;
			bstate = pd->bstate;
			btype = pd->btype;
			dead = pd->dead;
			lastd = pd->lastd;
		} else if(bt == FOOD_BOID)
			;
	}
}dataUnion;

namespace CONSTANT{
	// CONSTANTS FOR PREY BOIDS
	__device__ const float PREY_SENSE_RANGE_FOOD = 50;
	__device__ static const int PREY_HUNGER_LIMIT = 500;
	__device__ static const int PREY_STARVE_LIMIT = 1000;
	__device__ static const int PREY_MATING_TIME = 3;
	__device__ static const float PREY_STD_SPEED = 0.7;

	// CONSTANTS FOR PREDATORS
	__device__ static const int PRED_HUNGER_LIMIT = 600;
	__device__ static const float PRED_DEFAULT_SPEED = 0.4;
	__device__ static const float PRED_MAX_SPEED = 1.8;
	__device__ static const int PRED_ACC_DURATION = 150;
	__device__ static const int PRED_ACC_COOLDOWN = 250;
	__device__ static const int PRED_STARVE_LIMIT = 1500;
	__device__ static const float PRED_NEIGHBORHOOD_SENSE = 50;
	__device__ static const float PRED_SURROUNDING_SENSE = 40;
	__device__ static const float PRED_STD_JUMP = 0.5;
	__device__ static const float PRED_MATING_TIME = 3;

	// CONSTANTS FOR FOOD
	__device__ static const int FOOD_AMOUNT = 100;

	// CONSTANTS OF MODEL
	__device__ static const int NUM_PREY = 100; //default = 75
	__device__ static const int NUM_PRED = 5; //default = 10

	// Set to false if the GUI is not used
	// (mainly for ea runs)
	__device__ static const bool IS_VISUALIZED = true;

	// CONSTANTS OF EA
	__device__ static const int NUM_REPETITIONS = 50;
	__device__ static const int SIM_STEPS = 5000;
	__device__ static const int GENERATIONS = 50;
}

#endif