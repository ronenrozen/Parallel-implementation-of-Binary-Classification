#ifndef __HEADER_H
#define __HEADER_H 

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#define SIGN(x) ( x >= 0 ? 1 :-1 )
#define MAX_ELEMENTS 20 //max possible Coordinates
#define MAX_POINTS 500000 //max possible points
#define OUTPUT "C:/Output.txt"
#define INPUT "C:/data1.txt"
/**********************************************
Model struct
Includes- array of Weights, the bias, alpha, q
***********************************************/
typedef struct {
	double weights[MAX_ELEMENTS];
	double bias;
	double alpha;
	double q;
} Model;
/**********************************************
Vector struct
Includes- array of Points and the expected value of the train
***********************************************/
typedef struct {
	double points[MAX_ELEMENTS];
	int expected;
}Vector;

cudaError_t initCuda(int numOfTasks,  Vector **dev_tasks, Model **dev_mod, Vector *tasks);
cudaError_t calculateWithCuda(Vector *tasks, int numOfTasks, double *result,  Model * mod, Model * dev_mod, int K);
#endif // !