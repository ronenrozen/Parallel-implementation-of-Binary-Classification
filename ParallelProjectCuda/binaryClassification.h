#ifndef __BINARYCLASSIFICATION_H
#define __BINARYCLASSIFICATION_H
#include "Header.h"
#include "IOHandler.h"

void initWightes(Model * mod, int k);
int function(Model mod, double * points, int k);
void adjustWeights(Model * mod, int  actual, int prediction, double * mat, int k);
void train(Model * mod, int N, int K, int LIMIT, Vector *vec);
void conclusion(int myid, int numprocs, Model * resultsArr, int K, double aMax, double QC);

#endif
