#ifndef __IOHandler_H
#define __IOHandler_H
#include "Header.h"
#include <stdlib.h>

void readFromFile(Vector * vec, int * N, int *K, int * LIMIT, double * a0, double  * aMax, double * QC);
void printOutput(Model *mod, int k);
void printEroorOutput();

#endif