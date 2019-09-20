#ifndef __MPI_H
#define __MPI_H
#include <mpi.h>
#include "Header.h"

void createMPIVectorDataType(MPI_Datatype * VectorMPIType);
void createMPIModelDataType(MPI_Datatype * ModelMPIType);
void broadcast(int myid, int numprocs, int *N, int *K, int *LIMIT, double *a0, double *aMax, double * QC, Model ** mod, Vector ** vec, MPI_Datatype VectorMPIType);

#endif