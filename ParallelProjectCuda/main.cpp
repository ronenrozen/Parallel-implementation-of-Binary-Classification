#pragma warning(disable:4996)
#include "Header.h"
#include <string.h>
#include <stdlib.h>
#include "mpiUtils.h"
#include "IOHandler.h"
#include "binaryClassification.h"


void main(int argc, char *argv[])
{
	int  namelen, numprocs, myid;
	int N, K, LIMIT;

	double a0, aMax, QC ,q=0;

	char processor_name[MPI_MAX_PROCESSOR_NAME];
	MPI_Init(&argc, &argv);
	
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Get_processor_name(processor_name, &namelen);
	/**********************************************
	Creating two MPI DataType for Model and Vector
	***********************************************/
	MPI_Datatype VectorMPIType;
	createMPIVectorDataType(&VectorMPIType);
	MPI_Datatype ModelMPIType;
	createMPIModelDataType(&ModelMPIType);

	Model *resultsArr = (Model*)malloc(sizeof(Model) * numprocs);
	Model *mod = (Model *)malloc(sizeof(Model));
	Vector *vectors = (Vector*)malloc(sizeof(Vector) * MAX_POINTS);
	Vector *dev_points = 0;
	Model *dev_mod = 0;
	/**********************************************
	Master read from file the points and then brodcasting
	them to other processes 
	also init the cuda kernel with the points
	***********************************************/
	if (myid == 0)
	{
		readFromFile(vectors, &N, &K, &LIMIT, &a0, &aMax, &QC);
	}
	
	broadcast(myid,numprocs,&N,&K,&LIMIT,&a0,&aMax,&QC,&mod,&vectors,VectorMPIType);
	initCuda(N, &dev_points, &dev_mod, vectors);
	/**********************************************
	Starting the algo, we train the Perceptron 
	and then with cuda valdiate the weight we got from the train phase
	***********************************************/
	do {
		train(mod,N,K,LIMIT,vectors);
		
		cudaError_t cudaStatus = calculateWithCuda(dev_points, N, &q, mod,dev_mod, K);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "calculateWithCuda failed!");
			MPI_Finalize();
			exit(1);
		}

		if (q < QC) {
			mod->q = q;
			break;
		}
		else {
			mod->q = 1;
		}
		mod->alpha += a0;
	} while (mod->alpha <= aMax);

	/**********************************************
	Gathering the Model from all the processes
	and calculate the result
	***********************************************/
	MPI_Gather(mod, 1, ModelMPIType, resultsArr, 1, ModelMPIType, 0, MPI_COMM_WORLD);
	conclusion(myid ,numprocs,resultsArr,K,aMax,QC);

	cudaFree(dev_points);
	cudaFree(dev_mod);
	MPI_Finalize();

}

