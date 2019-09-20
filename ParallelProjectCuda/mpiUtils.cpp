#include "mpiUtils.h"

/**********************************************
Create the Vector datatype
***********************************************/
void createMPIVectorDataType(MPI_Datatype * VectorMPIType) {
	Vector vector;
	MPI_Datatype type[4] = { MPI_DOUBLE, MPI_INT };
	int blocklen[2] = { 20, 1 };
	MPI_Aint disp[2];
	disp[0] = (char *)&vector.points - (char *)&vector;
	disp[1] = (char *)&vector.expected - (char *)&vector;
	MPI_Type_create_struct(2, blocklen, disp, type, VectorMPIType);
	MPI_Type_commit(VectorMPIType);
}

/**********************************************
Create the Model datatype
***********************************************/
void createMPIModelDataType(MPI_Datatype * ModelMPIType) {
	Model model;
	MPI_Datatype type[4] = { MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE };
	int blocklen[4] = { 20, 1,1,1 };
	MPI_Aint disp[4];
	disp[0] = (char *)&model.weights - (char *)&model;
	disp[1] = (char *)&model.bias - (char *)&model;
	disp[2] = (char *)&model.alpha - (char *)&model;
	disp[3] = (char *)&model.q - (char *)&model;
	MPI_Type_create_struct(4, blocklen, disp, type, ModelMPIType);
	MPI_Type_commit(ModelMPIType);
}

/**********************************************
brodcast elments from the master to slaves
***********************************************/
void broadcast(int myid, int numprocs, int *N, int *K, int *LIMIT, double *a0, double *aMax, double * QC, Model ** mod, Vector ** vec, MPI_Datatype VectorMPIType)
{
	MPI_Status status;
	if (myid == 0)
	{
		int sizeOfAlphas = (int)(*aMax / *a0);
		int alphasPerProc = sizeOfAlphas / numprocs;
		int diff = sizeOfAlphas - alphasPerProc*numprocs;
		int index = 0;
		double lastMax = 0;
		for (int i = 1; i < numprocs; i++, index++) {
			int size = 0;
			if (diff > 0)
			{
				size = alphasPerProc + 1;
				diff--;
			}
			else
				size = alphasPerProc;
			lastMax += *a0;
			double tempA0 = lastMax;
			double tempAMax = tempA0 + (size - 1) * *a0;
			lastMax = tempAMax;
			MPI_Send(&tempA0, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
			MPI_Send(&tempAMax, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
		}
		(*mod)->alpha = lastMax + *a0;
	}
	else
	{
		MPI_Recv(&(*mod)->alpha, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(&aMax, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);;
	}

	MPI_Bcast(a0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(N, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(K, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(LIMIT, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(QC, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(*vec, *N, VectorMPIType, 0, MPI_COMM_WORLD);

}


