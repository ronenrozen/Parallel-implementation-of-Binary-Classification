#include "IOHandler.h"
#pragma warning(disable:4996)
/**********************************************
read from file the elements - N,K,LIMIT,a0,aMax,QC and the points
***********************************************/
void readFromFile(Vector * vec, int * N, int *K, int * LIMIT, double * a0, double  * aMax, double * QC)
{
	FILE *file;
	file = fopen(INPUT, "r");

	if (file == NULL)
	{
		perror("Error while opening the file.\n");
		exit(EXIT_FAILURE);
	}
	fscanf(file, "%d", N);
	fscanf(file, "%d", K);
	fscanf(file, "%lf", a0);
	fscanf(file, "%lf", aMax);
	fscanf(file, "%d", LIMIT);
	fscanf(file, "%lf", QC);

	for (int i = 0; i < *N; i++)
	{
		for (int j = 0; j < *K; j++)
		{
			fscanf(file, "%lf", &(vec[i].points[j]));
		}
		fscanf(file, "%d", &(vec[i].expected));
	}
	fclose(file);
}
/**********************************************
print output if quality of Classifier is reached 
***********************************************/
void printOutput(Model *mod, int k)
{
	FILE *file = fopen(OUTPUT, "w");
	if (file == NULL)
	{
		printf("Error opening file!\n");
		exit(1);
	}
	fprintf(file, "Alpha minimum-%lf , q-%lf \n", mod->alpha, mod->q);
	for (int i = 0; i < k; i++)
	{
		fprintf(file, "W(%d)-%lf\n", i + 1, mod->weights[i]);
	}
	fprintf(file, "W0 - %lf\n", mod->bias);
	printf("Finished successfully, please check output file - %s", OUTPUT);
}
/**********************************************
print output if quality of Classifier isnt reached
***********************************************/
void printEroorOutput()
{
	FILE *file = fopen(OUTPUT, "w");
	if (file == NULL)
	{
		printf("Error opening file!\n");
		exit(1);
	}
	fprintf(file, "Couldnt find alpah");
	printf("Finished with error, please check output file - %s", OUTPUT);
}