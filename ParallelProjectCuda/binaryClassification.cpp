#include "binaryClassification.h"
/**********************************************
function that calculate which model is the best one
***********************************************/
void conclusion(int myid, int numprocs, Model * resultsArr, int K, double aMax, double QC) {
	double minimalAlphaValue = aMax + 1;
	Model  minimalAlphaModel;

	if (myid == 0)
	{
		for (int i = 0; i < numprocs; i++)
			if (resultsArr[i].q < QC && resultsArr[i].q != 1)
				if (resultsArr[i].alpha < minimalAlphaValue) {
					minimalAlphaValue = resultsArr[i].alpha;
					minimalAlphaModel = resultsArr[i];
				}
		if (minimalAlphaValue > aMax)
			printEroorOutput();
		else
			printOutput(&minimalAlphaModel, K);
	}
}
/**********************************************
the training of the model
***********************************************/
void train(Model * mod, int N, int K, int LIMIT, Vector *vec)
{
	int flag = 0;
	initWightes(mod, K);
	for (int itt = 0; itt < LIMIT && !flag; itt++) {
		flag = 1;
		for (int i = 0; i < N; i++)
		{
			int prediction = function(*mod, vec[i].points, K);
			if (vec[i].expected != prediction)
			{
				adjustWeights(mod, vec[i].expected, prediction, vec[i].points, K);
				flag = 0;
				break;
			}
		}
	}
}

/**********************************************
init the wieght before first run
***********************************************/
void initWightes(Model * mod, int k)
{
#pragma omp parallel for
	for (int i = 0; i < k; i++)
	{
		mod->weights[i] = 0;
	}
	mod->bias = 0;
}
/**********************************************
The function that caluclate the sign
***********************************************/
int function(Model mod, double * points, int k)
{
	double sum = mod.bias;
#pragma omp parallel for
	for (int i = 0; i < k; i++)
	{
		sum += mod.weights[i] * points[i];
	}
	return SIGN(sum);
}
/**********************************************
adjust the weights for the train
***********************************************/
void adjustWeights(Model * mod, int  expected, int prediction, double * points, int k)
{
#pragma omp parallel for
	for (int i = 0; i < k; i++)
	{
		mod->weights[i] += mod->alpha*(prediction)* points[i];
	}

	mod->bias += mod->alpha*(prediction);
}


/**********************************************
Another option to the adjustWeights 
Change in error calculate , see link- 
https://en.wikipedia.org/wiki/Perceptron
***********************************************/
//void adjustWeights(Model * mod, int  expected, int prediction, double * points, int k)
//{
//#pragma omp parallel for
//	for (int i = 0; i < k; i++)
//	{
//		mod->weights[i] += mod->alpha*(SIGN(expected - prediction))* points[i];
//	}
//
//	mod->bias += mod->alpha*(SIGN(expected - prediction));
//}
