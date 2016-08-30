
void train_svm(int trainN, int NFEAT, __global float *x_space, __constant float *y, __global float* alpha, float* bias, 
				__global int *trainIndex, __global const float *kmatrix, int N)
{
	float lambda = 0.001;
	
	// initialize alpha
	float c1 = 0.5;
	float C = 1.;
	for (int k=0; k<trainN; ++k)
	{
		alpha[k] = C * c1;
	}
	
	// optimize: use the gradient descent learning rule to update all alpha values
	int EPOCS = 50;
	for (int e=0; e<EPOCS; ++e)
	{
		for (int k=0; k<trainN; ++k)
		{
			float sum = 0;
			for (int j=0; j<trainN; ++j)
			{
				sum += alpha[j]*y[trainIndex[j]]*y[trainIndex[k]] * kmatrix[trainIndex[j] + N*trainIndex[k]]; //dotproduct(x_space + trainIndex[k]*NFEAT, x_space + trainIndex[j]*NFEAT, NFEAT); //kernel_mat[k+j*l];
				//printf("alpha=%f yj=%f yk=%f dotkj=%f\n",alpha[j],y[j],y[k],dot);
			}
			alpha[k] += lambda*(1-sum);
			alpha[k] = alpha[k] > C ? C : alpha[k];
			alpha[k] = alpha[k] < 0 ? 0 : alpha[k];
		}
	}

	// compute bias value
	for (int k=0; k<trainN; ++k)
	{
		float g = 0;
		for (int j=0; j<trainN; ++j)
		{
			g += alpha[j] * y[trainIndex[j]] * kmatrix[trainIndex[j] + N * trainIndex[k]]; // *dotproduct(x_space + trainIndex[k]*NFEAT, x_space + trainIndex[j]*NFEAT, NFEAT); //kernel_mat[j+k*l] ;
		}
		*bias = *bias + y[trainIndex[k]] - ((g > 0) ? 1 : ((g < 0) ? -1 : 0));
		//printf("bias=%f g=%f y=%f\n", *bias, g, y[k]);
	}
	*bias = *bias / trainN;

}

void predict_svm(int trainN, int NFEAT, int xInd, __global float* x_space, __global float* alpha, __constant float* y, 
				float bias, float* pred, float* margin, __global int *trainIndex, __global const float* kmatrix, int N)
{
	float g = 0;
	
	for (int j=0; j<trainN; ++j)
	{
		//printf("predicting alpha= %f y=%f \n", alpha[j], y[j]);
		g += alpha[j] * y[trainIndex[j]] * kmatrix[trainIndex[j] + N * xInd];
	}
	//printf("predicting g=%f bias=%f\n", g, bias);
	
	g += bias;
	*pred = (g > 0) ? 1 : ((g < 0) ? -1 : 0);
	*margin = g;
	
	//printf("predicting g=%f\n", g);
}


void normalize_x(__global float *x_space, int N, int d)
{
	float eps = 0.00001;
	for (int m=0; m<d; ++m)
	{
		float minimum = x_space[m];
		float maximum = x_space[m];
		for (int k=1; k<N; ++k)
		{
			if( x_space[k*d+m] > maximum)
				maximum = x_space[k*d+m];
			if( x_space[k*d+m] < minimum)
				minimum = x_space[k*d+m];
		}

		float scale = 2/(maximum-minimum);
		float minimum_scaled = minimum * scale;

		//printf("m=%i scale=%f minimum_scaled=%f\n", m, scale, minimum_scaled);
		if ((maximum-minimum) < eps && fabs(maximum) > eps)
		{
			scale = 1/maximum;
			minimum_scaled = 0;
		}

		for (int k=0; k<N; ++k)
		{
			x_space[k*d+m] = x_space[k*d+m] * scale - minimum_scaled - 1.;
			//printf("x[%i][%i]=%f ", k, m, x_space[k*d+m] );
		}
		//printf("\nnext dimension\n");

	}

}


float doFold(__global float *dense_points, __global int *trainIndex, __global int *testIndex, int d, __global float *alpha, 
		 __constant float *c_Correct_Classes, int trainN, int numExemplos, __global const float* kmatrix, int fold)
{
	int n_correct = 0;
	
	//printf("begin doFold\n");
	
	int idx = 0;
	for (int k=0; k<numExemplos; k++)
	{
		if(k!=fold)
		{
			trainIndex[idx] = k;
			++idx;
		}
		else
		{
			testIndex[0] = k;
		}
	}

	float bias = 0;
	train_svm(trainN, d, dense_points, c_Correct_Classes, alpha, &bias, trainIndex, kmatrix, numExemplos);
	
	float pred = 0, margin = 0;
	predict_svm(trainN, d, fold, dense_points, alpha, c_Correct_Classes, bias, &pred, &margin, trainIndex, kmatrix, numExemplos);
	
	return margin;
}

/*
float doLeavePairOut(__global float *w, __global float *dense_points, __global int *trainIndex, int d, __global float *alph, 
		__global float *error_cache, __constant int *c_Correct_Classes, int trainN, int numExemplos, unsigned int *seed, __global float *predicted)
{
	int n_correct = 0;
	int n_wrong = 0;
	int b = 0;
	
	__global int testIndex[2];
	  
	for (int i = 0; i<numExemplos;)
	{
		int idx = 0;
		for (int k=0; k<numExemplos; )
		{
			if(k!=i)
			{
				trainIndex[idx] = k;
				++idx;
				trainIndex[idx] = k+1;
				++idx;
			}
			else
			{
				testIndex[0] = k;
				testIndex[1] = k+1;
			}

			k+=2;
		}

		train(w, b, dense_points, trainIndex, d, alph, error_cache, c_Correct_Classes, trainN, seed);
				
		for (int kt=0;kt<2;++kt)
		{
			int testIdx = kt;
			predicted[testIndex[kt]] = predict(testIdx, w, b, dense_points, testIndex, d);
			//printf("\nindex: %i verdadeiro: %i predito: %f",testIndex[kt], c_Correct_Classes[testIndex[kt]], predicted[testIndex[kt]]);
			if (predicted[testIndex[kt]]  > 0 == c_Correct_Classes[testIndex[kt]] > 0)
			{
				n_correct++;
			}
			else
			{
				n_wrong++;
			}

		}

		i+=2;
	}
	return (float)n_correct / (float)(n_correct+n_wrong);
}
*/

/*
int main(void) {
	
	printf("intervalo da rand: [0,%d]\n", RAND_MAX);
	srand( (unsigned)time(NULL) );
	int N = 40;
	int d = 10;
	float x_space[N*d];
	float y[N];
	for (int k=0; k<N; )
	{
		for (int m=0; m<d; ++m)
		{
			x_space[k*d+m] = (float)rand() / (float)RAND_MAX;
		}
		//y[k] = rand()>(RAND_MAX/2) ? 1 : -1;
		y[k] = 1;
		printf("y=%f ",y[k]);
		k+=2;

	}

	for (int k=1; k<N;)
	{
		for (int m=0; m<d; ++m)
		{
			x_space[k*d+m] = ((float)rand() / (float)RAND_MAX)  +	0.2;
		}
		//y[k] = rand() > (RAND_MAX/2) ? 1 : -1;
		y[k] = -1;
		printf("y=%f ",y[k]);
		k+=2;
	}

	normalize_x(x_space, N, d);

	float alpha[100];
	float bias = 0;
	printf("\nstart training\n");
	int l=N;
	int NFEAT = d;
	train_svm( l, NFEAT, x_space, y, alpha, &bias);
	printf("\ntraining finished\n");

	for (int k=0; k<80; ++k)
	{
		printf( "%f ", alpha[k] );
	}

	int count = 0;
	int countcorrect = 0;
	for (int k=(int)(0.8*N); k<N; ++k)
	{
		float pred = 0;
		float margin = 0;
		predict_svm(l, NFEAT, x_space+k*d, x_space, alpha, y, bias, &pred, &margin);
		//printf( "ponto %i: y verdadeiro: %f y predicted: %f\n",k, y[k], pred);

		if(y[k]==pred) countcorrect++;
		count++;
	}

	printf("\nAccuracy: %f\n", (float)countcorrect / (float)count);





	return EXIT_SUCCESS;
}
*/
