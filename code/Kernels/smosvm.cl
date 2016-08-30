//__constant int maxIter = 10000;
//__constant float C = 1;
//__constant float tolerance = 0.001;
//__constant float eps = 0.001;

#define maxIter 10000
#define C 1.
#define tolerance 0.001
#define eps 0.001

int takeStep(int i1, int i2, __local float *w, int b, __local float *dense_points, __local int *trainIndex, int d, __local float *alph, 
		__global float *error_cache, __constant int *c_Correct_Classes, int trainN);


inline float kernel_func(int i1, int i2, __local float *dense_points, __local int *trainIndex, int d)
{
	//printf("begin kernel_func\n");
	if(true) return -1;
	float dot = 0.;
	for (int i = 0; i<d; i++)
		dot += dense_points[trainIndex[i1]*d+i] * dense_points[trainIndex[i2]*d+i];

	return dot;
}

inline float predict(int k, __local float *w, int b, __local float *dense_points, __local int *testIndex, int d)
{
	//("begin predict\n");
	//if(true) return -1;
		
	float s = 0.;

	//printf("\npredicting %i:\n", testIndex[k]);
	for (int i = 0; i<d; i++)
	{
		s += w[i] * dense_points[testIndex[k]*d+i];
		//printf("w %i:%f ",i,w[i]);
	}

	s -= b;
	return s;
}

//inline float learned_func(int k, __local float *w, int b, __local float *dense_points, __local int *trainIndex, int d)
/*inline float learned_func(int k, int b, int d)
{
	//printf("begin learned_func\n");
	//if(true) return -1;
	float s = 0.;

	for (int i = 0; i<d; i++)
		s += w[i] * dense_points[trainIndex[k]*d+i];

	s -= b;
	return s;
}*/

int takeStep(int i1, int i2, __local float *w, int b, __local float *dense_points, __local int *trainIndex, int d, __local float *alph, __global float *error_cache, __constant int *c_Correct_Classes, int trainN) 
{
	printf("called takeStep\n");
	
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);
	
	printf("x: %i y: %i z: %i \n",x,y,z);

	
	int y1, y2, s;
	float alph1, alph2; /* old_values of alpha_1, alpha_2 */
	float a1, a2;       /* new values of alpha_1, alpha_2 */
	float E1, E2, L, H, k11, k22, k12, eta, Lobj, Hobj;

	if (i1 == i2) return 0;

	alph1 = alph[i1];
	y1 = c_Correct_Classes[trainIndex[i1]];
	if (alph1 > 0 && alph1 < C)
		E1 = error_cache[i1];
	else
	{
		//E1 = learned_func(i1, w, b, dense_points, trainIndex, d) - y1;
		float slf = 0.0;
		for (int i = 0; i<d; i++)
			slf += w[i] * dense_points[trainIndex[i1]*d+i];
		//s -= b;
		E1 = slf - b - y1;
	}
		

	alph2 = alph[i2];
	y2 = c_Correct_Classes[trainIndex[i2]];
	if (alph2 > 0 && alph2 < C)
		E2 = error_cache[i2];
	else
	{
		//E2 = learned_func(i2, b, d) - y2;
		float slf = 0.0;
		for (int i = 0; i<d; i++)
			slf += w[i] * dense_points[trainIndex[i2]*d+i];
		//s -= b;
		E2 = slf - b - y2;
	}
//	E2 = learned_func(i2, w, b, dense_points, trainIndex, d) - y2;

	s = y1 * y2;

	if (y1 == y2) {
		float gamma = alph1 + alph2;
		if (gamma > C) {
			L = gamma - C;
			H = C;
		}
		else {
			L = 0;
			H = gamma;
		}
	}
	else {
		float gamma = alph1 - alph2;
		if (gamma > 0) {
			L = 0;
			H = C - gamma;
		}
		else {
			L = -gamma;
			H = C;
		}
	}

	if (L == H)
		return 0;
	
	/*
	k11 = 0.;
	for (int i = 0; i<d; i++)
		k11 += dense_points[trainIndex[i1]*d+i] * dense_points[trainIndex[i1]*d+i];
	
	k12 = 0.;
	for (int i = 0; i<d; i++)
		k12 += dense_points[trainIndex[i1]*d+i] * dense_points[trainIndex[i2]*d+i];

	k22 = 0.;
	for (int i = 0; i<d; i++)
		k22 += dense_points[trainIndex[i2]*d+i] * dense_points[trainIndex[i2]*d+i];
*/

	k11 = 	dense_points[trainIndex[i1]*d+0] * dense_points[trainIndex[i1]*d+0] +
			dense_points[trainIndex[i1]*d+1] * dense_points[trainIndex[i1]*d+1] +
			dense_points[trainIndex[i1]*d+2] * dense_points[trainIndex[i1]*d+2] +
			dense_points[trainIndex[i1]*d+3] * dense_points[trainIndex[i1]*d+3] +
			dense_points[trainIndex[i1]*d+4] * dense_points[trainIndex[i1]*d+4] +
			dense_points[trainIndex[i1]*d+5] * dense_points[trainIndex[i1]*d+5] +
			dense_points[trainIndex[i1]*d+6] * dense_points[trainIndex[i1]*d+6] +
			dense_points[trainIndex[i1]*d+7] * dense_points[trainIndex[i1]*d+7] +
			dense_points[trainIndex[i1]*d+8] * dense_points[trainIndex[i1]*d+8] +
			dense_points[trainIndex[i1]*d+9] * dense_points[trainIndex[i1]*d+9] +
			dense_points[trainIndex[i1]*d+10] * dense_points[trainIndex[i1]*d+10] +
			dense_points[trainIndex[i1]*d+11] * dense_points[trainIndex[i1]*d+11] +
			dense_points[trainIndex[i1]*d+12] * dense_points[trainIndex[i1]*d+12] +
			dense_points[trainIndex[i1]*d+13] * dense_points[trainIndex[i1]*d+13] +
			dense_points[trainIndex[i1]*d+14] * dense_points[trainIndex[i1]*d+14] +
			dense_points[trainIndex[i1]*d+15] * dense_points[trainIndex[i1]*d+15] +
			dense_points[trainIndex[i1]*d+16] * dense_points[trainIndex[i1]*d+16] +
			dense_points[trainIndex[i1]*d+17] * dense_points[trainIndex[i1]*d+17] +
			dense_points[trainIndex[i1]*d+18] * dense_points[trainIndex[i1]*d+18] ;

	
	k12 =   dense_points[trainIndex[i1]*d+0] * dense_points[trainIndex[i2]*d+0] +
			dense_points[trainIndex[i1]*d+1] * dense_points[trainIndex[i2]*d+1] +
			dense_points[trainIndex[i1]*d+2] * dense_points[trainIndex[i2]*d+2] +
			dense_points[trainIndex[i1]*d+3] * dense_points[trainIndex[i2]*d+3] +
			dense_points[trainIndex[i1]*d+4] * dense_points[trainIndex[i2]*d+4] +
			dense_points[trainIndex[i1]*d+5] * dense_points[trainIndex[i2]*d+5] +
			dense_points[trainIndex[i1]*d+6] * dense_points[trainIndex[i2]*d+6] +
			dense_points[trainIndex[i1]*d+7] * dense_points[trainIndex[i2]*d+7] +
			dense_points[trainIndex[i1]*d+8] * dense_points[trainIndex[i2]*d+8] +
			dense_points[trainIndex[i1]*d+9] * dense_points[trainIndex[i2]*d+9] +
			dense_points[trainIndex[i1]*d+10] * dense_points[trainIndex[i2]*d+10] +
			dense_points[trainIndex[i1]*d+11] * dense_points[trainIndex[i2]*d+11] +
			dense_points[trainIndex[i1]*d+12] * dense_points[trainIndex[i2]*d+12] +
			dense_points[trainIndex[i1]*d+13] * dense_points[trainIndex[i2]*d+13] +
			dense_points[trainIndex[i1]*d+14] * dense_points[trainIndex[i2]*d+14] +
			dense_points[trainIndex[i1]*d+15] * dense_points[trainIndex[i2]*d+15] +
			dense_points[trainIndex[i1]*d+16] * dense_points[trainIndex[i2]*d+16] +
			dense_points[trainIndex[i1]*d+17] * dense_points[trainIndex[i2]*d+17] +
			dense_points[trainIndex[i1]*d+18] * dense_points[trainIndex[i2]*d+18] ;
	
	k22 = 	dense_points[trainIndex[i2]*d+0] * dense_points[trainIndex[i2]*d+0] +
			dense_points[trainIndex[i2]*d+1] * dense_points[trainIndex[i2]*d+1] +
			dense_points[trainIndex[i2]*d+2] * dense_points[trainIndex[i2]*d+2] +
			dense_points[trainIndex[i2]*d+3] * dense_points[trainIndex[i2]*d+3] +
			dense_points[trainIndex[i2]*d+4] * dense_points[trainIndex[i2]*d+4] +
			dense_points[trainIndex[i2]*d+5] * dense_points[trainIndex[i2]*d+5] +
			dense_points[trainIndex[i2]*d+6] * dense_points[trainIndex[i2]*d+6] +
			dense_points[trainIndex[i2]*d+7] * dense_points[trainIndex[i2]*d+7] +
			dense_points[trainIndex[i2]*d+8] * dense_points[trainIndex[i2]*d+8] +
			dense_points[trainIndex[i2]*d+9] * dense_points[trainIndex[i2]*d+9] +
			dense_points[trainIndex[i2]*d+10] * dense_points[trainIndex[i2]*d+10] +
			dense_points[trainIndex[i2]*d+11] * dense_points[trainIndex[i2]*d+11] +
			dense_points[trainIndex[i2]*d+12] * dense_points[trainIndex[i2]*d+12] +
			dense_points[trainIndex[i2]*d+13] * dense_points[trainIndex[i2]*d+13] +
			dense_points[trainIndex[i2]*d+14] * dense_points[trainIndex[i2]*d+14] +
			dense_points[trainIndex[i2]*d+15] * dense_points[trainIndex[i2]*d+15] +
			dense_points[trainIndex[i2]*d+16] * dense_points[trainIndex[i2]*d+16] +
			dense_points[trainIndex[i2]*d+17] * dense_points[trainIndex[i2]*d+17] +
			dense_points[trainIndex[i2]*d+18] * dense_points[trainIndex[i2]*d+18] ;

//	k11 = kernel_func(i1, i1, dense_points, trainIndex, d);
//	k12 = kernel_func(i1, i2, dense_points, trainIndex, d);
//	k22 = kernel_func(i2, i2, dense_points, trainIndex, d);
	eta = 2 * k12 - k11 - k22;


	if (eta < 0) {
		a2 = alph2 + y2 * (E2 - E1) / eta;
		if (a2 < L)
			a2 = L;
		else if (a2 > H)
			a2 = H;
	}
	else {
		{
			float c1 = eta / 2;
			float c2 = y2 * (E1 - E2) - eta * alph2;
			Lobj = c1 * L * L + c2 * L;
			Hobj = c1 * H * H + c2 * H;
		}

		if (Lobj > Hobj + eps)
			a2 = L;
		else if (Lobj < Hobj - eps)
			a2 = H;
		else
			a2 = alph2;
	}

	if (fabs(a2 - alph2) < eps*(a2 + alph2 + eps))
		return 0;

	a1 = alph1 - s * (a2 - alph2);
	if (a1 < 0) {
		a2 += s * a1;
		a1 = 0;
	}
	else if (a1 > C) {
		float t = a1 - C;
		a2 += s * t;
		a1 = C;
	}
	if(true) return 0;

  float delta_b;
  {
	  float b1, b2, bnew;

	  if (a1 > 0 && a1 < C)
		  bnew = b + E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12;
	  else {
		  if (a2 > 0 && a2 < C)
			  bnew = b + E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22;
		  else {
			  b1 = b + E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12;
			  b2 = b + E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22;
			  bnew = (b1 + b2) / 2;
		  }
	  }

	  delta_b = bnew - b;
	  b = bnew;
  }

  if (1) {
	  float t1 = y1 * (a1 - alph1);
	  float t2 = y2 * (a2 - alph2);

	  for (int i = 0; i<d; i++)
		  w[i] += dense_points[trainIndex[i1]*d+i] * t1 + dense_points[trainIndex[i2]*d+i] * t2;
  }

	// ate aqui roda
  	if(true) return 0;
  {
	  float t1 = y1 * (a1 - alph1);
	  float t2 = y2 * (a2 - alph2);
	  
	  float dummy = 0;

	  for (int i = 0; i<trainN; i++)
	  {
		  //error_cache[i] = 0;
		  
		// if (0 < alph[i] && alph[i] < C)
		// {
		  	  
				  //float test = alph[0];			 
				  //error_cache[i] = 0.;
				  //float tmp1 = t1 * kernel_func(i1, i, dense_points, trainIndex, d);
				  //float tmp2 = t2 * kernel_func(i2, i, dense_points, trainIndex, d) - delta_b;
				  //error_cache[0] = 0;
				  //error_cache[i] = 0; //tmp1 + tmp2;
				  //error_cache[i] += t1 * kernel_func(i1, i, dense_points, trainIndex, d) + t2 * kernel_func(i2, i, dense_points, trainIndex, d) - delta_b;
			     // dummy += t1 * kernel_func(i1, i, dense_points, trainIndex, d) + t2 * kernel_func(i2, i, dense_points, trainIndex, d) - delta_b;
			
		//  }
		  
	  }
	  
	  if(true) return 0;
	  
	  //error_cache[i1] = 0.;
	  //error_cache[i2] = 0.;
  }

	

  //alph[i1] = a1;  /* Store a1 in the alpha array.*/
  //alph[i2] = a2;  /* Store a2 in the alpha array.*/

  return 1;
}


float rand2(unsigned int *seed) //uniform between 0-1
{
	//printf("called rand2\n");
  *seed = ((*seed) * 16807 ) % 2147483647;
  return  (float)(*seed) * 4.6566129e-10; //(4.6566129e-10 = 1/(2^31-1) = 1/2147483647)
}


int examineExample(int i1, __constant int *c_Correct_Classes, __local int *trainIndex, __local float *alph, __global float *error_cache, 
		__local float *w, int b, __local float *dense_points, int d, int trainN, unsigned int *seed)
{
	float y1, alph1, E1, r1;
	
	y1 = c_Correct_Classes[trainIndex[i1]];
	alph1 = alph[i1];

	if ((alph1 > 0) && (alph1 < C))
		E1 = error_cache[i1];
	else
	{
		//E1 = learned_func(i1, b, d) - y1;
		float s = 0.0;
		for (int i = 0; i<d; i++)
			s += w[i] * dense_points[trainIndex[i1]*d+i];
		//s -= b;
		E1 = s - b - y1;
	}
	//E1 = learned_func(i1, w, b, dense_points, trainIndex, d) - y1;

	r1 = y1 * E1;
	if ((r1 < -tolerance && alph1 < C)
		|| (r1 > tolerance && alph1 > 0))
	{
		/* Try i2 by three ways; if successful, then immediately return 1; */
	  {
		  int k, i2;
		  float tmax;

		  for (i2 = (-1), tmax = 0, k = 0; k < trainN; k++)
			  if (alph[k] > 0 && alph[k] < C) {
				  float E2, temp;

				  E2 = error_cache[k];
				  temp = fabs(E1 - E2);
				  if (temp > tmax)
				  {
					  tmax = temp;
					  i2 = k;
				  }
			  }

		  if (i2 >= 0) {
			  if (takeStep(i1, i2, w, b, dense_points, trainIndex, d, alph, error_cache, c_Correct_Classes, trainN))
				  return 1;
		  }
	  }
	  {
		  int k, k0;
		  int i2;

		  for (k0 = (int)(rand2(seed) * trainN), k = k0; k < trainN + k0; k++) {
			  i2 = k % trainN;
			  if (alph[i2] > 0 && alph[i2] < C) {
				  if (takeStep(i1, i2, w, b, dense_points, trainIndex, d, alph, error_cache, c_Correct_Classes, trainN))
					  return 1;
			  }
		  }
	  }
	  {
		  int k0, k, i2;

		  for (k0 = (int)(rand2(seed) * trainN), k = k0; k < trainN + k0; k++) {
			  i2 = k % trainN;
			  if (takeStep(i1, i2, w, b, dense_points, trainIndex, d, alph, error_cache, c_Correct_Classes, trainN))
				  return 1;
		  }
	  }

	}

	return 0;
}

void train(__local float *w, int b, __local float *dense_points, __local int *trainIndex, int d, __local float *alph, 
		__global float *error_cache, __constant int *c_Correct_Classes, int trainN, unsigned int *seed)
{
	int numChanged;
	int examineAll;
	int iter = 0;
	
	//printf("begin train\n");
	//if(true) return;
	
	// reset model
	b = 0;
	for (int wid=0; wid<d; ++wid)
	{
		w[wid] = 0;
	}
	for (int aid=0; aid<trainN; ++aid)
	{
		alph[aid] = 0;
	}

	if (1)
	{
		numChanged = 0;
		examineAll = 1;
		while (numChanged > 0 || examineAll)
		{
			numChanged = 0;
			if (examineAll)
			{
				for (int k = 0; k < trainN; k++)
					numChanged += examineExample(k, c_Correct_Classes, trainIndex, alph, error_cache, w, b, dense_points, d,  trainN, seed);
			}
			else
			{
				for (int k = 0; k < trainN; k++)
					if (alph[k] != 0 && alph[k] != C)
						numChanged += examineExample(k, c_Correct_Classes, trainIndex, alph, error_cache, w, b, dense_points, d,  trainN, seed);
			}
			if (examineAll == 1) examineAll = 0;
			else if (numChanged == 0) examineAll = 1;

			/* L_D */
			{
				int non_bound_support = 0;
				int bound_support = 0;
				for (int i = 0; i < trainN; i++)
					if (alph[i] > 0)
					{
						if (alph[i] < C)
							non_bound_support++;
						else
							bound_support++;
					}
				//printf("non_bound=%d\t", non_bound_support);
				//printf("bound_support=%d\t\n", bound_support);
			}

			iter += 1;
			if (iter >= maxIter)
			{
				//printf("warning: maximum number of iterations reached");
				break;
			}
		}
	}
}

float doLeaveOneOut(__local float *w, __local float *dense_points, __local int *trainIndex, __local int *testIndex, int d, __local float *alph, 
		__global float *error_cache, __constant int *c_Correct_Classes, int trainN, int numExemplos, unsigned int *seed, __local float *predicted)
{
	int n_correct = 0;
	int n_wrong = 0;
	int b = 0;
	
	//printf("begin doLeaveOneOut\n");
	
	for (int i = 0; i<numExemplos; i++)
	{
		int idx = 0;
		for (int k=0; k<numExemplos; k++)
		{
			if(k!=i)
			{
				trainIndex[idx] = k;
				++idx;
			}
			else
			{
				testIndex[0] = k;
			}
		}
	
		//printf("after setting indexes\n");
	
		train(w, b, dense_points, trainIndex, d, alph, error_cache, c_Correct_Classes, trainN, seed);
		
		//printf("after train\n");
		int testIdx = 0;
		predicted[testIndex[0]] = predict(testIdx, w, b, dense_points, testIndex, d);
		//printf("\nindex: %i verdadeiro: %i predito: %f",testIndex[0], c_Correct_Classes[testIndex[0]], predicted[testIndex[0]]);
		if (predicted[testIndex[0]]  > 0 == c_Correct_Classes[testIndex[0]] > 0)
		{
			n_correct++;
		}
		else
		{
			n_wrong++;
		}
		//printf(".");
	}
	//printf("\ncorrect: %i wrong: %i",n_correct, n_wrong);
	return (float)n_correct / (float)(n_correct+n_wrong);
}

float doLeavePairOut(__local float *w, __local float *dense_points, __local int *trainIndex, int d, __local float *alph, 
		__global float *error_cache, __constant int *c_Correct_Classes, int trainN, int numExemplos, unsigned int *seed, __local float *predicted)
{
	int n_correct = 0;
	int n_wrong = 0;
	int b = 0;
	
	__local int testIndex[2];
	  
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
