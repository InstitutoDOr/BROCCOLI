/*
 * BROCCOLI: Software for fast fMRI analysis on many-core CPUs and GPUs
 * Copyright (C) <2013>  Anders Eklund, andek034@gmail.com
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "broccoli_lib.h"
#include <stdio.h>
#include <stdlib.h>
#include "nifti1_io.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <math.h>

#include "HelpFunctions.cpp"

#define ADD_FILENAME true
#define DONT_ADD_FILENAME true

#define CHECK_EXISTING_FILE true
#define DONT_CHECK_EXISTING_FILE false


int main(int argc, char **argv)
{
    //-----------------------
    // Input
    
    float           *h_Data, *h_Mask; 
  
    float           *h_X_GLM, *h_xtxxt_GLM, *h_X_GLM_Confounds, *h_Contrasts, *h_ctxtxc_GLM, *h_Highres_Regressor, *h_LowpassFiltered_Regressor, *h_Motion_Parameters;
                  
    int             DATA_W, DATA_H, DATA_D, DATA_T;               
    float           VOXEL_SIZE_X, VOXEL_SIZE_Y, VOXEL_SIZE_Z, TR;

	int             NUMBER_OF_GLM_REGRESSORS, NUMBER_OF_TOTAL_GLM_REGRESSORS;
    int             NUMBER_OF_DETRENDING_REGRESSORS = 4;
    int             NUMBER_OF_MOTION_REGRESSORS = 6;	

	int				NUMBER_OF_EVENTS;
	int				HIGHRES_FACTOR = 100;

    //-----------------------
    // Output
    
    float           *h_Beta_Volumes, *h_Contrast_Volumes, *h_Residuals, *h_Residual_Variances, *h_Statistical_Maps;        
    float           *h_AR1_Estimates, *h_AR2_Estimates, *h_AR3_Estimates, *h_AR4_Estimates;
	float           *h_Design_Matrix, *h_Design_Matrix2;

	//--------------

    void*           allMemoryPointers[500];
	for (int i = 0; i < 500; i++)
	{
		allMemoryPointers[i] = NULL;
	}
    
	nifti_image*	allNiftiImages[500];
	for (int i = 0; i < 500; i++)
	{
		allNiftiImages[i] = NULL;
	}

    int             numberOfMemoryPointers = 0;
	int				numberOfNiftiImages = 0;

	size_t			allocatedHostMemory = 0;

	//--------------
    
    // Default parameters
        
    int             OPENCL_PLATFORM = 0;
    int             OPENCL_DEVICE = 0;
    bool            DEBUG = false;
    bool            PRINT = true;
	bool			VERBOS = false;
   	bool			CHANGE_OUTPUT_NAME = false;    
                   
	float           AR_SMOOTHING_AMOUNT = 6.0f;
	int				NUMBER_OF_CONTRASTS = 1; 
	int	 NUMBER_OF_STATISTICAL_MAPS = 1;
    float           CLUSTER_DEFINING_THRESHOLD = 2.5f;
	int				STATISTICAL_TEST = 0;
	int				INFERENCE_MODE = 1;
	bool			MASK = false;
	int             USE_TEMPORAL_DERIVATIVES = 0;

	bool REGRESS_GLOBALMEAN = false;
	bool FOUND_DESIGN = false;
	bool FOUND_CONTRASTS = false;
	bool ANALYZE_GROUP_MEAN = false;
	bool ANALYZE_TTEST = false;
	bool ANALYZE_FTEST = false;
	bool BETAS_ONLY = false;
	bool CONTRASTS_ONLY = false;
	bool BETAS_AND_CONTRASTS_ONLY = false;
	bool RAW_DESIGNMATRIX = false;
	bool MODEL_AUTO_CORRELATION = true;
	bool ADD_DETRENDING_REGRESSORS = false;
	bool REGRESS_MOTION = false;
	bool REGRESS_GLOBAL_MEAN = false;
	bool RAW_REGRESSORS = false;

	bool FIRST_LEVEL = false;
	bool SECOND_LEVEL = false;

	bool WRITE_DESIGNMATRIX = false;
	bool WRITE_ORIGINAL_DESIGNMATRIX = false;
	bool WRITE_RESIDUALS = false;
	bool WRITE_RESIDUAL_VARIANCES = false;
	bool WRITE_AR_ESTIMATES = false;

	const char*		MASK_NAME;
	const char*		DESIGN_FILE;        
	const char*		CONTRASTS_FILE;
	const char*		MOTION_PARAMETERS_FILE;
	const char*		outputFilename;
       
    //---------------------    
    
    /* Input arguments */
    FILE *fp = NULL; 
    
    // No inputs, so print help text
    if (argc == 1)
    {   
		printf("\nThe function applies the GLM for single subject analysis and group analysis.\n\n");     
        printf("Usage first level, design.txt contains all regressors:\n\n");
        printf("GLM volumes.nii -design design.txt -contrasts design.txt -firstlevel [options]\n\n");
        printf("Usage first level, regressors.txt contains a file name to each regressor:\n\n");
        printf("GLM volumes.nii -designfiles regressors.txt -contrasts design.txt -firstlevel [options]\n\n");
        printf("Usage second level:\n\n");
        printf("GLM volumes.nii -design design.txt -contrasts design.txt -secondlevel [options]\n\n");
//        printf("Regression only:\n\n");
//        printf("GLM volumes.nii -design design.txt -regressonly [options]\n\n");
        printf("Options:\n\n");
        printf(" -platform                  The OpenCL platform to use (default 0) \n");
        printf(" -device                    The OpenCL device to use for the specificed platform (default 0) \n");
        printf(" -design                    The design matrix to use, one row per volume, one column per regressor \n");
        printf(" -designfiles               File containing regressor files to use to create design matrix,\n");
        printf("                            intended for first level analysis.\n");
        printf(" -contrasts                 The contrast vector(s) to apply to the estimated beta values \n");
        printf(" -firstlevel                Analyze data from a single subject \n");
        printf(" -secondlevel               Analyze data from several subjects \n");
        printf(" -teststatistics            Test statistics to use, 0 = GLM t-test, 1 = GLM F-test  (default 0) \n");
        printf(" -betasonly                 Only save the beta values, contrast file not needed (default no) \n");
        printf(" -contrastsonly             Only save the contrast values (default no) \n");
        printf(" -betasandcontrastsonly     Only save the beta values and the contrast values (default no) \n");
        printf(" \nOptions for single subject analysis \n\n");
        printf(" -temporalderivatives       Use temporal derivatives for the activity regressors (default no) \n");
        printf(" -regressmotion             Provide file with motion regressors to use in design matrix (default no) \n");
        printf(" -regressglobalmean         Include global mean in design matrix (default no) \n");
        printf(" \n\n");
        printf(" -mask                      A mask that defines which voxels to run the GLM for (default none) \n");
		printf(" -saveresiduals             Save the residuals (default no) \n");
		printf(" -saveresidualvariance      Save residual variance (default no) \n");
        printf(" -savearparameters          Save the estimated AR coefficients (first level only, default no) \n");
		printf(" -saveoriginaldesignmatrix  Save the original design matrix used (default no) \n");
        printf(" -savedesignmatrix          Save the total design matrix used (default no) \n");        
		printf(" -output                    Set output filename (default volumes_) \n");
        printf(" -quiet                     Don't print anything to the terminal (default false) \n");
        printf(" -verbose                   Print extra stuff (default false) \n");
        printf("\n\n");
        
        return EXIT_SUCCESS;
    }
    // Try to open file
    else if (argc > 1)
    {        
        fp = fopen(argv[1],"r");
        if (fp == NULL)
        {
            printf("Could not open file %s !\n",argv[1]);
            return EXIT_FAILURE;
        }
        fclose(fp);             
    }
    
    // Loop over additional inputs

    int i = 2;
    while (i < argc)
    {
        char *input = argv[i];
        char *p;
        if (strcmp(input,"-platform") == 0)
        {
			if ( (i+1) >= argc  )
			{
			    printf("Unable to read value after -platform !\n");
                return EXIT_FAILURE;
			}

            OPENCL_PLATFORM = (int)strtol(argv[i+1], &p, 10);

			if (!isspace(*p) && *p != 0)
		    {
		        printf("OpenCL platform must be an integer! You provided %s \n",argv[i+1]);
				return EXIT_FAILURE;
		    }
            else if (OPENCL_PLATFORM < 0)
            {
                printf("OpenCL platform must be >= 0!\n");
                return EXIT_FAILURE;
            }
            i += 2;
        }
        else if (strcmp(input,"-device") == 0)
        {
			if ( (i+1) >= argc  )
			{
			    printf("Unable to read value after -device !\n");
                return EXIT_FAILURE;
			}

            OPENCL_DEVICE = (int)strtol(argv[i+1], &p, 10);

			if (!isspace(*p) && *p != 0)
		    {
		        printf("OpenCL device must be an integer! You provided %s \n",argv[i+1]);
				return EXIT_FAILURE;
		    }
            else if (OPENCL_DEVICE < 0)
            {
                printf("OpenCL device must be >= 0!\n");
                return EXIT_FAILURE;
            }
            i += 2;
        }


        else if (strcmp(input,"-design") == 0)
        {
			if ( (i+1) >= argc  )
			{
			    printf("Unable to read value after -design !\n");
                return EXIT_FAILURE;
			}

            DESIGN_FILE = argv[i+1];
			RAW_DESIGNMATRIX = true;
			FOUND_DESIGN = true;
            i += 2;
        }
        else if (strcmp(input,"-designfiles") == 0)
        {
			if ( (i+1) >= argc  )
			{
			    printf("Unable to read value after -design !\n");
                return EXIT_FAILURE;
			}

            DESIGN_FILE = argv[i+1];
			RAW_DESIGNMATRIX = false;
			FOUND_DESIGN = true;
            i += 2;
        }
        else if (strcmp(input,"-contrasts") == 0)
        {
			if ( (i+1) >= argc  )
			{
			    printf("Unable to read value after -contrasts !\n");
                return EXIT_FAILURE;
			}

            CONTRASTS_FILE = argv[i+1];
			FOUND_CONTRASTS = true;
            i += 2;
        }
        else if (strcmp(input,"-firstlevel") == 0)
        {
			FIRST_LEVEL = true;
            i += 1;
        }
        else if (strcmp(input,"-secondlevel") == 0)
        {
			SECOND_LEVEL = true;
            i += 1;
        }
        else if (strcmp(input,"-teststatistics") == 0)
        {
			if ( (i+1) >= argc  )
			{
			    printf("Unable to read value after -teststatistics !\n");
                return EXIT_FAILURE;
			}

            STATISTICAL_TEST = (int)strtol(argv[i+1], &p, 10);

			if (!isspace(*p) && *p != 0)
		    {
		        printf("Test statistics must be an integer! You provided %s \n",argv[i+1]);
				return EXIT_FAILURE;
		    }
            if ((STATISTICAL_TEST != 0) && (STATISTICAL_TEST != 1))
            {
                printf("Test statistics must be 0 or 1!\n");
                return EXIT_FAILURE;
            }
            i += 2;
        }
        else if (strcmp(input,"-betasonly") == 0)
        {
			BETAS_ONLY = true;
            i += 1;
        }
        else if (strcmp(input,"-contrastsonly") == 0)
        {
			CONTRASTS_ONLY = true;
            i += 1;
        }
        else if (strcmp(input,"-betasandcontrastsonly") == 0)
        {
			BETAS_AND_CONTRASTS_ONLY = true;
            i += 1;
        }


        else if (strcmp(input,"-temporalderivatives") == 0)
        {
            USE_TEMPORAL_DERIVATIVES = 1;
            i += 1;
        }
        else if (strcmp(input,"-autocorr") == 0)
        {
			MODEL_AUTO_CORRELATION = true;
            i += 1;
        }
        else if (strcmp(input,"-adddetrendingregressors") == 0)
        {
			ADD_DETRENDING_REGRESSORS = true;
            i += 1;
        }
        else if (strcmp(input,"-regressmotion") == 0)
        {
			if ( (i+1) >= argc  )
			{
			    printf("Unable to read value after -regressmotion!\n");
                return EXIT_FAILURE;
			}

            MOTION_PARAMETERS_FILE = argv[i+1];
            REGRESS_MOTION = true;
            i += 2;
        }
        else if (strcmp(input,"-regressglobalmean") == 0)
        {
            REGRESS_GLOBALMEAN = 1;
            i += 1;
        }

		else if (strcmp(input,"-mask") == 0)
        {
			if ( (i+1) >= argc  )
			{
			    printf("Unable to read name after -mask !\n");
                return EXIT_FAILURE;
			}

			MASK = true;
            MASK_NAME = argv[i+1];
            i += 2;
        }
        else if (strcmp(input,"-saveresiduals") == 0)
        {
            WRITE_RESIDUALS = true;
            i += 1;
        }
        else if (strcmp(input,"-saveresidualvariance") == 0)
        {
            WRITE_RESIDUAL_VARIANCES = true;
            i += 1;
        }
        else if (strcmp(input,"-savearparameters") == 0)
        {
            WRITE_AR_ESTIMATES = true;
            i += 1;
        }
        else if (strcmp(input,"-savedesignmatrix") == 0)
        {
            WRITE_DESIGNMATRIX = true;
            i += 1;
        }
        else if (strcmp(input,"-saveoriginaldesignmatrix") == 0)
        {
            WRITE_ORIGINAL_DESIGNMATRIX = true;
            i += 1;
        }
        else if (strcmp(input,"-debug") == 0)
        {
            DEBUG = true;
            i += 1;
        }
        else if (strcmp(input,"-quiet") == 0)
        {
            PRINT = false;
            i += 1;
        }
        else if (strcmp(input,"-verbose") == 0)
        {
            VERBOS = true;
            i += 1;
        }
        else if (strcmp(input,"-output") == 0)
        {
			if ( (i+1) >= argc  )
			{
			    printf("Unable to read name after -output !\n");
                return EXIT_FAILURE;
			}

			CHANGE_OUTPUT_NAME = true;
            outputFilename = argv[i+1];
            i += 2;
        }
        else
        {
            printf("Unrecognized option! %s \n",argv[i]);
            return EXIT_FAILURE;
        }                
    }

	if (!FOUND_DESIGN)
	{
    	printf("No design file detected, aborting! \n");
        return EXIT_FAILURE;
	}

	if (!FOUND_CONTRASTS && !BETAS_ONLY)
	{
    	printf("No contrasts file detected, aborting! \n");
        return EXIT_FAILURE;
	}

	if (!FIRST_LEVEL && !SECOND_LEVEL)
	{
    	printf("Have to define if GLM is for single subject or several subjects, aborting! \n");
        return EXIT_FAILURE;
	}


	//------------------------------------------
	// Check for invalid combinations

	if (WRITE_AR_ESTIMATES && (SECOND_LEVEL || BETAS_ONLY || CONTRASTS_ONLY || BETAS_AND_CONTRASTS_ONLY))
	{
    	printf("No AR parameters will be estimated for the selected analysis, cannot save them, aborting! \n");
        return EXIT_FAILURE;
	}
	
	if (RAW_DESIGNMATRIX && USE_TEMPORAL_DERIVATIVES)
	{
    	printf("Cannot use temporal derivatives for raw design matrix, aborting! \n");
        return EXIT_FAILURE;
	}

	if (SECOND_LEVEL && USE_TEMPORAL_DERIVATIVES)
	{
    	printf("Cannot use temporal derivatives for second level analysis, aborting! \n");
        return EXIT_FAILURE;
	}

	if (SECOND_LEVEL && REGRESS_GLOBALMEAN)
	{
    	printf("Cannot regress global mean for second level analysis, aborting! \n");
        return EXIT_FAILURE;
	}

	if (SECOND_LEVEL && REGRESS_MOTION)
	{
    	printf("Cannot regress motion for second level analysis, aborting! \n");
        return EXIT_FAILURE;
	}

	// Check if BROCCOLI_DIR variable is set
	if (getenv("BROCCOLI_DIR") == NULL)
	{
        printf("The environment variable BROCCOLI_DIR is not set!\n");
        return EXIT_FAILURE;
	}

	if (STATISTICAL_TEST == 0)
	{
		ANALYZE_TTEST = true;
	}
	else if (STATISTICAL_TEST == 1)
	{
		ANALYZE_FTEST = true;
	}

	//------------------------------------------
    // Read number of regressors from design matrix file
  	//------------------------------------------

	std::ifstream design;
    std::string tempString;
    int tempNumber;
    std::string NR("NumRegressors");

    design.open(DESIGN_FILE);    
    if (!design.good())
    {
        design.close();
        printf("Unable to open design file %s. Aborting! \n",DESIGN_FILE);
        return EXIT_FAILURE;
    }
    
    // Get number of regressors
    design >> tempString; // NumRegressors as string
    if (tempString.compare(NR) != 0)
    {
        design.close();
        printf("First element of the design file %s should be the string 'NumRegressors', but it is %s. Aborting! \n",DESIGN_FILE,tempString.c_str());
        return EXIT_FAILURE;
    }
    design >> NUMBER_OF_GLM_REGRESSORS;
    
    if (NUMBER_OF_GLM_REGRESSORS <= 0)
    {
        design.close();
        printf("Number of regressors must be > 0 ! You provided %i regressors in the design file %s. Aborting! \n",NUMBER_OF_GLM_REGRESSORS,DESIGN_FILE);
        return EXIT_FAILURE;
    }
    design.close();

	//------------------------------------------  
    // Read number of contrasts from contrasts file
	//------------------------------------------

   	std::ifstream contrasts;    

	if (!BETAS_ONLY)
	{
	    contrasts.open(CONTRASTS_FILE);  
	    if (!contrasts.good())
	    {
	        contrasts.close();
	        printf("Unable to open contrasts file %s. Aborting! \n",CONTRASTS_FILE);
	        return EXIT_FAILURE;
	    }
    
	    contrasts >> tempString; // NumRegressors as string
	    if (tempString.compare(NR) != 0)
	    {
	        contrasts.close();
	        printf("First element of the contrasts file should be the string 'NumRegressors', but it is %s. Aborting! \n",tempString.c_str());
	        return EXIT_FAILURE;
	    }
	    contrasts >> tempNumber;
    
	    // Check for consistency
	    if ( tempNumber != NUMBER_OF_GLM_REGRESSORS )
   		{
	        contrasts.close();
	        printf("Design file says that number of regressors is %i, while contrast file says there are %i regressors. Aborting! \n",NUMBER_OF_GLM_REGRESSORS,tempNumber);
	        return EXIT_FAILURE;
	    }
    
	    contrasts >> tempString; // NumContrasts as string
	    std::string NC("NumContrasts");
	    if (tempString.compare(NC) != 0)
	    {
	        contrasts.close();
	        printf("Third element of the contrasts file should be the string 'NumContrasts', but it is %s. Aborting! \n",tempString.c_str());
	        return EXIT_FAILURE;
	    }
	    contrasts >> NUMBER_OF_CONTRASTS;
			
	    if (NUMBER_OF_CONTRASTS <= 0)
	    {
	        contrasts.close();
   		    printf("Number of contrasts must be > 0 ! You provided %i in the contrasts file. Aborting! \n",NUMBER_OF_CONTRASTS);
	        return EXIT_FAILURE;
	    }
	    contrasts.close();
	}
  
	//------------------------------------------
    // Read data
	//------------------------------------------

	double startTime = GetWallTime();

    nifti_image *inputData = nifti_image_read(argv[1],1);
    
    if (inputData == NULL)
    {
        printf("Could not open volumes!\n");
        return EXIT_FAILURE;
    }
	allNiftiImages[numberOfNiftiImages] = inputData;
	numberOfNiftiImages++;
    
	nifti_image *inputMask;
	if (MASK)
	{
	    inputMask = nifti_image_read(MASK_NAME,1);
    
	    if (inputMask == NULL)
	    {
        	printf("Could not open mask volume!\n");
	        return EXIT_FAILURE;
	    }
		allNiftiImages[numberOfNiftiImages] = inputMask;
		numberOfNiftiImages++;
	}
	else
	{
       	printf("\nWarning: No mask being used, doing GLM for all voxels.\n\n");
	}
    	
	double endTime = GetWallTime();

	if (VERBOS)
 	{
		printf("It took %f seconds to read the nifti file(s)\n",(float)(endTime - startTime));
	}

    // Get data dimensions from input data
   	DATA_W = inputData->nx;
    DATA_H = inputData->ny;
    DATA_D = inputData->nz;    
    DATA_T = inputData->nt;    

    // Get voxel sizes
    VOXEL_SIZE_X = inputData->dx;
    VOXEL_SIZE_Y = inputData->dy;
    VOXEL_SIZE_Z = inputData->dz;

	// Get fMRI repetition time
    TR = inputData->dt;

	// Check if there is more than one volume
	if (DATA_T <= 1)
	{
		printf("Input data is a single volume, cannot run GLM! \n");
		FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
		return EXIT_FAILURE;	
	}

	// Check number of regressors
	if (NUMBER_OF_GLM_REGRESSORS > DATA_T)
	{
		printf("More regressor than data points, cannot run GLM! \n");
		FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
		return EXIT_FAILURE;	
	}
    	
	// Check if mask volume has the same dimensions as the data
	if (MASK)
	{
		int TEMP_DATA_W = inputMask->nx;
		int TEMP_DATA_H = inputMask->ny;
		int TEMP_DATA_D = inputMask->nz;

		if ( (TEMP_DATA_W != DATA_W) || (TEMP_DATA_H != DATA_H) || (TEMP_DATA_D != DATA_D) )
		{
			printf("Input data has the dimensions %i x %i x %i, while the mask volume has the dimensions %i x %i x %i. Aborting! \n",DATA_W,DATA_H,DATA_D,TEMP_DATA_W,TEMP_DATA_H,TEMP_DATA_D);
			FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
			return EXIT_FAILURE;
		}
	}
    

	
    // ------------------------------------------------  
	// Allocate memory
    // ------------------------------------------------  

	startTime = GetWallTime();

    size_t STATISTICAL_MAPS_SIZE, CONTRAST_SCALAR_SIZE;
	if (ANALYZE_TTEST)
	{
		CONTRAST_SCALAR_SIZE = NUMBER_OF_CONTRASTS * sizeof(float);
		STATISTICAL_MAPS_SIZE = DATA_W * DATA_H * DATA_D * NUMBER_OF_CONTRASTS * sizeof(float);
	}
	else if (ANALYZE_FTEST)
	{
		CONTRAST_SCALAR_SIZE = NUMBER_OF_CONTRASTS * NUMBER_OF_CONTRASTS * sizeof(float);
		STATISTICAL_MAPS_SIZE = DATA_W * DATA_H * DATA_D * sizeof(float);
	}

	if (SECOND_LEVEL)
	{
		NUMBER_OF_TOTAL_GLM_REGRESSORS = NUMBER_OF_GLM_REGRESSORS;
	}
	else if (FIRST_LEVEL)
	{
		NUMBER_OF_TOTAL_GLM_REGRESSORS = NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS + REGRESS_GLOBALMEAN + NUMBER_OF_MOTION_REGRESSORS * REGRESS_MOTION;
	}

    // ------------------------------------------------

    // Print some info
    if (PRINT)
    {
        printf("Authored by K.A. Eklund \n");
        printf("Data size: %i x %i x %i x %i \n",  DATA_W, DATA_H, DATA_D, DATA_T);
		if (SECOND_LEVEL)
		{
	        printf("Number of regressors: %i \n",  NUMBER_OF_GLM_REGRESSORS);
		}
		else
		{
			printf("Number of original regressors: %i \n",  NUMBER_OF_GLM_REGRESSORS);
			printf("Number of total regressors: %i \n",  NUMBER_OF_TOTAL_GLM_REGRESSORS);
		}
	
        printf("Number of contrasts: %i \n",  NUMBER_OF_CONTRASTS);
		if (BETAS_ONLY)
		{
			printf("Performing GLM and saving betas \n");
		}
		else if (CONTRASTS_ONLY)
		{
			printf("Performing GLM and saving contrasts \n");
		}
		else if (BETAS_AND_CONTRASTS_ONLY)
		{
			printf("Performing GLM and saving betas and contrasts \n");
		}
		else if (ANALYZE_TTEST)
		{
			printf("Performing %i t-tests \n",  NUMBER_OF_CONTRASTS);
		}
		else if (ANALYZE_FTEST)
		{
			printf("Performing one F-test \n");
		}
    }

    // ------------------------------------------------

    size_t DATA_SIZE = DATA_W * DATA_H * DATA_D * DATA_T * sizeof(float);
    size_t VOLUME_SIZE = DATA_W * DATA_H * DATA_D * sizeof(float);
      
    size_t GLM_SIZE = DATA_T * NUMBER_OF_GLM_REGRESSORS * sizeof(float);
    size_t CONTRAST_SIZE = NUMBER_OF_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float);
    size_t DESIGN_MATRIX_SIZE = NUMBER_OF_TOTAL_GLM_REGRESSORS * DATA_T * sizeof(float);
	size_t HIGHRES_REGRESSOR_SIZE = DATA_T * HIGHRES_FACTOR * sizeof(float);    
    size_t BETA_DATA_SIZE = DATA_W * DATA_H * DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float);
    size_t RESIDUALS_DATA_SIZE = DATA_W * DATA_H * DATA_D * DATA_T * sizeof(float);
    size_t MOTION_PARAMETERS_SIZE = NUMBER_OF_MOTION_REGRESSORS * DATA_T * sizeof(float);
   
	// If the data is in float format, we can just copy the pointer
	if ( inputData->datatype != DT_FLOAT )
	{
		AllocateMemory(h_Data, DATA_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "INPUT_VOLUMES");
	}
	else
	{
		allocatedHostMemory += DATA_SIZE;
	}

    AllocateMemory(h_X_GLM, GLM_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "DESIGN_MATRIX");
    AllocateMemory(h_Highres_Regressor, HIGHRES_REGRESSOR_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "HIGHRES_REGRESSOR");
    AllocateMemory(h_LowpassFiltered_Regressor, HIGHRES_REGRESSOR_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "LOWPASSFILTERED_REGRESSOR");
    AllocateMemory(h_xtxxt_GLM, GLM_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "DESIGN_MATRIX_INVERSE");
    AllocateMemory(h_Contrasts, CONTRAST_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "CONTRASTS");
    AllocateMemory(h_ctxtxc_GLM, CONTRAST_SCALAR_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "CONTRAST_SCALARS");
	AllocateMemory(h_Design_Matrix, DESIGN_MATRIX_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "TOTAL_DESIGN_MATRIX");
   	AllocateMemory(h_Design_Matrix2, DESIGN_MATRIX_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "TOTAL_DESIGN_MATRIX2");

	AllocateMemory(h_Mask, VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "MASK");

	if (REGRESS_MOTION)
	{
		AllocateMemory(h_Motion_Parameters, MOTION_PARAMETERS_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "MOTION_PARAMETERS");       
	}

    AllocateMemory(h_Beta_Volumes, BETA_DATA_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "BETA_VOLUMES");

	if (!BETAS_ONLY)
	{
		AllocateMemory(h_Contrast_Volumes, STATISTICAL_MAPS_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "CONTRAST_VOLUMES");
	}
	if (!BETAS_ONLY && !BETAS_AND_CONTRASTS_ONLY)
	{
		AllocateMemory(h_Statistical_Maps, STATISTICAL_MAPS_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "STATISTICALMAPS");
	}
	if (WRITE_RESIDUALS)
	{
		AllocateMemory(h_Residuals, RESIDUALS_DATA_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "RESIDUALS");  
	}
	AllocateMemory(h_Residual_Variances, VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "RESIDUAL_VARIANCES");  

    if (FIRST_LEVEL)
    {
        AllocateMemory(h_AR1_Estimates, VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "AR1_ESTIMATES");
        AllocateMemory(h_AR2_Estimates, VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "AR2_ESTIMATES");
        AllocateMemory(h_AR3_Estimates, VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "AR3_ESTIMATES");
        AllocateMemory(h_AR4_Estimates, VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "AR4_ESTIMATES");
    }

	endTime = GetWallTime();
    
	if (VERBOS)
 	{
		printf("It took %f seconds to allocate memory\n",(float)(endTime - startTime));
	}

    // ------------------------------------------------  
	// Read regressors and contrasts
    // ------------------------------------------------  

	if (RAW_DESIGNMATRIX)
	{
	    // Open design file
	    design.open(DESIGN_FILE);
	    // Read first two values
	    design >> tempString; // NumRegressors as string
	    design >> NUMBER_OF_GLM_REGRESSORS;

		float tempFloat;	
		for (int t = 0; t < DATA_T; t++)
		{
			for (int r = 0; r < NUMBER_OF_GLM_REGRESSORS; r++)
			{
				if (! (design >> h_X_GLM[t + r * DATA_T]) )
				{
					design.close();
			        printf("Could not read all values of the design file %s, aborting! Stopped reading at time point %i for regressor %i. Please check if the number of regressors and time points are correct. \n",DESIGN_FILE,t,r);      
			        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
			        FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
			        return EXIT_FAILURE;
				}
			}
		}	
		design.close();
	}
	else if (!RAW_DESIGNMATRIX)
	{
	    // Each line of the design file is a filename
    
	    // Open design file again
	    design.open(DESIGN_FILE);
	    // Read first two values again
	    design >> tempString; // NumRegressors as string
	    design >> NUMBER_OF_GLM_REGRESSORS;

		if (!RAW_REGRESSORS)
		{    
		    // Loop over the number of regressors provided in the design file
		    for (int r = 0; r < NUMBER_OF_GLM_REGRESSORS; r++)
	    	{
				// Reset highres regressor
			    for (int t = 0; t < DATA_T * HIGHRES_FACTOR; t++)
		    	{
					h_Highres_Regressor[t] = 0.0f;
				}

		        // Each regressor is a filename, so try to open the file
		        std::ifstream regressor;
		        std::string filename;
		        design >> filename;
		        regressor.open(filename.c_str());
		        if (!regressor.good())
		        {
		            regressor.close();
		            printf("Unable to open the regressor file %s . Aborting! \n",filename.c_str());
		            FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
		            FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
		            return EXIT_FAILURE;
		        }
        
		        // Read number of events for current regressor
		        regressor >> tempString; // NumEvents as string
		        std::string NE("NumEvents");
		        if (tempString.compare(NE) != 0)
		        {
    		        design.close();
		            printf("First element of each regressor file should be the string 'NumEvents', but it is %s for the regressor file %s. Aborting! \n",tempString.c_str(),filename.c_str());
		            FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
		            FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
		            return EXIT_FAILURE;
    		    }
		        regressor >> NUMBER_OF_EVENTS;

				if (DEBUG)
				{
					printf("Number of events for regressor %i is %i \n",r,NUMBER_OF_EVENTS);
				}
    	    
    		    // Loop over events
    		    for (int e = 0; e < NUMBER_OF_EVENTS; e++)
    		    {
    		        float onset;
    		        float duration;
    		        float value;
    	        
    		        // Read onset, duration and value for current event
					if (! (regressor >> onset) )
					{
						regressor.close();
    		            design.close();
    		            printf("Unable to read the onset for event %i in regressor file %s, aborting! Check the regressor file. \n",e,filename.c_str());
    		            FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
    		            FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
    		            return EXIT_FAILURE;
					}

    		        if (! (regressor >> duration) )
					{
						regressor.close();
    		            design.close();
    		            printf("Unable to read the duration for event %i in regressor file %s, aborting! Check the regressor file. \n",e,filename.c_str());
    		            FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
    		            FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
    		            return EXIT_FAILURE;
					}

					if (! (regressor >> value) )
					{
						regressor.close();
    		            design.close();
    		            printf("Unable to read the value for event %i in regressor file %s, aborting! Check the regressor file. \n",e,filename.c_str());
    		            FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
    		            FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
    		            return EXIT_FAILURE;
					}
    	    
					if (DEBUG)
					{
						printf("Event %i: Onset is %f, duration is %f and value is %f \n",e,onset,duration,value);
					}
    	        
    		        int start = (int)round(onset * (float)HIGHRES_FACTOR / TR);
    		        int activityLength = (int)round(duration * (float)HIGHRES_FACTOR / TR);
    	        
					if (DEBUG)
					{
						printf("Event %i: Start is %i, activity length is %i \n",e,start,activityLength);
					}

    		        // Put values into highres GLM
    		        for (int i = 0; i < activityLength; i++)
    		        {
    		            if ((start + i) < (DATA_T * HIGHRES_FACTOR) )
    		            {
    		                h_Highres_Regressor[start + i] = value;
    		            }
    		            else
    		            {
							regressor.close();
    		                design.close();
    		                printf("The activity start or duration for event %i in regressor file %s is longer than the duration of the fMRI data, aborting! Check the regressor file .\n",e,filename.c_str());	
    		                FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
    		                FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
    		                return EXIT_FAILURE;
    		            }
    		        }            
    		    }

				regressor.close();

    		    // Lowpass filter highres regressor
				LowpassFilterRegressor(h_LowpassFiltered_Regressor,h_Highres_Regressor,DATA_T,HIGHRES_FACTOR,TR);
        
    		    // Downsample highres GLM and put values into regular GLM
    		    for (int t = 0; t < DATA_T; t++)
    		    {
    		        h_X_GLM[t + r * DATA_T] = h_LowpassFiltered_Regressor[t*HIGHRES_FACTOR];
    		    }
    		}
		}
		else if (RAW_REGRESSORS)
		{
			// Loop over the number of regressors provided in the design file
		    for (int r = 0; r < NUMBER_OF_GLM_REGRESSORS; r++)
    		{
		        // Each regressor is a filename, so try to open the file
		        std::ifstream regressor;
		        std::string filename;
		        design >> filename;
		        regressor.open(filename.c_str());
		        if (!regressor.good())
		        {
		            regressor.close();
		            printf("Unable to open the regressor file %s . Aborting! \n",filename.c_str());
		            FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
		            FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
		            return EXIT_FAILURE;
		        }

				float value;
				int readValues = 0;
			    for (int t = 0; t < DATA_T; t++)
		    	{
					if (! (regressor >> value) )
					{
						regressor.close();
    		            design.close();
    		            printf("Unable to read the value for TR %i in regressor file %s, aborting! Check the regressor file. \n",t,filename.c_str());
    		            FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
    		            FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
    		            return EXIT_FAILURE;
					}
					h_X_GLM[t + r * DATA_T] = value;
					readValues++;
				}
	
				// Check if number of values is the same as the number of TRs
				if (readValues != DATA_T)
				{
					regressor.close();
    		        design.close();
    		        printf("Number of values in regressor file %s is not the same as the number of TRs in the fMRI data (%i vs %i), aborting! Check the regressor file. \n",filename.c_str(),readValues,DATA_T);
    		        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
    		        FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
    		        return EXIT_FAILURE;
				}
	
				regressor.close();
			}
		}
    	design.close();
	}

	//------------------------------------------
	// Read the contrasts
	//------------------------------------------

	if (!BETAS_ONLY)
	{
		// Open contrast file again
	    contrasts.open(CONTRASTS_FILE);

	    // Read first two values again
		contrasts >> tempString; // NumRegressors as string
	    contrasts >> tempNumber;
	    contrasts >> tempString; // NumContrasts as string
	    contrasts >> tempNumber;
   
		// Read all contrast values
		for (int c = 0; c < NUMBER_OF_CONTRASTS; c++)
		{
			for (int r = 0; r < NUMBER_OF_GLM_REGRESSORS; r++)
			{
				if (! (contrasts >> h_Contrasts[r + c * NUMBER_OF_GLM_REGRESSORS]) )
				{
				    contrasts.close();
	                printf("Unable to read all the contrast values, aborting! Check the contrasts file. \n");
	                FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
	                FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
	                return EXIT_FAILURE;
				}
			}
		}
		contrasts.close();
	}

	Eigen::MatrixXd Contrasts(NUMBER_OF_CONTRASTS,NUMBER_OF_GLM_REGRESSORS);

	// Read contrasts into Eigen object
	if (ANALYZE_FTEST)
	{
		for (int c = 0; c < NUMBER_OF_CONTRASTS; c++)
		{
			for (int r = 0; r < NUMBER_OF_GLM_REGRESSORS; r++)
			{
				Contrasts(c,r) = h_Contrasts[r + c * NUMBER_OF_GLM_REGRESSORS];		
			}
		}

		// Check if contrast matrix has full rank
		Eigen::FullPivLU<Eigen::MatrixXd> luA(Contrasts);
		int rank = luA.rank();
		if (rank < NUMBER_OF_CONTRASTS)
		{
	        printf("Contrast matrix does not have full rank, at least one contrast can be written as a linear combination of other contrasts, not OK for F-test, aborting!\n");      
	        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
	        FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
	        return EXIT_FAILURE;	
		}
	}

	// Put design matrix into Eigen object 
	Eigen::MatrixXd X(DATA_T,NUMBER_OF_GLM_REGRESSORS);

	for (int s = 0; s < DATA_T; s++)
	{
		for (int r = 0; r < NUMBER_OF_GLM_REGRESSORS; r++)
		{
			X(s,r) = (double)h_X_GLM[s + r * DATA_T];
		}
	}

	// Calculate pseudo inverse
	Eigen::MatrixXd xtx(NUMBER_OF_GLM_REGRESSORS,NUMBER_OF_GLM_REGRESSORS);
	xtx = X.transpose() * X;
	Eigen::MatrixXd inv_xtx = xtx.inverse();
	Eigen::MatrixXd xtxxt = inv_xtx * X.transpose();

	if (SECOND_LEVEL)
	{
		// Put pseudo inverse into regular array
		for (int s = 0; s < DATA_T; s++)
		{
			for (int r = 0; r < NUMBER_OF_GLM_REGRESSORS; r++)
			{
				h_xtxxt_GLM[s + r * DATA_T] = (float)xtxxt(r,s);
			}
		}
	}

	// Calculate contrast scalars
	if (ANALYZE_TTEST && SECOND_LEVEL)
	{
		for (int c = 0; c < NUMBER_OF_CONTRASTS; c++)
		{
			// Put contrast vector into eigen object
			Eigen::VectorXd Contrast(NUMBER_OF_GLM_REGRESSORS);
			for (int r = 0; r < NUMBER_OF_GLM_REGRESSORS; r++)
			{
				Contrast(r) = h_Contrasts[r + c * NUMBER_OF_GLM_REGRESSORS];		
			}
	
			Eigen::VectorXd scalar = Contrast.transpose() * inv_xtx * Contrast;
			h_ctxtxc_GLM[c] = scalar(0);
		}
	}
	else if (ANALYZE_FTEST && SECOND_LEVEL)
	{
		Eigen::MatrixXd temp = Contrasts * inv_xtx * Contrasts.transpose();
		Eigen::MatrixXd ctxtxc = temp.inverse();

		for (int c = 0; c < NUMBER_OF_CONTRASTS; c++)
		{
			for (int cc = 0; cc < NUMBER_OF_CONTRASTS; cc++)
			{
				h_ctxtxc_GLM[c + cc  * NUMBER_OF_CONTRASTS] = ctxtxc(c,cc);
			}
		}
	}

    // Write original design matrix to file
	if (WRITE_ORIGINAL_DESIGNMATRIX)
	{
		std::ofstream designmatrix;
	    designmatrix.open("original_designmatrix.txt");  

	    if ( designmatrix.good() )
	    {
    	    for (int t = 0; t < DATA_T; t++)
	        {
	    	    for (int r = 0; r < NUMBER_OF_GLM_REGRESSORS; r++)
		        {
            		designmatrix << std::setprecision(6) << std::fixed << (double)h_X_GLM[t + r * DATA_T] << "  ";
				}
				designmatrix << std::endl;
			}
		    designmatrix.close();
        } 	
	    else
	    {
			designmatrix.close();
	        printf("Could not open the file for writing the original design matrix!\n");
	    }
	}
	
    //------------------------------------------
	// Read motion parameters
	//------------------------------------------

	if (REGRESS_MOTION)
	{
	    // Open motion parameters file
		std::ifstream motionparameters;
	    motionparameters.open(MOTION_PARAMETERS_FILE);  

	    if ( motionparameters.good() )
	    {
			for (int t = 0; t < DATA_T; t++)
			{
				for (int r = 0; r < NUMBER_OF_MOTION_REGRESSORS; r++)
				{
					if (! (motionparameters >> h_Motion_Parameters[t + r * DATA_T]) )
					{
						motionparameters.close();
				        printf("Could not read all values of the motion parameters file %s, aborting! Stopped reading at time point %i for parameter %i. Please check if the motion parameters file\n",MOTION_PARAMETERS_FILE,t,r);      
				        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
				        FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
				        return EXIT_FAILURE;
					}
				}
			}	
			motionparameters.close();
		}
		else
		{
			motionparameters.close();
	        printf("Could not open the motion parameters file %s !\n",MOTION_PARAMETERS_FILE);
		}
	}

    //------------------------------------------
	// Read data
	//------------------------------------------

	startTime = GetWallTime();

    // Convert data to floats
    if ( inputData->datatype == DT_SIGNED_SHORT )
    {
        short int *p = (short int*)inputData->data;
    
        for (int i = 0; i < DATA_W * DATA_H * DATA_D * DATA_T; i++)
        {
            h_Data[i] = (float)p[i];
        }
    }
	else if ( inputData->datatype == DT_UINT8 )
    {
        unsigned char *p = (unsigned char*)inputData->data;
    
        for (int i = 0; i < DATA_W * DATA_H * DATA_D * DATA_T; i++)
        {
            h_Data[i] = (float)p[i];
        }
    }
    else if ( inputData->datatype == DT_UINT16 )
    {
        unsigned short int *p = (unsigned short int*)inputData->data;
    
        for (int i = 0; i < DATA_W * DATA_H * DATA_D * DATA_T; i++)
        {
            h_Data[i] = (float)p[i];
        }
    }
	// Correct data type, just copy the pointer
    else if ( inputData->datatype == DT_FLOAT )
    {		
		h_Data = (float*)inputData->data;

		// Save the pointer in the pointer list
		allMemoryPointers[numberOfMemoryPointers] = (void*)h_Data;
        numberOfMemoryPointers++;

        //float *p = (float*)inputfMRI->data;
    
        //for (int i = 0; i < EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T; i++)
        //{
        //    h_fMRI_Volumes[i] = p[i];
        //}
    }
    else
    {
        printf("Unknown data type in input data, aborting!\n");
        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
		FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
        return EXIT_FAILURE;
    }
    
	// Free input data, it has been converted to floats
	if ( inputData->datatype != DT_FLOAT )
	{		
		free(inputData->data);
		inputData->data = NULL;
	}
	// Pointer has been copied to h_Data and pointer list, so set the input data pointer to NULL
	else
	{		
		inputData->data = NULL;
	}

	// Mask is provided by user
	if (MASK)
	{
	    if ( inputMask->datatype == DT_SIGNED_SHORT )
	    {
	        short int *p = (short int*)inputMask->data;
    
	        for (int i = 0; i < DATA_W * DATA_H * DATA_D; i++)
	        {
	            h_Mask[i] = (float)p[i];
	        }
	    }
	    else if ( inputMask->datatype == DT_FLOAT )
	    {
	        float *p = (float*)inputMask->data;
    
	        for (int i = 0; i < DATA_W * DATA_H * DATA_D; i++)
        	{
	            h_Mask[i] = p[i];
	        }
	    }
	    else if ( inputMask->datatype == DT_UINT8 )
	    {
    	    unsigned char *p = (unsigned char*)inputMask->data;
    
	        for (int i = 0; i < DATA_W * DATA_H * DATA_D; i++)
	        {
	            h_Mask[i] = (float)p[i];
	        }
	    }
	    else
	    {
	        printf("Unknown data type in mask volume, aborting!\n");
	        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
			FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
	        return EXIT_FAILURE;
	    }
	}
	// Mask is NOT provided by user, set all mask voxels to 1
	else
	{
        for (int i = 0; i < DATA_W * DATA_H * DATA_D; i++)
        {
            h_Mask[i] = 1.0f;
        }
	}

	endTime = GetWallTime();

	if (VERBOS)
 	{
		printf("It took %f seconds to convert data to floats\n",(float)(endTime - startTime));
	}
    
    //------------------------
    
	startTime = GetWallTime();

	// Initialize BROCCOLI
    BROCCOLI_LIB BROCCOLI(OPENCL_PLATFORM,OPENCL_DEVICE,2,VERBOS); // 2 = Bash wrapper

	endTime = GetWallTime();

	if (VERBOS)
 	{
		printf("It took %f seconds to initiate BROCCOLI\n",(float)(endTime - startTime));
	}

    // Print build info to file (always)
	std::vector<std::string> buildInfo = BROCCOLI.GetOpenCLBuildInfo();
	std::vector<std::string> kernelFileNames = BROCCOLI.GetKernelFileNames();

	std::string buildInfoPath;
	buildInfoPath.append(getenv("BROCCOLI_DIR"));
	buildInfoPath.append("compiled/Kernels/");

	for (int k = 0; k < BROCCOLI.GetNumberOfKernelFiles(); k++)
	{
		std::string temp = buildInfoPath;
		temp.append("buildInfo_");
		temp.append(BROCCOLI.GetOpenCLPlatformName());
		temp.append("_");	
		temp.append(BROCCOLI.GetOpenCLDeviceName());
		temp.append("_");	
		std::string name = kernelFileNames[k];
		// Remove "kernel" and ".cpp" from kernel filename
		name = name.substr(0,name.size()-4);
		name = name.substr(6,name.size());
		temp.append(name);
		temp.append(".txt");
		fp = fopen(temp.c_str(),"w");
		if (fp == NULL)
		{     
		    printf("Could not open %s for writing ! \n",temp.c_str());
		}
		else
		{	
			if (buildInfo[k].c_str() != NULL)
			{
			    int error = fputs(buildInfo[k].c_str(),fp);
			    if (error == EOF)
			    {
			        printf("Could not write to %s ! \n",temp.c_str());
			    }
			}
			fclose(fp);
		}
	}
   
    // Something went wrong...
    if (!BROCCOLI.GetOpenCLInitiated())
    {              
        printf("Initialization error is \"%s\" \n",BROCCOLI.GetOpenCLInitializationError().c_str());
		printf("OpenCL error is \"%s\" \n",BROCCOLI.GetOpenCLError());

        // Print create kernel errors
        int* createKernelErrors = BROCCOLI.GetOpenCLCreateKernelErrors();
        for (int i = 0; i < BROCCOLI.GetNumberOfOpenCLKernels(); i++)
        {
            if (createKernelErrors[i] != 0)
            {
                printf("Create kernel error for kernel '%s' is '%s' \n",BROCCOLI.GetOpenCLKernelName(i),BROCCOLI.GetOpenCLErrorMessage(createKernelErrors[i]));
            }
        }                               		

        printf("OpenCL initialization failed, aborting! \nSee buildInfo* for output of OpenCL compilation!\n");      
        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
        FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
        return EXIT_FAILURE;
    }
    // Initialization OK
    else
    {        
		BROCCOLI.SetAllocatedHostMemory(allocatedHostMemory);

        BROCCOLI.SetNumberOfGLMRegressors(NUMBER_OF_GLM_REGRESSORS);
        BROCCOLI.SetNumberOfContrasts(NUMBER_OF_CONTRASTS);    
        BROCCOLI.SetDesignMatrix(h_X_GLM, h_xtxxt_GLM);
        BROCCOLI.SetContrasts(h_Contrasts);
        BROCCOLI.SetGLMScalars(h_ctxtxc_GLM);

		BROCCOLI.SetRawRegressors(RAW_REGRESSORS);
		BROCCOLI.SetRawDesignMatrix(RAW_DESIGNMATRIX);
		BROCCOLI.SetSaveDesignMatrix(WRITE_DESIGNMATRIX);
		BROCCOLI.SetSaveResidualsEPI(WRITE_RESIDUALS);
		BROCCOLI.SetSaveResidualsMNI(WRITE_RESIDUALS);
		BROCCOLI.SetSaveResidualVariances(WRITE_RESIDUAL_VARIANCES);

		BROCCOLI.SetOutputDesignMatrix(h_Design_Matrix, h_Design_Matrix2);
        BROCCOLI.SetOutputResidualVariances(h_Residual_Variances);        

		BROCCOLI.SetBetasOnly(BETAS_ONLY);
		BROCCOLI.SetContrastsOnly(CONTRASTS_ONLY);
		BROCCOLI.SetBetasAndContrastsOnly(BETAS_AND_CONTRASTS_ONLY);
       		
		BROCCOLI.SetPrint(PRINT);		

        // Run the GLM

		startTime = GetWallTime();
		if (ANALYZE_TTEST && SECOND_LEVEL)
		{
	        BROCCOLI.SetInputFirstLevelResults(h_Data);        
	        BROCCOLI.SetMask(h_Mask); 
       
	        BROCCOLI.SetMNIWidth(DATA_W);
	        BROCCOLI.SetMNIHeight(DATA_H);
	        BROCCOLI.SetMNIDepth(DATA_D);                
	        BROCCOLI.SetNumberOfSubjects(DATA_T);
			
    	    BROCCOLI.SetOutputBetaVolumesMNI(h_Beta_Volumes);  
			BROCCOLI.SetOutputContrastVolumesMNI(h_Contrast_Volumes);     
	        BROCCOLI.SetOutputStatisticalMapsMNI(h_Statistical_Maps);   
	        BROCCOLI.SetOutputResidualsMNI(h_Residuals);   

	        BROCCOLI.SetStatisticalTest(0); // t-test
	        BROCCOLI.PerformGLMTTestSecondLevelWrapper();                            
		}
		else if (ANALYZE_FTEST && SECOND_LEVEL)
		{
	        BROCCOLI.SetInputFirstLevelResults(h_Data);        
	        BROCCOLI.SetMask(h_Mask);        

	        BROCCOLI.SetMNIWidth(DATA_W);
	        BROCCOLI.SetMNIHeight(DATA_H);
	        BROCCOLI.SetMNIDepth(DATA_D);                
	        BROCCOLI.SetNumberOfSubjects(DATA_T);

    	    BROCCOLI.SetOutputBetaVolumesMNI(h_Beta_Volumes);        
			BROCCOLI.SetOutputContrastVolumesMNI(h_Contrast_Volumes);     
	        BROCCOLI.SetOutputStatisticalMapsMNI(h_Statistical_Maps);   
	        BROCCOLI.SetOutputResidualsMNI(h_Residuals);   

	        BROCCOLI.SetStatisticalTest(1); // F-test
	        BROCCOLI.PerformGLMFTestSecondLevelWrapper();                            
		}
		else if (ANALYZE_TTEST && FIRST_LEVEL)
		{
			BROCCOLI.SetInputfMRIVolumes(h_Data);
			BROCCOLI.SetOutputEPIMask(h_Mask);
			BROCCOLI.SetOutputMotionParameters(h_Motion_Parameters);

	        BROCCOLI.SetEPIWidth(DATA_W);
	        BROCCOLI.SetEPIHeight(DATA_H);
	        BROCCOLI.SetEPIDepth(DATA_D);
	        BROCCOLI.SetEPITimepoints(DATA_T);     
	        BROCCOLI.SetEPITR(TR); 
	        BROCCOLI.SetEPIVoxelSizeX(VOXEL_SIZE_X);
	        BROCCOLI.SetEPIVoxelSizeY(VOXEL_SIZE_Y);
	        BROCCOLI.SetEPIVoxelSizeZ(VOXEL_SIZE_Z);  
			BROCCOLI.SetARSmoothingAmount(AR_SMOOTHING_AMOUNT);
			BROCCOLI.SetTemporalDerivatives(USE_TEMPORAL_DERIVATIVES);
			BROCCOLI.SetRegressMotion(REGRESS_MOTION);
	        BROCCOLI.SetRegressGlobalMean(REGRESS_GLOBALMEAN);

    	    BROCCOLI.SetOutputBetaVolumesEPI(h_Beta_Volumes);        
			BROCCOLI.SetOutputContrastVolumesEPI(h_Contrast_Volumes);   
	        BROCCOLI.SetOutputStatisticalMapsEPI(h_Statistical_Maps);     
	        BROCCOLI.SetOutputResidualsEPI(h_Residuals);   
			BROCCOLI.SetOutputAREstimatesEPI(h_AR1_Estimates, h_AR2_Estimates, h_AR3_Estimates, h_AR4_Estimates);

	        BROCCOLI.SetStatisticalTest(0); // t-test
	        BROCCOLI.PerformGLMTTestFirstLevelWrapper();                            
		}
		else if (ANALYZE_FTEST && FIRST_LEVEL)
		{
			BROCCOLI.SetInputfMRIVolumes(h_Data);
			BROCCOLI.SetOutputEPIMask(h_Mask);
			BROCCOLI.SetOutputMotionParameters(h_Motion_Parameters);

	        BROCCOLI.SetEPIWidth(DATA_W);
	        BROCCOLI.SetEPIHeight(DATA_H);
	        BROCCOLI.SetEPIDepth(DATA_D);
	        BROCCOLI.SetEPITimepoints(DATA_T);     
	        BROCCOLI.SetEPITR(TR); 
	        BROCCOLI.SetEPIVoxelSizeX(VOXEL_SIZE_X);
	        BROCCOLI.SetEPIVoxelSizeY(VOXEL_SIZE_Y);
	        BROCCOLI.SetEPIVoxelSizeZ(VOXEL_SIZE_Z);  
			BROCCOLI.SetARSmoothingAmount(AR_SMOOTHING_AMOUNT);
			BROCCOLI.SetTemporalDerivatives(USE_TEMPORAL_DERIVATIVES);
			BROCCOLI.SetRegressMotion(REGRESS_MOTION);
	        BROCCOLI.SetRegressGlobalMean(REGRESS_GLOBALMEAN);

    	    BROCCOLI.SetOutputBetaVolumesEPI(h_Beta_Volumes);        
			BROCCOLI.SetOutputContrastVolumesEPI(h_Contrast_Volumes);     
	        BROCCOLI.SetOutputStatisticalMapsEPI(h_Statistical_Maps);     
	        BROCCOLI.SetOutputResidualsEPI(h_Residuals);   
			BROCCOLI.SetOutputAREstimatesEPI(h_AR1_Estimates, h_AR2_Estimates, h_AR3_Estimates, h_AR4_Estimates);

	        BROCCOLI.SetStatisticalTest(1); // F-test
	        BROCCOLI.PerformGLMFTestFirstLevelWrapper();                            
		}

		endTime = GetWallTime();

		if (VERBOS)
	 	{
			printf("\nIt took %f seconds to run the GLM\n",(float)(endTime - startTime));
		}

        // Print create buffer errors
        int* createBufferErrors = BROCCOLI.GetOpenCLCreateBufferErrors();
        for (int i = 0; i < BROCCOLI.GetNumberOfOpenCLKernels(); i++)
        {
            if (createBufferErrors[i] != 0)
            {
                printf("Create buffer error %i is %s \n",i,BROCCOLI.GetOpenCLErrorMessage(createBufferErrors[i]));
            }
        }
        
        // Print create kernel errors
        int* createKernelErrors = BROCCOLI.GetOpenCLCreateKernelErrors();
        for (int i = 0; i < BROCCOLI.GetNumberOfOpenCLKernels(); i++)
        {
            if (createKernelErrors[i] != 0)
            {
                printf("Create kernel error for kernel '%s' is '%s' \n",BROCCOLI.GetOpenCLKernelName(i),BROCCOLI.GetOpenCLErrorMessage(createKernelErrors[i]));
            }
        } 

        // Print run kernel errors
        int* runKernelErrors = BROCCOLI.GetOpenCLRunKernelErrors();
        for (int i = 0; i < BROCCOLI.GetNumberOfOpenCLKernels(); i++)
        {
            if (runKernelErrors[i] != 0)
            {
                printf("Run kernel error for kernel '%s' is '%s' \n",BROCCOLI.GetOpenCLKernelName(i),BROCCOLI.GetOpenCLErrorMessage(runKernelErrors[i]));
            }
        } 
    }        
       
	//-------------------------------------
	// Write total design matrix to file
	//-------------------------------------

	if (WRITE_DESIGNMATRIX)
	{
		std::ofstream designmatrix;
	    designmatrix.open("total_designmatrix.txt");  

	    if ( designmatrix.good() )
	    {
    	    for (int t = 0; t < DATA_T; t++)
	        {
	    	    for (int r = 0; r < NUMBER_OF_TOTAL_GLM_REGRESSORS; r++)
		        {
            		designmatrix << std::setprecision(6) << std::fixed << (double)h_Design_Matrix[t + r * DATA_T] << "  ";
				}
				designmatrix << std::endl;
			}
		    designmatrix.close();
        } 	
	    else
	    {
			designmatrix.close();
	        printf("Could not open the file for writing the design matrix!\n");
	    }
	}

	//-------------------------------------
	// Write results to nifti files
	//-------------------------------------

    // Create new nifti image
	nifti_image *outputNifti = nifti_copy_nim_info(inputData);      
	nifti_free_extensions(outputNifti);
    allNiftiImages[numberOfNiftiImages] = outputNifti;
	numberOfNiftiImages++;

    std::string beta = "_beta";
    std::string cope = "_cope";
    std::string tscores = "_tscores";
    std::string fscores = "_fscores";

	if (WRITE_RESIDUALS)
	{
		outputNifti->nt = DATA_T;
		outputNifti->ndim = 4;
		outputNifti->dim[0] = 4;
	    outputNifti->dim[4] = DATA_T;
	    outputNifti->nvox = DATA_W * DATA_H * DATA_D * DATA_T;
		
		WriteNifti(outputNifti,h_Residuals,"_residuals",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
	}

	outputNifti->nt = 1;
	outputNifti->ndim = 3;
	outputNifti->dim[0] = 3;
    outputNifti->dim[4] = 1;
    outputNifti->nvox = DATA_W *DATA_H * DATA_D;

	if (WRITE_RESIDUAL_VARIANCES)
	{	
		WriteNifti(outputNifti,h_Residual_Variances,"_residualvariance",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
	}

	// Write each beta weight as a separate file
	if (!CONTRASTS_ONLY)
	{
		for (int i = 0; i < NUMBER_OF_TOTAL_GLM_REGRESSORS; i++)
		{
			std::string temp = beta;
		    std::stringstream ss;
			if ((i+1) < 10)
			{
    	    	ss << "_regressor000";
			}
			else if ((i+1) < 100)
			{
				ss << "_regressor00";
			}
			else if ((i+1) < 1000)
			{
				ss << "_regressor0";
			}
			else
			{
				ss << "_regressor";
			}						
			ss << i + 1;
			temp.append(ss.str());
			WriteNifti(outputNifti,&h_Beta_Volumes[i * DATA_W * DATA_H * DATA_D],temp.c_str(),ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
		}
	}

    // Write each contrast volume as a separate file
	if (!BETAS_ONLY && !SECOND_LEVEL && !ANALYZE_FTEST)
	{
	    for (int i = 0; i < NUMBER_OF_CONTRASTS; i++)
    	{
	    	std::string temp = cope;
		    std::stringstream ss;
			if ((i+1) < 10)
			{
    			ss << "_contrast000";
			}
			else if ((i+1) < 100)
			{
				ss << "_contrast00";
			}
			else if ((i+1) < 1000)
			{
				ss << "_contrast0";
			}
			else
			{
				ss << "_contrast";
			}						
		    ss << i + 1;
		    temp.append(ss.str());
		    WriteNifti(outputNifti,&h_Contrast_Volumes[i * DATA_W * DATA_H * DATA_D],temp.c_str(),ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
		}
	}  

	if (!BETAS_ONLY && !CONTRASTS_ONLY && !BETAS_AND_CONTRASTS_ONLY)
	{
		if (ANALYZE_TTEST)
		{
	        // Write each t-map as a separate file
    	    for (int i = 0; i < NUMBER_OF_CONTRASTS; i++)
    	    {
				// nifti file contains t-scores
				outputNifti->intent_code = 3;
			
    	        std::string temp = tscores;
    	        std::stringstream ss;
				if ((i+1) < 10)
				{
		            ss << "_contrast000";
				}
				else if ((i+1) < 100)
				{
					ss << "_contrast00";
				}
				else if ((i+1) < 1000)
				{
					ss << "_contrast0";
				}
				else
				{
					ss << "_contrast";
				}						
    	        ss << i + 1;
    	        temp.append(ss.str());
    		    WriteNifti(outputNifti,&h_Statistical_Maps[i * DATA_W * DATA_H * DATA_D],temp.c_str(),ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
        	}
		}
		else if (ANALYZE_FTEST)
		{
		    WriteNifti(outputNifti,h_Statistical_Maps,"_fscores",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
		}
	}

	if (WRITE_AR_ESTIMATES)
	{
		WriteNifti(outputNifti,h_AR1_Estimates,"_ar1",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
		WriteNifti(outputNifti,h_AR2_Estimates,"_ar2",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
		WriteNifti(outputNifti,h_AR3_Estimates,"_ar3",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
		WriteNifti(outputNifti,h_AR4_Estimates,"_ar4",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
	}

	endTime = GetWallTime();

	if (VERBOS)
 	{
		printf("It took %f seconds to write the nifti file(s)\n",(float)(endTime - startTime));
	}

    // Free all memory
    FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
    FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
        
    return EXIT_SUCCESS;
}


