
#include <time.h>
#include "nifti1_io.h"
#include "broccoli_lib.h"

void CreateFilename(char *& filenameWithExtension, nifti_image* inputNifti, const char* extension, bool CHANGE_OUTPUT_FILENAME, const char* outputFilename);

void LowpassFilterRegressor(float* h_LowpassFiltered_Regressor, float* h_Regressor, int DATA_T, int HIGHRES_FACTOR, float TR);

void LowpassFilterRegressors(float* h_LowpassFiltered_Regressors, float* h_Regressors, int DATA_T, int HIGHRES_FACTOR, float TR, int NUMBER_OF_REGRESSORS);

void ConvertFloat2ToFloats(float* Real, float* Imag, cl_float2* Complex, int DATA_W, int DATA_H, int DATA_D);

void FreeAllMemory(void **pointers, int N);

void FreeAllNiftiImages(nifti_image **niftiImages, int N);

void ReadBinaryFile(float* pointer, int size, const char* filename, void** pointers, int& Npointers, nifti_image** niftiImages, int Nimages);

void AllocateMemory(float *& pointer, size_t size, void** pointers, int& Npointers, nifti_image** niftiImages, int Nimages, size_t& allocatedMemory, const char* variable);

void AllocateMemoryInt(unsigned short int *& pointer, size_t size, void** pointers, int& Npointers, nifti_image** niftiImages, int Nimages, size_t allocatedMemory, const char* variable);

void AllocateMemoryInt2(int *& pointer, size_t size, void** pointers, int& Npointers, nifti_image** niftiImages, int Nimages, size_t allocatedMemory, const char* variable);
    
void AllocateMemoryFloat2(cl_float2 *& pointer, int size, void** pointers, int& Npointers, nifti_image** niftiImages, int Nimages, size_t allocatedMemory, const char* variable);

float mymax(float* data, int N);

float mymin(float* data, int N);

bool WriteNifti(nifti_image* inputNifti, float* data, const char* filename, bool addFilename, bool checkFilename);

double GetWallTime();
