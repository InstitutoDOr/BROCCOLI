/*
    BROCCOLI: Software for Fast fMRI Analysis on Many-Core CPUs and GPUs
    Copyright (C) <2013>  Anders Eklund, andek034@gmail.com
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
    
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
    INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
    PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
    FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
    OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
    DEALINGS IN THE SOFTWARE.
    
    Added sphere radius for 2 and 3 voxels. Sebastian Hoefle - IDOR
 
*/


// Help functions
int Calculate2DIndex(int x, int y, int DATA_W)
{
	return x + y * DATA_W;
}

int Calculate3DIndex(int x, int y, int z, int DATA_W, int DATA_H)
{
	return x + y * DATA_W + z * DATA_W * DATA_H;
}

int Calculate4DIndex(int x, int y, int z, int t, int DATA_W, int DATA_H, int DATA_D)
{
	return x + y * DATA_W + z * DATA_W * DATA_H + t * DATA_W * DATA_H * DATA_D;
}




void ReadSphere(__local float* Volume,
                __global const float* Volumes,
                int x,
                int y,
                int z,
                int t,
                int3 tIdx,
                int DATA_W,
                int DATA_H,
                int DATA_D)
{
    
    Volume[Calculate3DIndex(tIdx.x,tIdx.y,tIdx.z, 16, 16)] = 0.0f;
    Volume[Calculate3DIndex(tIdx.x + 8,tIdx.y,tIdx.z, 16, 16)] = 0.0f;
    Volume[Calculate3DIndex(tIdx.x,tIdx.y + 8,tIdx.z, 16, 16)] = 0.0f;
    Volume[Calculate3DIndex(tIdx.x + 8,tIdx.y + 8,tIdx.z, 16, 16)] = 0.0f;
    Volume[Calculate3DIndex(tIdx.x,tIdx.y,tIdx.z + 8, 16, 16)] = 0.0f;
    Volume[Calculate3DIndex(tIdx.x + 8,tIdx.y,tIdx.z + 8, 16, 16)] = 0.0f;
    Volume[Calculate3DIndex(tIdx.x,tIdx.y + 8,tIdx.z + 8, 16, 16)] = 0.0f;
    Volume[Calculate3DIndex(tIdx.x + 8,tIdx.y + 8,tIdx.z + 8, 16, 16)] = 0.0f;
    
    
    // X, Y, Z
    if ( ((x - 4) < DATA_W) && ((x - 4) >= 0) && ((y - 4) < DATA_H) && ((y - 4) >= 0) && ((z - 4) < DATA_D) && ((z - 4) >= 0) )
    {
        Volume[Calculate3DIndex(tIdx.x,tIdx.y,tIdx.z, 16, 16)] = Volumes[Calculate4DIndex(x - 4,y - 4,z - 4,t,DATA_W, DATA_H, DATA_D)];
    }
    
    // X + 8, Y, Z
    if ( ((x + 4) < DATA_W) && ((y - 4) < DATA_H) && ((y - 4) >= 0) && ((z - 4) < DATA_D) && ((z - 4) >= 0) )
    {
        Volume[Calculate3DIndex(tIdx.x + 8,tIdx.y,tIdx.z, 16, 16)] = Volumes[Calculate4DIndex(x + 4,y - 4,z - 4,t,DATA_W, DATA_H, DATA_D)];
    }
    
    // X, Y + 8, Z
    if ( ((x - 4) < DATA_W) && ((x - 4) >= 0) && ((y + 4) < DATA_H) && ((z - 4) < DATA_D) && ((z - 4) >= 0) )
    {
        Volume[Calculate3DIndex(tIdx.x,tIdx.y + 8,tIdx.z, 16, 16)] = Volumes[Calculate4DIndex(x - 4,y + 4,z - 4,t,DATA_W, DATA_H, DATA_D)];
    }
    
    // X + 8, Y + 8, Z
    if ( ((x + 4) < DATA_W) && ((y + 4) < DATA_H) && ((z - 4) < DATA_D) && ((z - 4) >= 0) )
    {
        Volume[Calculate3DIndex(tIdx.x + 8,tIdx.y + 8,tIdx.z, 16, 16)]= Volumes[Calculate4DIndex(x + 4,y + 4,z - 4,t,DATA_W, DATA_H, DATA_D)];
    }
    
    
    
    // X, Y, Z + 8
    if ( ((x - 4) < DATA_W) && ((x - 4) >= 0) && ((y - 4) < DATA_H) && ((y - 4) >= 0) && ((z + 4) < DATA_D) )
    {
        Volume[Calculate3DIndex(tIdx.x,tIdx.y,tIdx.z + 8, 16, 16)] = Volumes[Calculate4DIndex(x - 4,y - 4,z + 4,t,DATA_W, DATA_H, DATA_D)];
    }
    
    // X + 8, Y, Z + 8
    if ( ((x + 4) < DATA_W) && ((y - 4) < DATA_H) && ((y - 4) >= 0) && ((z + 4) < DATA_D) )
    {
        Volume[Calculate3DIndex(tIdx.x + 8,tIdx.y,tIdx.z + 8, 16, 16)] = Volumes[Calculate4DIndex(x + 4,y - 4,z + 4,t,DATA_W, DATA_H, DATA_D)];
    }
    
    
    // X, Y + 8, Z + 8
    if ( ((x - 4) < DATA_W) && ((x - 4) >= 0) && ((y + 4) < DATA_H) && ((z + 4) < DATA_D) )
    {
        Volume[Calculate3DIndex(tIdx.x,tIdx.y + 8,tIdx.z + 8, 16, 16)] = Volumes[Calculate4DIndex(x - 4,y + 4,z + 4,t,DATA_W, DATA_H, DATA_D)];
    }
    
    // X + 8, Y + 8, Z + 8
    if ( ((x + 4) < DATA_W) && ((y + 4) < DATA_H) && ((z + 4) < DATA_D) )
    {
        Volume[Calculate3DIndex(tIdx.x + 8,tIdx.y + 8,tIdx.z + 8, 16, 16)] = Volumes[Calculate4DIndex(x + 4,y + 4,z + 4,t,DATA_W, DATA_H, DATA_D)];
    }
}




__kernel void CalculateStatisticalMapSearchlight_(__global float* Classifier_Performance,
                                                        __global const float* Volumes,
                                                        __global const float* Mask,
                                                        __constant float* c_d,
                                                        __constant float* c_Correct_Classes,
                                                        __private int DATA_W,
                                                        __private int DATA_H,
                                                        __private int DATA_D,
                                                        __private int NUMBER_OF_VOLUMES,
                                                        __private float n,
                                                        __private int EPOCS)

{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    
    int3 tIdx = {get_local_id(0), get_local_id(1), get_local_id(2)};
    
    if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
        return;
    
    if ( Mask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] != 1.0f )
    {
        Classifier_Performance[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = 0.0f;
        return;
    }
    
    __local float l_Volume[16][16][16];    // z, y, x
    
    int classification_performance = 0;
	
    // Leave one out cross validation
    for (int validation = 0; validation < NUMBER_OF_VOLUMES; validation++)
    {
        float weights[20];
        
        weights[0]  = 0.0f;
        weights[1]  = 0.0f;
        weights[2]  = 0.0f;
        weights[3]  = 0.0f;
        weights[4]  = 0.0f;
        weights[5]  = 0.0f;
        weights[6]  = 0.0f;
        weights[7]  = 0.0f;
        weights[8]  = 0.0f;
        weights[9]  = 0.0f;
        weights[10] = 0.0f;
        weights[11] = 0.0f;
        weights[12] = 0.0f;
        weights[13] = 0.0f;
        weights[14] = 0.0f;
        weights[15] = 0.0f;
        weights[16] = 0.0f;
        weights[17] = 0.0f;
        weights[18] = 0.0f;
        weights[19] = 0.0f;
        
        // Do training for a number of iterations
        for (int epoc = 0; epoc < EPOCS; epoc++)
        {
            float gradient[20];
            
            gradient[0] = 0.0f;
            gradient[1] = 0.0f;
            gradient[2] = 0.0f;
            gradient[3] = 0.0f;
            gradient[4] = 0.0f;
            gradient[5] = 0.0f;
            gradient[6] = 0.0f;
            gradient[7] = 0.0f;
            gradient[8] = 0.0f;
            gradient[9] = 0.0f;
            gradient[10] = 0.0f;
            gradient[11] = 0.0f;
            gradient[12] = 0.0f;
            gradient[13] = 0.0f;
            gradient[14] = 0.0f;
            gradient[15] = 0.0f;
            gradient[16] = 0.0f;
            gradient[17] = 0.0f;
            gradient[18] = 0.0f;
            gradient[19] = 0.0f;
            
            for (int t = 0; t < NUMBER_OF_VOLUMES; t++)
            {
                // Skip training with validation time point
                if (t == validation)
                {
                    continue;
                }
                
                float s;
                
                // Classification for current timepoint
                ReadSphere((__local float*)l_Volume, Volumes, x, y, z, t, tIdx, DATA_W, DATA_H, DATA_D);
                
                // Make sure all threads have written to local memory
                barrier(CLK_LOCAL_MEM_FENCE);
                
                // Make classification
                s =  weights[0] * 1.0f;
                
                // z - 1
                s += weights[1] * l_Volume[tIdx.z + 4 - 1][tIdx.y + 4][tIdx.x + 4 - 1]; 		//
                s += weights[2] * l_Volume[tIdx.z + 4 - 1][tIdx.y + 4 - 1][tIdx.x + 4]; 		//
                s += weights[3] * l_Volume[tIdx.z + 4 - 1][tIdx.y + 4][tIdx.x + 4]; 			// center pixel
                s += weights[4] * l_Volume[tIdx.z + 4 - 1][tIdx.y + 4 + 1][tIdx.x + 4]; 		//
                s += weights[5] * l_Volume[tIdx.z + 4 - 1][tIdx.y + 4][tIdx.x + 4 + 1]; 		//
                
                // z
                s += weights[6] * l_Volume[tIdx.z + 4][tIdx.y + 4 - 1][tIdx.x + 4 - 1]; 	//
                s += weights[7] * l_Volume[tIdx.z + 4][tIdx.y + 4 - 1][tIdx.x + 4]; 		//
                s += weights[8] * l_Volume[tIdx.z + 4][tIdx.y + 4 - 1][tIdx.x + 4 + 1]; 	//
                s += weights[9] * l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 4 - 1]; 		//
                s += weights[10] * l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 4]; 			//	center pixel
                s += weights[11] * l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 4 + 1]; 		//
                s += weights[12] * l_Volume[tIdx.z + 4][tIdx.y + 4 + 1][tIdx.x + 4 - 1]; 	//
                s += weights[13] * l_Volume[tIdx.z + 4][tIdx.y + 4 + 1][tIdx.x + 4]; 		//
                s += weights[14] * l_Volume[tIdx.z + 4][tIdx.y + 4 + 1][tIdx.x + 4 + 1]; 	//
                
                // z + 1
                s += weights[15] * l_Volume[tIdx.z + 4 + 1][tIdx.y + 4][tIdx.x + 4 - 1]; 		//
                s += weights[16] * l_Volume[tIdx.z + 4 + 1][tIdx.y + 4 - 1][tIdx.x + 4]; 		//
                s += weights[17] * l_Volume[tIdx.z + 4 + 1][tIdx.y + 4][tIdx.x + 4]; 			// center pixel
                s += weights[18] * l_Volume[tIdx.z + 4 + 1][tIdx.y + 4 + 1][tIdx.x + 4]; 		//
                s += weights[19] * l_Volume[tIdx.z + 4 + 1][tIdx.y + 4][tIdx.x + 4 + 1]; 		//
                
                // Calculate contribution to gradient
                gradient[0] += (s - c_d[t]) * 1.0f;
                
                // z - 1
                gradient[1]  += (s - c_d[t]) * l_Volume[tIdx.z + 4 - 1][tIdx.y + 4][tIdx.x + 4 - 1]; 		//
                gradient[2]  += (s - c_d[t]) * l_Volume[tIdx.z + 4 - 1][tIdx.y + 4 - 1][tIdx.x + 4]; 		//
                gradient[3]  += (s - c_d[t]) * l_Volume[tIdx.z + 4 - 1][tIdx.y + 4][tIdx.x + 4]; 			// center pixel
                gradient[4]  += (s - c_d[t]) * l_Volume[tIdx.z + 4 - 1][tIdx.y + 4 + 1][tIdx.x + 4]; 		//
                gradient[5]  += (s - c_d[t]) * l_Volume[tIdx.z + 4 - 1][tIdx.y + 4][tIdx.x + 4 + 1]; 		//
                
                // z
                gradient[6]  += (s - c_d[t]) * l_Volume[tIdx.z + 4][tIdx.y + 4 - 1][tIdx.x + 4 - 1]; 		//
                gradient[7]  += (s - c_d[t]) * l_Volume[tIdx.z + 4][tIdx.y + 4 - 1][tIdx.x + 4]; 			//
                gradient[8]  += (s - c_d[t]) * l_Volume[tIdx.z + 4][tIdx.y + 4 - 1][tIdx.x + 4 + 1]; 		//
                gradient[9]  += (s - c_d[t]) * l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 4 - 1]; 			//
                gradient[10] += (s - c_d[t]) * l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 4]; 				//	center pixel
                gradient[11] += (s - c_d[t]) * l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 4 + 1]; 			//
                gradient[12] += (s - c_d[t]) * l_Volume[tIdx.z + 4][tIdx.y + 4 + 1][tIdx.x + 4 - 1]; 		//
                gradient[13] += (s - c_d[t]) * l_Volume[tIdx.z + 4][tIdx.y + 4 + 1][tIdx.x + 4]; 			//
                gradient[14] += (s - c_d[t]) * l_Volume[tIdx.z + 4][tIdx.y + 4 + 1][tIdx.x + 4 + 1]; 		//
                
                // z + 1
                gradient[15] += (s - c_d[t]) * l_Volume[tIdx.z + 4 + 1][tIdx.y + 4][tIdx.x + 4 - 1]; 		//
                gradient[16] += (s - c_d[t]) * l_Volume[tIdx.z + 4 + 1][tIdx.y + 4 - 1][tIdx.x + 4]; 		//
                gradient[17] += (s - c_d[t]) * l_Volume[tIdx.z + 4 + 1][tIdx.y + 4][tIdx.x + 4]; 			// center pixel
                gradient[18] += (s - c_d[t]) * l_Volume[tIdx.z + 4 + 1][tIdx.y + 4 + 1][tIdx.x + 4]; 		//
                gradient[19] += (s - c_d[t]) * l_Volume[tIdx.z + 4 + 1][tIdx.y + 4][tIdx.x + 4 + 1]; 		//
                
                
                // end for t
            }
            
            // Update weights
            weights[0] -= n/(float)NUMBER_OF_VOLUMES * gradient[0];
            weights[1] -= n/(float)NUMBER_OF_VOLUMES * gradient[1];
            weights[2] -= n/(float)NUMBER_OF_VOLUMES * gradient[2];
            weights[3] -= n/(float)NUMBER_OF_VOLUMES * gradient[3];
            weights[4] -= n/(float)NUMBER_OF_VOLUMES * gradient[4];
            weights[5] -= n/(float)NUMBER_OF_VOLUMES * gradient[5];
            weights[6] -= n/(float)NUMBER_OF_VOLUMES * gradient[6];
            weights[7] -= n/(float)NUMBER_OF_VOLUMES * gradient[7];
            weights[8] -= n/(float)NUMBER_OF_VOLUMES * gradient[8];
            weights[9] -= n/(float)NUMBER_OF_VOLUMES * gradient[9];
            weights[10] -= n/(float)NUMBER_OF_VOLUMES * gradient[10];
            weights[11] -= n/(float)NUMBER_OF_VOLUMES * gradient[11];
            weights[12] -= n/(float)NUMBER_OF_VOLUMES * gradient[12];
            weights[13] -= n/(float)NUMBER_OF_VOLUMES * gradient[13];
            weights[14] -= n/(float)NUMBER_OF_VOLUMES * gradient[14];
            weights[15] -= n/(float)NUMBER_OF_VOLUMES * gradient[15];
            weights[16] -= n/(float)NUMBER_OF_VOLUMES * gradient[16];
            weights[17] -= n/(float)NUMBER_OF_VOLUMES * gradient[17];
            weights[18] -= n/(float)NUMBER_OF_VOLUMES * gradient[18];
            weights[19] -= n/(float)NUMBER_OF_VOLUMES * gradient[19];
            
            // end for epocs
        }
        
        // Make classification on validation timepoint
        
        ReadSphere((__local float*)l_Volume, Volumes, x, y, z, validation, tIdx, DATA_W, DATA_H, DATA_D);
        
        // Make sure all threads have written to local memory
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Make classification
        float s;
        s =  weights[0] * 1.0f;
        
        // z - 1
        s += weights[1] * l_Volume[tIdx.z + 4 - 1][tIdx.y + 4][tIdx.x + 4 - 1]; 		//
        s += weights[2] * l_Volume[tIdx.z + 4 - 1][tIdx.y + 4 - 1][tIdx.x + 4]; 		//
        s += weights[3] * l_Volume[tIdx.z + 4 - 1][tIdx.y + 4][tIdx.x + 4]; 			// center pixel
        s += weights[4] * l_Volume[tIdx.z + 4 - 1][tIdx.y + 4 + 1][tIdx.x + 4]; 		//
        s += weights[5] * l_Volume[tIdx.z + 4 - 1][tIdx.y + 4][tIdx.x + 4 + 1]; 		//
        
        // z
        s += weights[6] * l_Volume[tIdx.z + 4][tIdx.y + 4 - 1][tIdx.x + 4 - 1]; 		//
        s += weights[7] * l_Volume[tIdx.z + 4][tIdx.y + 4 - 1][tIdx.x + 4]; 			//
        s += weights[8] * l_Volume[tIdx.z + 4][tIdx.y + 4 - 1][tIdx.x + 4 + 1]; 		//
        s += weights[9] * l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 4 - 1]; 			//
        s += weights[10] * l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 4]; 			//	center pixel
        s += weights[11] * l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 4 + 1]; 		//
        s += weights[12] * l_Volume[tIdx.z + 4][tIdx.y + 4 + 1][tIdx.x + 4 - 1]; 	//
        s += weights[13] * l_Volume[tIdx.z + 4][tIdx.y + 4 + 1][tIdx.x + 4]; 		//
        s += weights[14] * l_Volume[tIdx.z + 4][tIdx.y + 4 + 1][tIdx.x + 4 + 1]; 	//
        
        // z + 1
        s += weights[15] * l_Volume[tIdx.z + 4 + 1][tIdx.y + 4][tIdx.x + 4 - 1]; 		//
        s += weights[16] * l_Volume[tIdx.z + 4 + 1][tIdx.y + 4 - 1][tIdx.x + 4]; 		//
        s += weights[17] * l_Volume[tIdx.z + 4 + 1][tIdx.y + 4][tIdx.x + 4]; 			// center pixel
        s += weights[18] * l_Volume[tIdx.z + 4 + 1][tIdx.y + 4 + 1][tIdx.x + 4]; 		//
        s += weights[19] * l_Volume[tIdx.z + 4 + 1][tIdx.y + 4][tIdx.x + 4 + 1]; 		//
        
        float classification;
        if (s > 0.0f)
        {
            classification = 0.0f;
        }
        else
        {
            classification = 1.0f;
        }
        
        if (classification == c_Correct_Classes[validation])
        {
            classification_performance++;
        }
        
        // end for validation
    }
    
    Classifier_Performance[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = (float)classification_performance / (float)NUMBER_OF_VOLUMES;    
}

__kernel void CalculateStatisticalMapSearchlight_19(__global float* Classifier_Performance,
                                                  __global const float* Volumes,
                                                  __global const float* Mask,
                                                  __constant float* c_d,
                                                  __constant float* c_Correct_Classes,
                                                  __private int DATA_W,
                                                  __private int DATA_H,
                                                  __private int DATA_D,
                                                  __private int NUMBER_OF_VOLUMES,
                                                  __private float n,
                                                  __private int EPOCS)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    
    int3 tIdx = {get_local_id(0), get_local_id(1), get_local_id(2)};
    
    if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
        return;
    
    if ( Mask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] != 1.0f )
    {
        Classifier_Performance[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = 0.0f;
        return;
    }
    
    if ( ((x + 1) >= DATA_W) || ((y + 1) >= DATA_H) || ((z + 1) >= DATA_D) )
    {
        Classifier_Performance[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = 0.0f;
        return;
    }
    
    
    if ( ((x - 1) < 0) || ((y - 1) < 0) || ((z - 1) < 0) )
    {
        Classifier_Performance[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = 0.0f;
        return;
    }
    
    
    int classification_performance = 0;
    
    float weights[20];

	int uncensoredVolumes = 0;
    bool printed = false;

    // Leave one out cross validation
    for (int validation = 0; validation < NUMBER_OF_VOLUMES; validation++)
    {
		// Skip testing with censored volumes
        if (c_Correct_Classes[validation] == 9999.0f)
        {
            continue;
        } 

		uncensoredVolumes++;
       
        weights[0]  = 0.0f;
        weights[1]  = 0.0f;
        weights[2]  = 0.0f;
        weights[3]  = 0.0f;
        weights[4]  = 0.0f;
        weights[5]  = 0.0f;
        weights[6]  = 0.0f;
        weights[7]  = 0.0f;
        weights[8]  = 0.0f;
        weights[9]  = 0.0f;
        weights[10] = 0.0f;
        weights[11] = 0.0f;
        weights[12] = 0.0f;
        weights[13] = 0.0f;
        weights[14] = 0.0f;
        weights[15] = 0.0f;
        weights[16] = 0.0f;
        weights[17] = 0.0f;
        weights[18] = 0.0f;
        weights[19] = 0.0f;
        
        // Do training for a number of iterations
        for (int epoc = 0; epoc < EPOCS; epoc++)
        {
            float gradient[20];
            
            gradient[0] = 0.0f;
            gradient[1] = 0.0f;
            gradient[2] = 0.0f;
            gradient[3] = 0.0f;
            gradient[4] = 0.0f;
            gradient[5] = 0.0f;
            gradient[6] = 0.0f;
            gradient[7] = 0.0f;
            gradient[8] = 0.0f;
            gradient[9] = 0.0f;
            gradient[10] = 0.0f;
            gradient[11] = 0.0f;
            gradient[12] = 0.0f;
            gradient[13] = 0.0f;
            gradient[14] = 0.0f;
            gradient[15] = 0.0f;
            gradient[16] = 0.0f;
            gradient[17] = 0.0f;
            gradient[18] = 0.0f;
            gradient[19] = 0.0f;
            
            for (int t = 0; t < NUMBER_OF_VOLUMES; t++)
            {
                // Skip training with validation volume and censored volumes
                if ((t == validation) || (c_Correct_Classes[t] == 9999.0f))
                {
                    continue;
                }                                
                
                // Make classification
                float s;
                s =  weights[0] * 1.0f;
                
                float x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19;
                
                x1 = Volumes[Calculate4DIndex(x-1,y,z-1,t,DATA_W,DATA_H,DATA_D)];
                x2 = Volumes[Calculate4DIndex(x,y-1,z-1,t,DATA_W,DATA_H,DATA_D)];
                x3 = Volumes[Calculate4DIndex(x,y,z-1,t,DATA_W,DATA_H,DATA_D)];
                x4 = Volumes[Calculate4DIndex(x,y+1,z-1,t,DATA_W,DATA_H,DATA_D)];
                x5 = Volumes[Calculate4DIndex(x+1,y,z-1,t,DATA_W,DATA_H,DATA_D)];
                
                x6 = Volumes[Calculate4DIndex(x-1,y-1,z,t,DATA_W,DATA_H,DATA_D)];
                x7 = Volumes[Calculate4DIndex(x-1,y,z,t,DATA_W,DATA_H,DATA_D)];
                x8 = Volumes[Calculate4DIndex(x-1,y+1,z,t,DATA_W,DATA_H,DATA_D)];
                x9 = Volumes[Calculate4DIndex(x,y-1,z,t,DATA_W,DATA_H,DATA_D)];
                x10 = Volumes[Calculate4DIndex(x,y,z,t,DATA_W,DATA_H,DATA_D)];
                x11 = Volumes[Calculate4DIndex(x,y+1,z,t,DATA_W,DATA_H,DATA_D)];
                x12 = Volumes[Calculate4DIndex(x+1,y-1,z,t,DATA_W,DATA_H,DATA_D)];
                x13 = Volumes[Calculate4DIndex(x+1,y,z,t,DATA_W,DATA_H,DATA_D)];
                x14 = Volumes[Calculate4DIndex(x+1,y+1,z,t,DATA_W,DATA_H,DATA_D)];
                
                x15 = Volumes[Calculate4DIndex(x-1,y,z+1,t,DATA_W,DATA_H,DATA_D)];
                x16 = Volumes[Calculate4DIndex(x,y-1,z+1,t,DATA_W,DATA_H,DATA_D)];
                x17 = Volumes[Calculate4DIndex(x,y,z+1,t,DATA_W,DATA_H,DATA_D)];
                x18 = Volumes[Calculate4DIndex(x,y+1,z+1,t,DATA_W,DATA_H,DATA_D)];
                x19 = Volumes[Calculate4DIndex(x+1,y,z+1,t,DATA_W,DATA_H,DATA_D)];
                
                // z - 1
                s += weights[1] * x1;
                s += weights[2] * x2;
                s += weights[3] * x3;
                s += weights[4] * x4;
                s += weights[5] * x5;
                
                // z
                s += weights[6] * x6;
                s += weights[7] * x7;
                s += weights[8] * x8;
                s += weights[9] * x9;
                s += weights[10] * x10;
                s += weights[11] * x11;
                s += weights[12] * x12;
                s += weights[13] * x13;
                s += weights[14] * x14;
                
                // z + 1
                s += weights[15] * x15;
                s += weights[16] * x16;
                s += weights[17] * x17;
                s += weights[18] * x18;
                s += weights[19] * x19;
                
                // Calculate contribution to gradient
                gradient[0] += (s - c_d[t]) * 1.0f;
                
                // z - 1
                gradient[1]  += (s - c_d[t]) * x1;
                gradient[2]  += (s - c_d[t]) * x2;
                gradient[3]  += (s - c_d[t]) * x3;
                gradient[4]  += (s - c_d[t]) * x4;
                gradient[5]  += (s - c_d[t]) * x5;
                
                // z
                gradient[6]  += (s - c_d[t]) * x6;
                gradient[7]  += (s - c_d[t]) * x7;
                gradient[8]  += (s - c_d[t]) * x8;
                gradient[9]  += (s - c_d[t]) * x9;
                gradient[10] += (s - c_d[t]) * x10;
                gradient[11] += (s - c_d[t]) * x11;
                gradient[12] += (s - c_d[t]) * x12;
                gradient[13] += (s - c_d[t]) * x13;
                gradient[14] += (s - c_d[t]) * x14;
                
                // z + 1
                gradient[15] += (s - c_d[t]) * x15;
                gradient[16] += (s - c_d[t]) * x16;
                gradient[17] += (s - c_d[t]) * x17;
                gradient[18] += (s - c_d[t]) * x18;
                gradient[19] += (s - c_d[t]) * x19;
                
                // end for t
            }
            
            // Update weights
            weights[0] -= n/(float)NUMBER_OF_VOLUMES * gradient[0];
            weights[1] -= n/(float)NUMBER_OF_VOLUMES * gradient[1];
            weights[2] -= n/(float)NUMBER_OF_VOLUMES * gradient[2];
            weights[3] -= n/(float)NUMBER_OF_VOLUMES * gradient[3];
            weights[4] -= n/(float)NUMBER_OF_VOLUMES * gradient[4];
            weights[5] -= n/(float)NUMBER_OF_VOLUMES * gradient[5];
            weights[6] -= n/(float)NUMBER_OF_VOLUMES * gradient[6];
            weights[7] -= n/(float)NUMBER_OF_VOLUMES * gradient[7];
            weights[8] -= n/(float)NUMBER_OF_VOLUMES * gradient[8];
            weights[9] -= n/(float)NUMBER_OF_VOLUMES * gradient[9];
            weights[10] -= n/(float)NUMBER_OF_VOLUMES * gradient[10];
            weights[11] -= n/(float)NUMBER_OF_VOLUMES * gradient[11];
            weights[12] -= n/(float)NUMBER_OF_VOLUMES * gradient[12];
            weights[13] -= n/(float)NUMBER_OF_VOLUMES * gradient[13];
            weights[14] -= n/(float)NUMBER_OF_VOLUMES * gradient[14];
            weights[15] -= n/(float)NUMBER_OF_VOLUMES * gradient[15];
            weights[16] -= n/(float)NUMBER_OF_VOLUMES * gradient[16];
            weights[17] -= n/(float)NUMBER_OF_VOLUMES * gradient[17];
            weights[18] -= n/(float)NUMBER_OF_VOLUMES * gradient[18];
            weights[19] -= n/(float)NUMBER_OF_VOLUMES * gradient[19];
        
            // end for epocs
        }
        
        // Make classification
        float s;
        s =  weights[0] * 1.0f;
        
        float x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19;
        
        x1 = Volumes[Calculate4DIndex(x-1,y,z-1,validation,DATA_W,DATA_H,DATA_D)];
        x2 = Volumes[Calculate4DIndex(x,y-1,z-1,validation,DATA_W,DATA_H,DATA_D)];
        x3 = Volumes[Calculate4DIndex(x,y,z-1,validation,DATA_W,DATA_H,DATA_D)];
        x4 = Volumes[Calculate4DIndex(x,y+1,z-1,validation,DATA_W,DATA_H,DATA_D)];
        x5 = Volumes[Calculate4DIndex(x+1,y,z-1,validation,DATA_W,DATA_H,DATA_D)];
        
        x6 = Volumes[Calculate4DIndex(x-1,y-1,z,validation,DATA_W,DATA_H,DATA_D)];
        x7 = Volumes[Calculate4DIndex(x-1,y,z,validation,DATA_W,DATA_H,DATA_D)];
        x8 = Volumes[Calculate4DIndex(x-1,y+1,z,validation,DATA_W,DATA_H,DATA_D)];
        x9 = Volumes[Calculate4DIndex(x,y-1,z,validation,DATA_W,DATA_H,DATA_D)];
        x10 = Volumes[Calculate4DIndex(x,y,z,validation,DATA_W,DATA_H,DATA_D)];
        x11 = Volumes[Calculate4DIndex(x,y+1,z,validation,DATA_W,DATA_H,DATA_D)];
        x12 = Volumes[Calculate4DIndex(x+1,y-1,z,validation,DATA_W,DATA_H,DATA_D)];
        x13 = Volumes[Calculate4DIndex(x+1,y,z,validation,DATA_W,DATA_H,DATA_D)];
        x14 = Volumes[Calculate4DIndex(x+1,y+1,z,validation,DATA_W,DATA_H,DATA_D)];
        
        x15 = Volumes[Calculate4DIndex(x-1,y,z+1,validation,DATA_W,DATA_H,DATA_D)];
        x16 = Volumes[Calculate4DIndex(x,y-1,z+1,validation,DATA_W,DATA_H,DATA_D)];
        x17 = Volumes[Calculate4DIndex(x,y,z+1,validation,DATA_W,DATA_H,DATA_D)];
        x18 = Volumes[Calculate4DIndex(x,y+1,z+1,validation,DATA_W,DATA_H,DATA_D)];
        x19 = Volumes[Calculate4DIndex(x+1,y,z+1,validation,DATA_W,DATA_H,DATA_D)];
        
        //if (x==9 && y==22 && z==26)
        if (x1 == 0.0f || x2 == 0.0f || x3 == 0.0f || x4 == 0.0f || x5 == 0.0f || x6 == 0.0f || x7 == 0.0f || x8 == 0.0f || x9 == 0.0f || x10 == 0.0f || x11 == 0.0f || x12 == 0.0f || x13 == 0.0f || x14 == 0.0f || x15 == 0.0f || x16 == 0.0f || x17 == 0.0f || x18 == 0.0f || x19 == 0.0f )
        {
        	printf("classifying %i\n", validation);
            printf("1:%f 2:%f 3:%f 4:%f 5:%f 6:%f 7:%f 8:%f 9:%f 10:%f 11:%f 12:%f 13:%f 14:%f 15:%f 16:%f 17:%f 18:%f 19:%f\n", x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19);
            printed = true;
        }

        // z - 1
        s += weights[1] * x1;
        s += weights[2] * x2;
        s += weights[3] * x3;
        s += weights[4] * x4;
        s += weights[5] * x5;
        
        // z
        s += weights[6] * x6;
        s += weights[7] * x7;
        s += weights[8] * x8;
        s += weights[9] * x9;
        s += weights[10] * x10;
        s += weights[11] * x11;
        s += weights[12] * x12;
        s += weights[13] * x13;
        s += weights[14] * x14;
        
        // z + 1
        s += weights[15] * x15;
        s += weights[16] * x16;
        s += weights[17] * x17;
        s += weights[18] * x18;
        s += weights[19] * x19;
        
        float classification;
        if (s > 0.0f)
        {
            classification = 0.0f;
        }
        else
        {
            classification = 1.0f;
        }
        
        if (classification == c_Correct_Classes[validation])
        {
            classification_performance++;
        }
        
        // end for validation
    }
    
    Classifier_Performance[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = (float)classification_performance / (float)uncensoredVolumes;
}

void normalize_x(__global float *x_space, int N, int d);

float doLeaveOneOut(__global float *dense_points, __global int *trainIndex, __global int *testIndex, int d, __global float *alph,
		 __constant float *c_Correct_Classes, int trainN, int numExemplos);

float doFold(__global float *dense_points, __global int *trainIndex, __global int *testIndex, int d, __global float *alpha,
		 __constant float *c_Correct_Classes, int trainN, int numExemplos, __global const float* kmatrix, int fold);

__kernel void PrepareInputSearchlight(	__global float* Volumes, 			// 0
										__global const int *voxelIndex1D, 	// 1
										__private int NUMBER_OF_VOLUMES, 	// 2
										__private int SIZ_VOLUME, 			// 3
										__private int VOXELS_MASK) 			// 4
{
	int voxind = get_global_id(0);
	int volumeId = voxelIndex1D[voxind];

	if (voxind >= VOXELS_MASK)
		return;

	float eps = 0.00001;
	float minimum = Volumes[volumeId];
	float maximum = minimum;

	int x=41,y=17,z=25;
	int DATA_W=53;
	int DATA_H=63;
	int testIndex = Calculate3DIndex(x,y,z,DATA_W,DATA_H);
	//if (testIndex != volumeId)
	//	return;

	//printf("%d\n", SIZ_VOLUME);
	// calculate maximum and minimum over time
	for (int t=1; t<NUMBER_OF_VOLUMES; ++t)
	{
		if( Volumes[volumeId+t*SIZ_VOLUME] > maximum)
			maximum = Volumes[volumeId+t*SIZ_VOLUME];
		if( Volumes[volumeId+t*SIZ_VOLUME] < minimum)
			minimum = Volumes[volumeId+t*SIZ_VOLUME];
	}

	// get scaling values to scale between -1 and +1
	float scale = 2/(maximum-minimum);
	float minimum_scaled = minimum * scale;

	if ((maximum-minimum) < eps )
	{
		scale = 1;
		minimum_scaled = 0;

		// if value is different from zero
		if (fabs(maximum) > eps)
		{
			scale = 1/maximum;
			minimum_scaled = 0;
		}
	}

	//printf("voxind=%d volumeId=%d scale=%f minimum_scaled=%f minimum= %f maximum=%f\n", voxind, volumeId, scale, minimum_scaled, minimum, maximum);

	// apply scaling
	for (int t=0; t<NUMBER_OF_VOLUMES; ++t)
	{
		Volumes[volumeId+t*SIZ_VOLUME] = Volumes[volumeId+t*SIZ_VOLUME] * scale - minimum_scaled - 1.;
		//printf("x[%i][%i]=%f ", k, m, x_space[k*d+m] );
	}
}

float dotproduct(__global float *a, __global float *b, int N )
{
	float ret=0;
	for (int k=0; k<N; ++k)
	{
		ret += a[k]*b[k];
	}
	return ret;
}

__kernel void PrepareSearchlight( 	__global const float* Volumes, 		// 0
									__global const int *voxelIndex1D, 	// 1
									__constant int *deltaIndex, 		// 2
									__private int SIZ_VOLUME,			// 3
									__private int NUMBER_OF_VOLUMES, 	// 4
									__private int VOXELS_MASK, 			// 5
									__private int NFEAT,				// 6
									__global float* x_space, 			// 7
									__global float *kmatrix, 			// 8
									__private int voxbatchoffset, 		// 9
									__private int voxbatchsize)			// 10

{

	int voxind = get_global_id(0);

	if (voxind >= voxbatchsize )
		return;

	// apply voxel batch offset
	int volumeId = voxelIndex1D[voxind + voxbatchoffset];

	if (volumeId+deltaIndex[0] < 0 || volumeId+deltaIndex[NFEAT-1] >= SIZ_VOLUME)
	{
		return;
	}

	int x=41,y=17,z=25;
	int DATA_W=53;
	int DATA_H=63;
	int testIndex = Calculate3DIndex(x,y,z,DATA_W,DATA_H);
	//if (testIndex != volumeId)
	//	return;

	// populate x
	int voxoffset = voxind*NFEAT*NUMBER_OF_VOLUMES;
	//if (testIndex == volumeId)
	//    	printf("voxind: %d x_space:\n",voxind);

	for (int t = 0; t < NUMBER_OF_VOLUMES; t++)
    {
	   for (int k=0; k<NFEAT; k++)
	   {
		   x_space[t*NFEAT + k + voxoffset] = Volumes[volumeId+deltaIndex[k]+t*SIZ_VOLUME]; // + c_d[t]

		   //if (testIndex == volumeId)
			//   printf("%f ", x_space[t*NFEAT + k + voxoffset]);
	   }
	  // if (testIndex == volumeId)
	   	//    printf("\n");
    }

    // add sphere offset based on voxel index
    x_space = x_space + voxind*NFEAT*NUMBER_OF_VOLUMES;
    kmatrix = kmatrix + voxind*NUMBER_OF_VOLUMES*NUMBER_OF_VOLUMES;

    // compute kernel matrix
    //if (testIndex == volumeId)
    //	printf("voxind: %d kernelmatrix:\n",voxind);

	for (int a=0; a<NUMBER_OF_VOLUMES; ++a)
	{
		for (int b=0; b<NUMBER_OF_VOLUMES; ++b)
		{
			kmatrix[a+b*NUMBER_OF_VOLUMES] = dotproduct(x_space + a*NFEAT, x_space + b*NFEAT, NFEAT);

			//if (testIndex == volumeId)
			 //   printf("%f ",kmatrix[a+b*NUMBER_OF_VOLUMES]);

		}
		//if (testIndex == volumeId)
		//	printf("\n");
	}

}

__kernel void CalculateStatisticalMapSearchlight_SVM( __global float* Classifier_Performance, 	// 0
												  __global const float* Volumes, 			// 1
                                                  __global const float* Mask, 				// 2
                                                  __constant float* c_d, 					// 3
                                                  __constant float* c_Correct_Classes, 		// 4
                                                  __private int DATA_W, 					// 5
                                                  __private int DATA_H, 					// 6
                                                  __private int DATA_D, 					// 7
                                                  __private int NUMBER_OF_VOLUMES, 			// 8
                                                  __private float n, 						// 9
                                                  __private int EPOCS, 						// 10
                                                  __global float* x_space, 					// 11
                                                  __global int* trainIndex, 				// 12
                                                  __global int* testIndex, 					// 13
                                                  __global float* alph, 					// 14
                                                  __global const int *voxelIndex1D, 		// 15
                                                  __constant int *deltaIndex, 				// 16
                                                  __private int VOXELS_MASK, 				// 17
                                                  __global const float *kmatrix, 			// 18
                                                  __private int NFEAT, 						// 19
                                                  __private int fold,						// 20
												  __private int voxbatchoffset, 			// 21
									              __private int voxbatchsize)				// 22
{
    int voxind = get_global_id(0);

    // apply voxbatchoffset
    int voxindWithOffset = voxind+voxbatchoffset;

    int volumeId = voxelIndex1D[voxindWithOffset];

    int x=41,y=17,z=25;
   	int testvoxIndex = Calculate3DIndex(x,y,z,DATA_W,DATA_H);

   //	if (testvoxIndex != volumeId)
   //		return;

    int SIZ_VOLUME = DATA_W*DATA_H*DATA_D;

    if (volumeId+deltaIndex[0] < 0 || volumeId+deltaIndex[NFEAT-1] >= SIZ_VOLUME)
    {
        Classifier_Performance[volumeId] = 0.0f;
        return;
    }

	int trainN = NUMBER_OF_VOLUMES-1;

    //printf("calling leaveoneout\n");
    int trainOffset = voxindWithOffset*(NUMBER_OF_VOLUMES-1);

    float margin = doFold( x_space + voxind*NFEAT*NUMBER_OF_VOLUMES, trainIndex+trainOffset, testIndex+voxindWithOffset, NFEAT, alph+trainOffset, c_d, trainN, NUMBER_OF_VOLUMES, kmatrix+voxind*NUMBER_OF_VOLUMES*NUMBER_OF_VOLUMES, fold);

 /*   if (testvoxIndex == volumeId)
    		printf("margin=%f verdadeiro=%f\n", margin, c_d[fold]);
*/
    int n_correct = 0;
    if (margin > 0 == c_d[fold] > 0)
	{
		n_correct++;
	}

    float accuracy = (float)n_correct / (float)(NUMBER_OF_VOLUMES);


	/*if (testvoxIndex == volumeId)
		printf("x_space within svm searchlight\n");


	x_space = x_space + voxind*NFEAT*NUMBER_OF_VOLUMES;
	for (int t = 0; t < NUMBER_OF_VOLUMES; t++)
	{
	   for (int k=0; k<NFEAT; k++)
	   {
		   if (testvoxIndex == volumeId)
			   printf("%f ", x_space[t*NFEAT + k]);
	   }
	   if (testvoxIndex == volumeId)
			printf("\n");
	}*/


/*
	if (testvoxIndex == volumeId)
	{
    	printf("voxind: %d volumeId: %d acc before: %f acc: %f\n",voxind, volumeId,Classifier_Performance[volumeId],accuracy);
    	//for(int k=0; k<NUMBER_OF_VOLUMES; ++k)
    		//printf("y[%d]=%f ", k, c_d[k]);
	}
*/
    Classifier_Performance[volumeId] = Classifier_Performance[volumeId] + accuracy; //(float)classification_performance / (float)uncensoredVolumes;
}

// sphere radius of 2 voxels resulting in 33 voxels
__kernel void CalculateStatisticalMapSearchlight_33(__global float* Classifier_Performance,
                                                     __global const float* Volumes,
                                                     __global const float* Mask,
                                                     __constant float* c_d,
                                                     __constant float* c_Correct_Classes,
                                                     __private int DATA_W,
                                                     __private int DATA_H,
                                                     __private int DATA_D,
                                                     __private int NUMBER_OF_VOLUMES,
                                                     __private float n,
                                                     __private int EPOCS)

{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    
    int3 tIdx = {get_local_id(0), get_local_id(1), get_local_id(2)};
    
    if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
        return;
    
    if ( Mask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] != 1.0f )
    {
        Classifier_Performance[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = 0.0f;
        return;
    }
    
    if ( ((x + 2) >= DATA_W) || ((y + 2) >= DATA_H) || ((z + 2) >= DATA_D) )
    {
        Classifier_Performance[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = 0.0f;
        return;
    }
    
    
    if ( ((x - 2) < 0) || ((y - 2) < 0) || ((z - 2) < 0) )
    {
        Classifier_Performance[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = 0.0f;
        return;
    }
    
    
    int classification_performance = 0;
    
    float weights[34]; // we use 33 voxels
    
    int uncensoredVolumes = 0;
    
    // Leave one out cross validation
    for (int validation = 0; validation < NUMBER_OF_VOLUMES; validation++)
    {
        // Skip testing with censored volumes
        if (c_Correct_Classes[validation] == 9999.0f)
        {
            continue;
        }
        
        uncensoredVolumes++;
        
        weights[0] = 0.0f;
        weights[1] = 0.0f;
        weights[2] = 0.0f;
        weights[3] = 0.0f;
        weights[4] = 0.0f;
        weights[5] = 0.0f;
        weights[6] = 0.0f;
        weights[7] = 0.0f;
        weights[8] = 0.0f;
        weights[9] = 0.0f;
        weights[10] = 0.0f;
        weights[11] = 0.0f;
        weights[12] = 0.0f;
        weights[13] = 0.0f;
        weights[14] = 0.0f;
        weights[15] = 0.0f;
        weights[16] = 0.0f;
        weights[17] = 0.0f;
        weights[18] = 0.0f;
        weights[19] = 0.0f;
        weights[20] = 0.0f;
        weights[21] = 0.0f;
        weights[22] = 0.0f;
        weights[23] = 0.0f;
        weights[24] = 0.0f;
        weights[25] = 0.0f;
        weights[26] = 0.0f;
        weights[27] = 0.0f;
        weights[28] = 0.0f;
        weights[29] = 0.0f;
        weights[30] = 0.0f;
        weights[31] = 0.0f;
        weights[32] = 0.0f;
        weights[33] = 0.0f;
        
        
        // Do training for a number of iterations
        for (int epoc = 0; epoc < EPOCS; epoc++)
        {
            float gradient[34];
            
            gradient[0] = 0.0f;
            gradient[1] = 0.0f;
            gradient[2] = 0.0f;
            gradient[3] = 0.0f;
            gradient[4] = 0.0f;
            gradient[5] = 0.0f;
            gradient[6] = 0.0f;
            gradient[7] = 0.0f;
            gradient[8] = 0.0f;
            gradient[9] = 0.0f;
            gradient[10] = 0.0f;
            gradient[11] = 0.0f;
            gradient[12] = 0.0f;
            gradient[13] = 0.0f;
            gradient[14] = 0.0f;
            gradient[15] = 0.0f;
            gradient[16] = 0.0f;
            gradient[17] = 0.0f;
            gradient[18] = 0.0f;
            gradient[19] = 0.0f;
            gradient[20] = 0.0f;
            gradient[21] = 0.0f;
            gradient[22] = 0.0f;
            gradient[23] = 0.0f;
            gradient[24] = 0.0f;
            gradient[25] = 0.0f;
            gradient[26] = 0.0f;
            gradient[27] = 0.0f;
            gradient[28] = 0.0f;
            gradient[29] = 0.0f;
            gradient[30] = 0.0f;
            gradient[31] = 0.0f;
            gradient[32] = 0.0f;
            gradient[33] = 0.0f;
            
            for (int t = 0; t < NUMBER_OF_VOLUMES; t++)
            {
                
                // Skip training with validation volume and censored volumes
                if ((t == validation) || (c_Correct_Classes[t] == 9999.0f))
                {
                    continue;
                }
                
                // Make classification
                float s;
                s =  weights[0] * 1.0f;
                
                float x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33;
                x1 = Volumes[Calculate4DIndex(x+0,y+0,z-2,t,DATA_W,DATA_H,DATA_D)];
                x2 = Volumes[Calculate4DIndex(x-1,y-1,z-1,t,DATA_W,DATA_H,DATA_D)];
                x3 = Volumes[Calculate4DIndex(x-1,y+0,z-1,t,DATA_W,DATA_H,DATA_D)];
                x4 = Volumes[Calculate4DIndex(x-1,y+1,z-1,t,DATA_W,DATA_H,DATA_D)];
                x5 = Volumes[Calculate4DIndex(x+0,y-1,z-1,t,DATA_W,DATA_H,DATA_D)];
                x6 = Volumes[Calculate4DIndex(x+0,y+0,z-1,t,DATA_W,DATA_H,DATA_D)];
                x7 = Volumes[Calculate4DIndex(x+0,y+1,z-1,t,DATA_W,DATA_H,DATA_D)];
                x8 = Volumes[Calculate4DIndex(x+1,y-1,z-1,t,DATA_W,DATA_H,DATA_D)];
                x9 = Volumes[Calculate4DIndex(x+1,y+0,z-1,t,DATA_W,DATA_H,DATA_D)];
                x10 = Volumes[Calculate4DIndex(x+1,y+1,z-1,t,DATA_W,DATA_H,DATA_D)];
                x11 = Volumes[Calculate4DIndex(x-2,y+0,z+0,t,DATA_W,DATA_H,DATA_D)];
                x12 = Volumes[Calculate4DIndex(x-1,y-1,z+0,t,DATA_W,DATA_H,DATA_D)];
                x13 = Volumes[Calculate4DIndex(x-1,y+0,z+0,t,DATA_W,DATA_H,DATA_D)];
                x14 = Volumes[Calculate4DIndex(x-1,y+1,z+0,t,DATA_W,DATA_H,DATA_D)];
                x15 = Volumes[Calculate4DIndex(x+0,y-2,z+0,t,DATA_W,DATA_H,DATA_D)];
                x16 = Volumes[Calculate4DIndex(x+0,y-1,z+0,t,DATA_W,DATA_H,DATA_D)];
                x17 = Volumes[Calculate4DIndex(x+0,y+0,z+0,t,DATA_W,DATA_H,DATA_D)];
                x18 = Volumes[Calculate4DIndex(x+0,y+1,z+0,t,DATA_W,DATA_H,DATA_D)];
                x19 = Volumes[Calculate4DIndex(x+0,y+2,z+0,t,DATA_W,DATA_H,DATA_D)];
                x20 = Volumes[Calculate4DIndex(x+1,y-1,z+0,t,DATA_W,DATA_H,DATA_D)];
                x21 = Volumes[Calculate4DIndex(x+1,y+0,z+0,t,DATA_W,DATA_H,DATA_D)];
                x22 = Volumes[Calculate4DIndex(x+1,y+1,z+0,t,DATA_W,DATA_H,DATA_D)];
                x23 = Volumes[Calculate4DIndex(x+2,y+0,z+0,t,DATA_W,DATA_H,DATA_D)];
                x24 = Volumes[Calculate4DIndex(x-1,y-1,z+1,t,DATA_W,DATA_H,DATA_D)];
                x25 = Volumes[Calculate4DIndex(x-1,y+0,z+1,t,DATA_W,DATA_H,DATA_D)];
                x26 = Volumes[Calculate4DIndex(x-1,y+1,z+1,t,DATA_W,DATA_H,DATA_D)];
                x27 = Volumes[Calculate4DIndex(x+0,y-1,z+1,t,DATA_W,DATA_H,DATA_D)];
                x28 = Volumes[Calculate4DIndex(x+0,y+0,z+1,t,DATA_W,DATA_H,DATA_D)];
                x29 = Volumes[Calculate4DIndex(x+0,y+1,z+1,t,DATA_W,DATA_H,DATA_D)];
                x30 = Volumes[Calculate4DIndex(x+1,y-1,z+1,t,DATA_W,DATA_H,DATA_D)];
                x31 = Volumes[Calculate4DIndex(x+1,y+0,z+1,t,DATA_W,DATA_H,DATA_D)];
                x32 = Volumes[Calculate4DIndex(x+1,y+1,z+1,t,DATA_W,DATA_H,DATA_D)];
                x33 = Volumes[Calculate4DIndex(x+0,y+0,z+2,t,DATA_W,DATA_H,DATA_D)];
                
                // weights sum
                s += weights[1] * x1;
                s += weights[2] * x2;
                s += weights[3] * x3;
                s += weights[4] * x4;
                s += weights[5] * x5;
                s += weights[6] * x6;
                s += weights[7] * x7;
                s += weights[8] * x8;
                s += weights[9] * x9;
                s += weights[10] * x10;
                s += weights[11] * x11;
                s += weights[12] * x12;
                s += weights[13] * x13;
                s += weights[14] * x14;
                s += weights[15] * x15;
                s += weights[16] * x16;
                s += weights[17] * x17;
                s += weights[18] * x18;
                s += weights[19] * x19;
                s += weights[20] * x20;
                s += weights[21] * x21;
                s += weights[22] * x22;
                s += weights[23] * x23;
                s += weights[24] * x24;
                s += weights[25] * x25;
                s += weights[26] * x26;
                s += weights[27] * x27;
                s += weights[28] * x28;
                s += weights[29] * x29;
                s += weights[30] * x30;
                s += weights[31] * x31;
                s += weights[32] * x32;
                s += weights[33] * x33;
                
                // Calculate contribution to gradient
                gradient[0] += (s - c_d[t]) * 1.0f;
                gradient[1] += (s - c_d[t]) * x1;
                gradient[2] += (s - c_d[t]) * x2;
                gradient[3] += (s - c_d[t]) * x3;
                gradient[4] += (s - c_d[t]) * x4;
                gradient[5] += (s - c_d[t]) * x5;
                gradient[6] += (s - c_d[t]) * x6;
                gradient[7] += (s - c_d[t]) * x7;
                gradient[8] += (s - c_d[t]) * x8;
                gradient[9] += (s - c_d[t]) * x9;
                gradient[10] += (s - c_d[t]) * x10;
                gradient[11] += (s - c_d[t]) * x11;
                gradient[12] += (s - c_d[t]) * x12;
                gradient[13] += (s - c_d[t]) * x13;
                gradient[14] += (s - c_d[t]) * x14;
                gradient[15] += (s - c_d[t]) * x15;
                gradient[16] += (s - c_d[t]) * x16;
                gradient[17] += (s - c_d[t]) * x17;
                gradient[18] += (s - c_d[t]) * x18;
                gradient[19] += (s - c_d[t]) * x19;
                gradient[20] += (s - c_d[t]) * x20;
                gradient[21] += (s - c_d[t]) * x21;
                gradient[22] += (s - c_d[t]) * x22;
                gradient[23] += (s - c_d[t]) * x23;
                gradient[24] += (s - c_d[t]) * x24;
                gradient[25] += (s - c_d[t]) * x25;
                gradient[26] += (s - c_d[t]) * x26;
                gradient[27] += (s - c_d[t]) * x27;
                gradient[28] += (s - c_d[t]) * x28;
                gradient[29] += (s - c_d[t]) * x29;
                gradient[30] += (s - c_d[t]) * x30;
                gradient[31] += (s - c_d[t]) * x31;
                gradient[32] += (s - c_d[t]) * x32;
                gradient[33] += (s - c_d[t]) * x33;
                
                // end for t
            }
            
            // Update weights
            weights[0] -= n/(float)NUMBER_OF_VOLUMES * gradient[0];
            weights[1] -= n/(float)NUMBER_OF_VOLUMES * gradient[1];
            weights[2] -= n/(float)NUMBER_OF_VOLUMES * gradient[2];
            weights[3] -= n/(float)NUMBER_OF_VOLUMES * gradient[3];
            weights[4] -= n/(float)NUMBER_OF_VOLUMES * gradient[4];
            weights[5] -= n/(float)NUMBER_OF_VOLUMES * gradient[5];
            weights[6] -= n/(float)NUMBER_OF_VOLUMES * gradient[6];
            weights[7] -= n/(float)NUMBER_OF_VOLUMES * gradient[7];
            weights[8] -= n/(float)NUMBER_OF_VOLUMES * gradient[8];
            weights[9] -= n/(float)NUMBER_OF_VOLUMES * gradient[9];
            weights[10] -= n/(float)NUMBER_OF_VOLUMES * gradient[10];
            weights[11] -= n/(float)NUMBER_OF_VOLUMES * gradient[11];
            weights[12] -= n/(float)NUMBER_OF_VOLUMES * gradient[12];
            weights[13] -= n/(float)NUMBER_OF_VOLUMES * gradient[13];
            weights[14] -= n/(float)NUMBER_OF_VOLUMES * gradient[14];
            weights[15] -= n/(float)NUMBER_OF_VOLUMES * gradient[15];
            weights[16] -= n/(float)NUMBER_OF_VOLUMES * gradient[16];
            weights[17] -= n/(float)NUMBER_OF_VOLUMES * gradient[17];
            weights[18] -= n/(float)NUMBER_OF_VOLUMES * gradient[18];
            weights[19] -= n/(float)NUMBER_OF_VOLUMES * gradient[19];
            weights[20] -= n/(float)NUMBER_OF_VOLUMES * gradient[20];
            weights[21] -= n/(float)NUMBER_OF_VOLUMES * gradient[21];
            weights[22] -= n/(float)NUMBER_OF_VOLUMES * gradient[22];
            weights[23] -= n/(float)NUMBER_OF_VOLUMES * gradient[23];
            weights[24] -= n/(float)NUMBER_OF_VOLUMES * gradient[24];
            weights[25] -= n/(float)NUMBER_OF_VOLUMES * gradient[25];
            weights[26] -= n/(float)NUMBER_OF_VOLUMES * gradient[26];
            weights[27] -= n/(float)NUMBER_OF_VOLUMES * gradient[27];
            weights[28] -= n/(float)NUMBER_OF_VOLUMES * gradient[28];
            weights[29] -= n/(float)NUMBER_OF_VOLUMES * gradient[29];
            weights[30] -= n/(float)NUMBER_OF_VOLUMES * gradient[30];
            weights[31] -= n/(float)NUMBER_OF_VOLUMES * gradient[31];
            weights[32] -= n/(float)NUMBER_OF_VOLUMES * gradient[32];
            weights[33] -= n/(float)NUMBER_OF_VOLUMES * gradient[33];
            
            // end for epocs
        }
        
        // Make classification
        float s;
        s =  weights[0] * 1.0f;
        
        //float x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33;
        float xs[34];
        xs[1] = Volumes[Calculate4DIndex(x+0,y+0,z-2,validation,DATA_W,DATA_H,DATA_D)];
        xs[2] = Volumes[Calculate4DIndex(x-1,y-1,z-1,validation,DATA_W,DATA_H,DATA_D)];
        xs[3] = Volumes[Calculate4DIndex(x-1,y+0,z-1,validation,DATA_W,DATA_H,DATA_D)];
        xs[4] = Volumes[Calculate4DIndex(x-1,y+1,z-1,validation,DATA_W,DATA_H,DATA_D)];
        xs[5] = Volumes[Calculate4DIndex(x+0,y-1,z-1,validation,DATA_W,DATA_H,DATA_D)];
        xs[6] = Volumes[Calculate4DIndex(x+0,y+0,z-1,validation,DATA_W,DATA_H,DATA_D)];
        xs[7] = Volumes[Calculate4DIndex(x+0,y+1,z-1,validation,DATA_W,DATA_H,DATA_D)];
        xs[8] = Volumes[Calculate4DIndex(x+1,y-1,z-1,validation,DATA_W,DATA_H,DATA_D)];
        xs[9] = Volumes[Calculate4DIndex(x+1,y+0,z-1,validation,DATA_W,DATA_H,DATA_D)];
        xs[10] = Volumes[Calculate4DIndex(x+1,y+1,z-1,validation,DATA_W,DATA_H,DATA_D)];
        xs[11] = Volumes[Calculate4DIndex(x-2,y+0,z+0,validation,DATA_W,DATA_H,DATA_D)];
        xs[12] = Volumes[Calculate4DIndex(x-1,y-1,z+0,validation,DATA_W,DATA_H,DATA_D)];
        xs[13] = Volumes[Calculate4DIndex(x-1,y+0,z+0,validation,DATA_W,DATA_H,DATA_D)];
        xs[14] = Volumes[Calculate4DIndex(x-1,y+1,z+0,validation,DATA_W,DATA_H,DATA_D)];
        xs[15] = Volumes[Calculate4DIndex(x+0,y-2,z+0,validation,DATA_W,DATA_H,DATA_D)];
        xs[16] = Volumes[Calculate4DIndex(x+0,y-1,z+0,validation,DATA_W,DATA_H,DATA_D)];
        xs[17] = Volumes[Calculate4DIndex(x+0,y+0,z+0,validation,DATA_W,DATA_H,DATA_D)];
        xs[18] = Volumes[Calculate4DIndex(x+0,y+1,z+0,validation,DATA_W,DATA_H,DATA_D)];
        xs[19] = Volumes[Calculate4DIndex(x+0,y+2,z+0,validation,DATA_W,DATA_H,DATA_D)];
        xs[20] = Volumes[Calculate4DIndex(x+1,y-1,z+0,validation,DATA_W,DATA_H,DATA_D)];
        xs[21] = Volumes[Calculate4DIndex(x+1,y+0,z+0,validation,DATA_W,DATA_H,DATA_D)];
        xs[22] = Volumes[Calculate4DIndex(x+1,y+1,z+0,validation,DATA_W,DATA_H,DATA_D)];
        xs[23] = Volumes[Calculate4DIndex(x+2,y+0,z+0,validation,DATA_W,DATA_H,DATA_D)];
        xs[24] = Volumes[Calculate4DIndex(x-1,y-1,z+1,validation,DATA_W,DATA_H,DATA_D)];
        xs[25] = Volumes[Calculate4DIndex(x-1,y+0,z+1,validation,DATA_W,DATA_H,DATA_D)];
        xs[26] = Volumes[Calculate4DIndex(x-1,y+1,z+1,validation,DATA_W,DATA_H,DATA_D)];
        xs[27] = Volumes[Calculate4DIndex(x+0,y-1,z+1,validation,DATA_W,DATA_H,DATA_D)];
        xs[28] = Volumes[Calculate4DIndex(x+0,y+0,z+1,validation,DATA_W,DATA_H,DATA_D)];
        xs[29] = Volumes[Calculate4DIndex(x+0,y+1,z+1,validation,DATA_W,DATA_H,DATA_D)];
        xs[30] = Volumes[Calculate4DIndex(x+1,y-1,z+1,validation,DATA_W,DATA_H,DATA_D)];
        xs[31] = Volumes[Calculate4DIndex(x+1,y+0,z+1,validation,DATA_W,DATA_H,DATA_D)];
        xs[32] = Volumes[Calculate4DIndex(x+1,y+1,z+1,validation,DATA_W,DATA_H,DATA_D)];
        xs[33] = Volumes[Calculate4DIndex(x+0,y+0,z+2,validation,DATA_W,DATA_H,DATA_D)];
        
        if (x==9 && y==22 && z==26)
                {
                    printf("classifying %i\n", validation);
                    printf("1:%f 2:%f 3:%f 4:%f 5:%f 6:%f 7:%f 8:%f 9:%f 10:%f 11:%f 12:%f 13:%f 14:%f 15:%f 16:%f 17:%f 18:%f 19:%f 20:%f 21:%f 22:%f 23:%f 24:%f 25:%f 26:%f 27:%f 28:%f 29:%f 30:%f 31:%f 32:%f 33:%f \n " ,xs[1], xs[2], xs[3], xs[4], xs[5], xs[6], xs[7], xs[8], xs[9], xs[10], xs[11], xs[12], xs[13], xs[14], xs[15], xs[16], xs[17], xs[18], xs[19], xs[20], xs[21], xs[22], xs[23], xs[24], xs[25], xs[26], xs[27], xs[28], xs[29], xs[30], xs[31], xs[32], xs[33]);
                }

        // weights sum
        s += weights[1] * xs[1];
        s += weights[2] * xs[2];
        s += weights[3] * xs[3];
        s += weights[4] * xs[4];
        s += weights[5] * xs[5];
        s += weights[6] * xs[6];
        s += weights[7] * xs[7];
        s += weights[8] * xs[8];
        s += weights[9] * xs[9];
        s += weights[10] * xs[10];
        s += weights[11] * xs[11];
        s += weights[12] * xs[12];
        s += weights[13] * xs[13];
        s += weights[14] * xs[14];
        s += weights[15] * xs[15];
        s += weights[16] * xs[16];
        s += weights[17] * xs[17];
        s += weights[18] * xs[18];
        s += weights[19] * xs[19];
        s += weights[20] * xs[20];
        s += weights[21] * xs[21];
        s += weights[22] * xs[22];
        s += weights[23] * xs[23];
        s += weights[24] * xs[24];
        s += weights[25] * xs[25];
        s += weights[26] * xs[26];
        s += weights[27] * xs[27];
        s += weights[28] * xs[28];
        s += weights[29] * xs[29];
        s += weights[30] * xs[30];
        s += weights[31] * xs[31];
        s += weights[32] * xs[32];
        s += weights[33] * xs[33];
        
        float classification;
        if (s > 0.0f)
        {
            classification = 0.0f;
        }
        else
        {
            classification = 1.0f;
        }
        
        if (classification == c_Correct_Classes[validation])
        {
            classification_performance++;
        }
        
        // end for validation
    }
    
    Classifier_Performance[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = (float)classification_performance / (float)uncensoredVolumes;
}

// sphere radius of 3 voxels resulting in 123 voxels
__kernel void CalculateStatisticalMapSearchlight(__global float* Classifier_Performance,
                                                     __global const float* Volumes,
                                                     __global const float* Mask,
                                                     __constant float* c_d,
                                                     __constant float* c_Correct_Classes,
                                                     __private int DATA_W,
                                                     __private int DATA_H,
                                                     __private int DATA_D,
                                                     __private int NUMBER_OF_VOLUMES,
                                                     __private float n,
                                                     __private int EPOCS,
                                                     __private int validation)

{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    
    if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
        return;
    
    if ( Mask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] != 1.0f )
    {
        Classifier_Performance[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = 0.0f;
        return;
    }
    
    if ( ((x + 3) >= DATA_W) || ((y + 3) >= DATA_H) || ((z + 3) >= DATA_D) )
    {
        Classifier_Performance[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = 0.0f;
        return;
    }
    
    
    if ( ((x - 3) < 0) || ((y - 3) < 0) || ((z - 3) < 0) )
    {
        Classifier_Performance[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = 0.0f;
        return;
    }

    int classification_performance = 0;
    
    float weights[124]; // we use 123 voxels
    
    int uncensoredVolumes = 0;
    
    //bool printed = false;
    
    // Leave one out cross validation
    //for (int validation = 0; validation < NUMBER_OF_VOLUMES; validation++)

    {
        // Skip testing with censored volumes
        if (c_Correct_Classes[validation] == 9999.0f)
        {
            //continue;
        }
        
        uncensoredVolumes++;
        
        weights[0] = 0.0f;
        weights[1] = 0.0f;
        weights[2] = 0.0f;
        weights[3] = 0.0f;
        weights[4] = 0.0f;
        weights[5] = 0.0f;
        weights[6] = 0.0f;
        weights[7] = 0.0f;
        weights[8] = 0.0f;
        weights[9] = 0.0f;
        weights[10] = 0.0f;
        weights[11] = 0.0f;
        weights[12] = 0.0f;
        weights[13] = 0.0f;
        weights[14] = 0.0f;
        weights[15] = 0.0f;
        weights[16] = 0.0f;
        weights[17] = 0.0f;
        weights[18] = 0.0f;
        weights[19] = 0.0f;
        weights[20] = 0.0f;
        weights[21] = 0.0f;
        weights[22] = 0.0f;
        weights[23] = 0.0f;
        weights[24] = 0.0f;
        weights[25] = 0.0f;
        weights[26] = 0.0f;
        weights[27] = 0.0f;
        weights[28] = 0.0f;
        weights[29] = 0.0f;
        weights[30] = 0.0f;
        weights[31] = 0.0f;
        weights[32] = 0.0f;
        weights[33] = 0.0f;
        weights[34] = 0.0f;
        weights[35] = 0.0f;
        weights[36] = 0.0f;
        weights[37] = 0.0f;
        weights[38] = 0.0f;
        weights[39] = 0.0f;
        weights[40] = 0.0f;
        weights[41] = 0.0f;
        weights[42] = 0.0f;
        weights[43] = 0.0f;
        weights[44] = 0.0f;
        weights[45] = 0.0f;
        weights[46] = 0.0f;
        weights[47] = 0.0f;
        weights[48] = 0.0f;
        weights[49] = 0.0f;
        weights[50] = 0.0f;
        weights[51] = 0.0f;
        weights[52] = 0.0f;
        weights[53] = 0.0f;
        weights[54] = 0.0f;
        weights[55] = 0.0f;
        weights[56] = 0.0f;
        weights[57] = 0.0f;
        weights[58] = 0.0f;
        weights[59] = 0.0f;
        weights[60] = 0.0f;
        weights[61] = 0.0f;
        weights[62] = 0.0f;
        weights[63] = 0.0f;
        weights[64] = 0.0f;
        weights[65] = 0.0f;
        weights[66] = 0.0f;
        weights[67] = 0.0f;
        weights[68] = 0.0f;
        weights[69] = 0.0f;
        weights[70] = 0.0f;
        weights[71] = 0.0f;
        weights[72] = 0.0f;
        weights[73] = 0.0f;
        weights[74] = 0.0f;
        weights[75] = 0.0f;
        weights[76] = 0.0f;
        weights[77] = 0.0f;
        weights[78] = 0.0f;
        weights[79] = 0.0f;
        weights[80] = 0.0f;
        weights[81] = 0.0f;
        weights[82] = 0.0f;
        weights[83] = 0.0f;
        weights[84] = 0.0f;
        weights[85] = 0.0f;
        weights[86] = 0.0f;
        weights[87] = 0.0f;
        weights[88] = 0.0f;
        weights[89] = 0.0f;
        weights[90] = 0.0f;
        weights[91] = 0.0f;
        weights[92] = 0.0f;
        weights[93] = 0.0f;
        weights[94] = 0.0f;
        weights[95] = 0.0f;
        weights[96] = 0.0f;
        weights[97] = 0.0f;
        weights[98] = 0.0f;
        weights[99] = 0.0f;
        weights[100] = 0.0f;
        weights[101] = 0.0f;
        weights[102] = 0.0f;
        weights[103] = 0.0f;
        weights[104] = 0.0f;
        weights[105] = 0.0f;
        weights[106] = 0.0f;
        weights[107] = 0.0f;
        weights[108] = 0.0f;
        weights[109] = 0.0f;
        weights[110] = 0.0f;
        weights[111] = 0.0f;
        weights[112] = 0.0f;
        weights[113] = 0.0f;
        weights[114] = 0.0f;
        weights[115] = 0.0f;
        weights[116] = 0.0f;
        weights[117] = 0.0f;
        weights[118] = 0.0f;
        weights[119] = 0.0f;
        weights[120] = 0.0f;
        weights[121] = 0.0f;
        weights[122] = 0.0f;
        weights[123] = 0.0f;
        
        
        // Do training for a number of iterations
        for (int epoc = 0; epoc < EPOCS; epoc++)
        {
            float gradient[124];
            
            gradient[0] = 0.0f;
            gradient[1] = 0.0f;
            gradient[2] = 0.0f;
            gradient[3] = 0.0f;
            gradient[4] = 0.0f;
            gradient[5] = 0.0f;
            gradient[6] = 0.0f;
            gradient[7] = 0.0f;
            gradient[8] = 0.0f;
            gradient[9] = 0.0f;
            gradient[10] = 0.0f;
            gradient[11] = 0.0f;
            gradient[12] = 0.0f;
            gradient[13] = 0.0f;
            gradient[14] = 0.0f;
            gradient[15] = 0.0f;
            gradient[16] = 0.0f;
            gradient[17] = 0.0f;
            gradient[18] = 0.0f;
            gradient[19] = 0.0f;
            gradient[20] = 0.0f;
            gradient[21] = 0.0f;
            gradient[22] = 0.0f;
            gradient[23] = 0.0f;
            gradient[24] = 0.0f;
            gradient[25] = 0.0f;
            gradient[26] = 0.0f;
            gradient[27] = 0.0f;
            gradient[28] = 0.0f;
            gradient[29] = 0.0f;
            gradient[30] = 0.0f;
            gradient[31] = 0.0f;
            gradient[32] = 0.0f;
            gradient[33] = 0.0f;
            gradient[34] = 0.0f;
            gradient[35] = 0.0f;
            gradient[36] = 0.0f;
            gradient[37] = 0.0f;
            gradient[38] = 0.0f;
            gradient[39] = 0.0f;
            gradient[40] = 0.0f;
            gradient[41] = 0.0f;
            gradient[42] = 0.0f;
            gradient[43] = 0.0f;
            gradient[44] = 0.0f;
            gradient[45] = 0.0f;
            gradient[46] = 0.0f;
            gradient[47] = 0.0f;
            gradient[48] = 0.0f;
            gradient[49] = 0.0f;
            gradient[50] = 0.0f;
            gradient[51] = 0.0f;
            gradient[52] = 0.0f;
            gradient[53] = 0.0f;
            gradient[54] = 0.0f;
            gradient[55] = 0.0f;
            gradient[56] = 0.0f;
            gradient[57] = 0.0f;
            gradient[58] = 0.0f;
            gradient[59] = 0.0f;
            gradient[60] = 0.0f;
            gradient[61] = 0.0f;
            gradient[62] = 0.0f;
            gradient[63] = 0.0f;
            gradient[64] = 0.0f;
            gradient[65] = 0.0f;
            gradient[66] = 0.0f;
            gradient[67] = 0.0f;
            gradient[68] = 0.0f;
            gradient[69] = 0.0f;
            gradient[70] = 0.0f;
            gradient[71] = 0.0f;
            gradient[72] = 0.0f;
            gradient[73] = 0.0f;
            gradient[74] = 0.0f;
            gradient[75] = 0.0f;
            gradient[76] = 0.0f;
            gradient[77] = 0.0f;
            gradient[78] = 0.0f;
            gradient[79] = 0.0f;
            gradient[80] = 0.0f;
            gradient[81] = 0.0f;
            gradient[82] = 0.0f;
            gradient[83] = 0.0f;
            gradient[84] = 0.0f;
            gradient[85] = 0.0f;
            gradient[86] = 0.0f;
            gradient[87] = 0.0f;
            gradient[88] = 0.0f;
            gradient[89] = 0.0f;
            gradient[90] = 0.0f;
            gradient[91] = 0.0f;
            gradient[92] = 0.0f;
            gradient[93] = 0.0f;
            gradient[94] = 0.0f;
            gradient[95] = 0.0f;
            gradient[96] = 0.0f;
            gradient[97] = 0.0f;
            gradient[98] = 0.0f;
            gradient[99] = 0.0f;
            gradient[100] = 0.0f;
            gradient[101] = 0.0f;
            gradient[102] = 0.0f;
            gradient[103] = 0.0f;
            gradient[104] = 0.0f;
            gradient[105] = 0.0f;
            gradient[106] = 0.0f;
            gradient[107] = 0.0f;
            gradient[108] = 0.0f;
            gradient[109] = 0.0f;
            gradient[110] = 0.0f;
            gradient[111] = 0.0f;
            gradient[112] = 0.0f;
            gradient[113] = 0.0f;
            gradient[114] = 0.0f;
            gradient[115] = 0.0f;
            gradient[116] = 0.0f;
            gradient[117] = 0.0f;
            gradient[118] = 0.0f;
            gradient[119] = 0.0f;
            gradient[120] = 0.0f;
            gradient[121] = 0.0f;
            gradient[122] = 0.0f;
            gradient[123] = 0.0f;
            
            for (int t = 0; t < NUMBER_OF_VOLUMES; t++)
            {
                
                // Skip training with validation volume and censored volumes
                if ((t == validation) || (c_Correct_Classes[t] == 9999.0f))
                {
                    continue;
                }
                
                // Make classification
                float s;
                s =  weights[0] * 1.0f;
                
                float x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38, x39, x40, x41, x42, x43, x44, x45, x46, x47, x48, x49, x50, x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, x61, x62, x63, x64, x65, x66, x67, x68, x69, x70, x71, x72, x73, x74, x75, x76, x77, x78, x79, x80, x81, x82, x83, x84, x85, x86, x87, x88, x89, x90, x91, x92, x93, x94, x95, x96, x97, x98, x99, x100, x101, x102, x103, x104, x105, x106, x107, x108, x109, x110, x111, x112, x113, x114, x115, x116, x117, x118, x119, x120, x121, x122, x123;
                
                x1 = Volumes[Calculate4DIndex(x+0,y+0,z-3,t,DATA_W,DATA_H,DATA_D)];
                x2 = Volumes[Calculate4DIndex(x-2,y-1,z-2,t,DATA_W,DATA_H,DATA_D)];
                x3 = Volumes[Calculate4DIndex(x-2,y+0,z-2,t,DATA_W,DATA_H,DATA_D)];
                x4 = Volumes[Calculate4DIndex(x-2,y+1,z-2,t,DATA_W,DATA_H,DATA_D)];
                x5 = Volumes[Calculate4DIndex(x-1,y-2,z-2,t,DATA_W,DATA_H,DATA_D)];
                x6 = Volumes[Calculate4DIndex(x-1,y-1,z-2,t,DATA_W,DATA_H,DATA_D)];
                x7 = Volumes[Calculate4DIndex(x-1,y+0,z-2,t,DATA_W,DATA_H,DATA_D)];
                x8 = Volumes[Calculate4DIndex(x-1,y+1,z-2,t,DATA_W,DATA_H,DATA_D)];
                x9 = Volumes[Calculate4DIndex(x-1,y+2,z-2,t,DATA_W,DATA_H,DATA_D)];
                x10 = Volumes[Calculate4DIndex(x+0,y-2,z-2,t,DATA_W,DATA_H,DATA_D)];
                x11 = Volumes[Calculate4DIndex(x+0,y-1,z-2,t,DATA_W,DATA_H,DATA_D)];
                x12 = Volumes[Calculate4DIndex(x+0,y+0,z-2,t,DATA_W,DATA_H,DATA_D)];
                x13 = Volumes[Calculate4DIndex(x+0,y+1,z-2,t,DATA_W,DATA_H,DATA_D)];
                x14 = Volumes[Calculate4DIndex(x+0,y+2,z-2,t,DATA_W,DATA_H,DATA_D)];
                x15 = Volumes[Calculate4DIndex(x+1,y-2,z-2,t,DATA_W,DATA_H,DATA_D)];
                x16 = Volumes[Calculate4DIndex(x+1,y-1,z-2,t,DATA_W,DATA_H,DATA_D)];
                x17 = Volumes[Calculate4DIndex(x+1,y+0,z-2,t,DATA_W,DATA_H,DATA_D)];
                x18 = Volumes[Calculate4DIndex(x+1,y+1,z-2,t,DATA_W,DATA_H,DATA_D)];
                x19 = Volumes[Calculate4DIndex(x+1,y+2,z-2,t,DATA_W,DATA_H,DATA_D)];
                x20 = Volumes[Calculate4DIndex(x+2,y-1,z-2,t,DATA_W,DATA_H,DATA_D)];
                x21 = Volumes[Calculate4DIndex(x+2,y+0,z-2,t,DATA_W,DATA_H,DATA_D)];
                x22 = Volumes[Calculate4DIndex(x+2,y+1,z-2,t,DATA_W,DATA_H,DATA_D)];
                x23 = Volumes[Calculate4DIndex(x-2,y-2,z-1,t,DATA_W,DATA_H,DATA_D)];
                x24 = Volumes[Calculate4DIndex(x-2,y-1,z-1,t,DATA_W,DATA_H,DATA_D)];
                x25 = Volumes[Calculate4DIndex(x-2,y+0,z-1,t,DATA_W,DATA_H,DATA_D)];
                x26 = Volumes[Calculate4DIndex(x-2,y+1,z-1,t,DATA_W,DATA_H,DATA_D)];
                x27 = Volumes[Calculate4DIndex(x-2,y+2,z-1,t,DATA_W,DATA_H,DATA_D)];
                x28 = Volumes[Calculate4DIndex(x-1,y-2,z-1,t,DATA_W,DATA_H,DATA_D)];
                x29 = Volumes[Calculate4DIndex(x-1,y-1,z-1,t,DATA_W,DATA_H,DATA_D)];
                x30 = Volumes[Calculate4DIndex(x-1,y+0,z-1,t,DATA_W,DATA_H,DATA_D)];
                x31 = Volumes[Calculate4DIndex(x-1,y+1,z-1,t,DATA_W,DATA_H,DATA_D)];
                x32 = Volumes[Calculate4DIndex(x-1,y+2,z-1,t,DATA_W,DATA_H,DATA_D)];
                x33 = Volumes[Calculate4DIndex(x+0,y-2,z-1,t,DATA_W,DATA_H,DATA_D)];
                x34 = Volumes[Calculate4DIndex(x+0,y-1,z-1,t,DATA_W,DATA_H,DATA_D)];
                x35 = Volumes[Calculate4DIndex(x+0,y+0,z-1,t,DATA_W,DATA_H,DATA_D)];
                x36 = Volumes[Calculate4DIndex(x+0,y+1,z-1,t,DATA_W,DATA_H,DATA_D)];
                x37 = Volumes[Calculate4DIndex(x+0,y+2,z-1,t,DATA_W,DATA_H,DATA_D)];
                x38 = Volumes[Calculate4DIndex(x+1,y-2,z-1,t,DATA_W,DATA_H,DATA_D)];
                x39 = Volumes[Calculate4DIndex(x+1,y-1,z-1,t,DATA_W,DATA_H,DATA_D)];
                x40 = Volumes[Calculate4DIndex(x+1,y+0,z-1,t,DATA_W,DATA_H,DATA_D)];
                x41 = Volumes[Calculate4DIndex(x+1,y+1,z-1,t,DATA_W,DATA_H,DATA_D)];
                x42 = Volumes[Calculate4DIndex(x+1,y+2,z-1,t,DATA_W,DATA_H,DATA_D)];
                x43 = Volumes[Calculate4DIndex(x+2,y-2,z-1,t,DATA_W,DATA_H,DATA_D)];
                x44 = Volumes[Calculate4DIndex(x+2,y-1,z-1,t,DATA_W,DATA_H,DATA_D)];
                x45 = Volumes[Calculate4DIndex(x+2,y+0,z-1,t,DATA_W,DATA_H,DATA_D)];
                x46 = Volumes[Calculate4DIndex(x+2,y+1,z-1,t,DATA_W,DATA_H,DATA_D)];
                x47 = Volumes[Calculate4DIndex(x+2,y+2,z-1,t,DATA_W,DATA_H,DATA_D)];
                x48 = Volumes[Calculate4DIndex(x-3,y+0,z+0,t,DATA_W,DATA_H,DATA_D)];
                x49 = Volumes[Calculate4DIndex(x-2,y-2,z+0,t,DATA_W,DATA_H,DATA_D)];
                x50 = Volumes[Calculate4DIndex(x-2,y-1,z+0,t,DATA_W,DATA_H,DATA_D)];
                x51 = Volumes[Calculate4DIndex(x-2,y+0,z+0,t,DATA_W,DATA_H,DATA_D)];
                x52 = Volumes[Calculate4DIndex(x-2,y+1,z+0,t,DATA_W,DATA_H,DATA_D)];
                x53 = Volumes[Calculate4DIndex(x-2,y+2,z+0,t,DATA_W,DATA_H,DATA_D)];
                x54 = Volumes[Calculate4DIndex(x-1,y-2,z+0,t,DATA_W,DATA_H,DATA_D)];
                x55 = Volumes[Calculate4DIndex(x-1,y-1,z+0,t,DATA_W,DATA_H,DATA_D)];
                x56 = Volumes[Calculate4DIndex(x-1,y+0,z+0,t,DATA_W,DATA_H,DATA_D)];
                x57 = Volumes[Calculate4DIndex(x-1,y+1,z+0,t,DATA_W,DATA_H,DATA_D)];
                x58 = Volumes[Calculate4DIndex(x-1,y+2,z+0,t,DATA_W,DATA_H,DATA_D)];
                x59 = Volumes[Calculate4DIndex(x+0,y-3,z+0,t,DATA_W,DATA_H,DATA_D)];
                x60 = Volumes[Calculate4DIndex(x+0,y-2,z+0,t,DATA_W,DATA_H,DATA_D)];
                x61 = Volumes[Calculate4DIndex(x+0,y-1,z+0,t,DATA_W,DATA_H,DATA_D)];
                x62 = Volumes[Calculate4DIndex(x+0,y+0,z+0,t,DATA_W,DATA_H,DATA_D)];
                x63 = Volumes[Calculate4DIndex(x+0,y+1,z+0,t,DATA_W,DATA_H,DATA_D)];
                x64 = Volumes[Calculate4DIndex(x+0,y+2,z+0,t,DATA_W,DATA_H,DATA_D)];
                x65 = Volumes[Calculate4DIndex(x+0,y+3,z+0,t,DATA_W,DATA_H,DATA_D)];
                x66 = Volumes[Calculate4DIndex(x+1,y-2,z+0,t,DATA_W,DATA_H,DATA_D)];
                x67 = Volumes[Calculate4DIndex(x+1,y-1,z+0,t,DATA_W,DATA_H,DATA_D)];
                x68 = Volumes[Calculate4DIndex(x+1,y+0,z+0,t,DATA_W,DATA_H,DATA_D)];
                x69 = Volumes[Calculate4DIndex(x+1,y+1,z+0,t,DATA_W,DATA_H,DATA_D)];
                x70 = Volumes[Calculate4DIndex(x+1,y+2,z+0,t,DATA_W,DATA_H,DATA_D)];
                x71 = Volumes[Calculate4DIndex(x+2,y-2,z+0,t,DATA_W,DATA_H,DATA_D)];
                x72 = Volumes[Calculate4DIndex(x+2,y-1,z+0,t,DATA_W,DATA_H,DATA_D)];
                x73 = Volumes[Calculate4DIndex(x+2,y+0,z+0,t,DATA_W,DATA_H,DATA_D)];
                x74 = Volumes[Calculate4DIndex(x+2,y+1,z+0,t,DATA_W,DATA_H,DATA_D)];
                x75 = Volumes[Calculate4DIndex(x+2,y+2,z+0,t,DATA_W,DATA_H,DATA_D)];
                x76 = Volumes[Calculate4DIndex(x+3,y+0,z+0,t,DATA_W,DATA_H,DATA_D)];
                x77 = Volumes[Calculate4DIndex(x-2,y-2,z+1,t,DATA_W,DATA_H,DATA_D)];
                x78 = Volumes[Calculate4DIndex(x-2,y-1,z+1,t,DATA_W,DATA_H,DATA_D)];
                x79 = Volumes[Calculate4DIndex(x-2,y+0,z+1,t,DATA_W,DATA_H,DATA_D)];
                x80 = Volumes[Calculate4DIndex(x-2,y+1,z+1,t,DATA_W,DATA_H,DATA_D)];
                x81 = Volumes[Calculate4DIndex(x-2,y+2,z+1,t,DATA_W,DATA_H,DATA_D)];
                x82 = Volumes[Calculate4DIndex(x-1,y-2,z+1,t,DATA_W,DATA_H,DATA_D)];
                x83 = Volumes[Calculate4DIndex(x-1,y-1,z+1,t,DATA_W,DATA_H,DATA_D)];
                x84 = Volumes[Calculate4DIndex(x-1,y+0,z+1,t,DATA_W,DATA_H,DATA_D)];
                x85 = Volumes[Calculate4DIndex(x-1,y+1,z+1,t,DATA_W,DATA_H,DATA_D)];
                x86 = Volumes[Calculate4DIndex(x-1,y+2,z+1,t,DATA_W,DATA_H,DATA_D)];
                x87 = Volumes[Calculate4DIndex(x+0,y-2,z+1,t,DATA_W,DATA_H,DATA_D)];
                x88 = Volumes[Calculate4DIndex(x+0,y-1,z+1,t,DATA_W,DATA_H,DATA_D)];
                x89 = Volumes[Calculate4DIndex(x+0,y+0,z+1,t,DATA_W,DATA_H,DATA_D)];
                x90 = Volumes[Calculate4DIndex(x+0,y+1,z+1,t,DATA_W,DATA_H,DATA_D)];
                x91 = Volumes[Calculate4DIndex(x+0,y+2,z+1,t,DATA_W,DATA_H,DATA_D)];
                x92 = Volumes[Calculate4DIndex(x+1,y-2,z+1,t,DATA_W,DATA_H,DATA_D)];
                x93 = Volumes[Calculate4DIndex(x+1,y-1,z+1,t,DATA_W,DATA_H,DATA_D)];
                x94 = Volumes[Calculate4DIndex(x+1,y+0,z+1,t,DATA_W,DATA_H,DATA_D)];
                x95 = Volumes[Calculate4DIndex(x+1,y+1,z+1,t,DATA_W,DATA_H,DATA_D)];
                x96 = Volumes[Calculate4DIndex(x+1,y+2,z+1,t,DATA_W,DATA_H,DATA_D)];
                x97 = Volumes[Calculate4DIndex(x+2,y-2,z+1,t,DATA_W,DATA_H,DATA_D)];
                x98 = Volumes[Calculate4DIndex(x+2,y-1,z+1,t,DATA_W,DATA_H,DATA_D)];
                x99 = Volumes[Calculate4DIndex(x+2,y+0,z+1,t,DATA_W,DATA_H,DATA_D)];
                x100 = Volumes[Calculate4DIndex(x+2,y+1,z+1,t,DATA_W,DATA_H,DATA_D)];
                x101 = Volumes[Calculate4DIndex(x+2,y+2,z+1,t,DATA_W,DATA_H,DATA_D)];
                x102 = Volumes[Calculate4DIndex(x-2,y-1,z+2,t,DATA_W,DATA_H,DATA_D)];
                x103 = Volumes[Calculate4DIndex(x-2,y+0,z+2,t,DATA_W,DATA_H,DATA_D)];
                x104 = Volumes[Calculate4DIndex(x-2,y+1,z+2,t,DATA_W,DATA_H,DATA_D)];
                x105 = Volumes[Calculate4DIndex(x-1,y-2,z+2,t,DATA_W,DATA_H,DATA_D)];
                x106 = Volumes[Calculate4DIndex(x-1,y-1,z+2,t,DATA_W,DATA_H,DATA_D)];
                x107 = Volumes[Calculate4DIndex(x-1,y+0,z+2,t,DATA_W,DATA_H,DATA_D)];
                x108 = Volumes[Calculate4DIndex(x-1,y+1,z+2,t,DATA_W,DATA_H,DATA_D)];
                x109 = Volumes[Calculate4DIndex(x-1,y+2,z+2,t,DATA_W,DATA_H,DATA_D)];
                x110 = Volumes[Calculate4DIndex(x+0,y-2,z+2,t,DATA_W,DATA_H,DATA_D)];
                x111 = Volumes[Calculate4DIndex(x+0,y-1,z+2,t,DATA_W,DATA_H,DATA_D)];
                x112 = Volumes[Calculate4DIndex(x+0,y+0,z+2,t,DATA_W,DATA_H,DATA_D)];
                x113 = Volumes[Calculate4DIndex(x+0,y+1,z+2,t,DATA_W,DATA_H,DATA_D)];
                x114 = Volumes[Calculate4DIndex(x+0,y+2,z+2,t,DATA_W,DATA_H,DATA_D)];
                x115 = Volumes[Calculate4DIndex(x+1,y-2,z+2,t,DATA_W,DATA_H,DATA_D)];
                x116 = Volumes[Calculate4DIndex(x+1,y-1,z+2,t,DATA_W,DATA_H,DATA_D)];
                x117 = Volumes[Calculate4DIndex(x+1,y+0,z+2,t,DATA_W,DATA_H,DATA_D)];
                x118 = Volumes[Calculate4DIndex(x+1,y+1,z+2,t,DATA_W,DATA_H,DATA_D)];
                x119 = Volumes[Calculate4DIndex(x+1,y+2,z+2,t,DATA_W,DATA_H,DATA_D)];
                x120 = Volumes[Calculate4DIndex(x+2,y-1,z+2,t,DATA_W,DATA_H,DATA_D)];
                x121 = Volumes[Calculate4DIndex(x+2,y+0,z+2,t,DATA_W,DATA_H,DATA_D)];
                x122 = Volumes[Calculate4DIndex(x+2,y+1,z+2,t,DATA_W,DATA_H,DATA_D)];
                x123 = Volumes[Calculate4DIndex(x+0,y+0,z+3,t,DATA_W,DATA_H,DATA_D)];
                
                
                // weights sum
                s += weights[1] * x1;
                s += weights[2] * x2;
                s += weights[3] * x3;
                s += weights[4] * x4;
                s += weights[5] * x5;
                s += weights[6] * x6;
                s += weights[7] * x7;
                s += weights[8] * x8;
                s += weights[9] * x9;
                s += weights[10] * x10;
                s += weights[11] * x11;
                s += weights[12] * x12;
                s += weights[13] * x13;
                s += weights[14] * x14;
                s += weights[15] * x15;
                s += weights[16] * x16;
                s += weights[17] * x17;
                s += weights[18] * x18;
                s += weights[19] * x19;
                s += weights[20] * x20;
                s += weights[21] * x21;
                s += weights[22] * x22;
                s += weights[23] * x23;
                s += weights[24] * x24;
                s += weights[25] * x25;
                s += weights[26] * x26;
                s += weights[27] * x27;
                s += weights[28] * x28;
                s += weights[29] * x29;
                s += weights[30] * x30;
                s += weights[31] * x31;
                s += weights[32] * x32;
                s += weights[33] * x33;
                s += weights[34] * x34;
                s += weights[35] * x35;
                s += weights[36] * x36;
                s += weights[37] * x37;
                s += weights[38] * x38;
                s += weights[39] * x39;
                s += weights[40] * x40;
                s += weights[41] * x41;
                s += weights[42] * x42;
                s += weights[43] * x43;
                s += weights[44] * x44;
                s += weights[45] * x45;
                s += weights[46] * x46;
                s += weights[47] * x47;
                s += weights[48] * x48;
                s += weights[49] * x49;
                s += weights[50] * x50;
                s += weights[51] * x51;
                s += weights[52] * x52;
                s += weights[53] * x53;
                s += weights[54] * x54;
                s += weights[55] * x55;
                s += weights[56] * x56;
                s += weights[57] * x57;
                s += weights[58] * x58;
                s += weights[59] * x59;
                s += weights[60] * x60;
                s += weights[61] * x61;
                s += weights[62] * x62;
                s += weights[63] * x63;
                s += weights[64] * x64;
                s += weights[65] * x65;
                s += weights[66] * x66;
                s += weights[67] * x67;
                s += weights[68] * x68;
                s += weights[69] * x69;
                s += weights[70] * x70;
                s += weights[71] * x71;
                s += weights[72] * x72;
                s += weights[73] * x73;
                s += weights[74] * x74;
                s += weights[75] * x75;
                s += weights[76] * x76;
                s += weights[77] * x77;
                s += weights[78] * x78;
                s += weights[79] * x79;
                s += weights[80] * x80;
                s += weights[81] * x81;
                s += weights[82] * x82;
                s += weights[83] * x83;
                s += weights[84] * x84;
                s += weights[85] * x85;
                s += weights[86] * x86;
                s += weights[87] * x87;
                s += weights[88] * x88;
                s += weights[89] * x89;
                s += weights[90] * x90;
                s += weights[91] * x91;
                s += weights[92] * x92;
                s += weights[93] * x93;
                s += weights[94] * x94;
                s += weights[95] * x95;
                s += weights[96] * x96;
                s += weights[97] * x97;
                s += weights[98] * x98;
                s += weights[99] * x99;
                s += weights[100] * x100;
                s += weights[101] * x101;
                s += weights[102] * x102;
                s += weights[103] * x103;
                s += weights[104] * x104;
                s += weights[105] * x105;
                s += weights[106] * x106;
                s += weights[107] * x107;
                s += weights[108] * x108;
                s += weights[109] * x109;
                s += weights[110] * x110;
                s += weights[111] * x111;
                s += weights[112] * x112;
                s += weights[113] * x113;
                s += weights[114] * x114;
                s += weights[115] * x115;
                s += weights[116] * x116;
                s += weights[117] * x117;
                s += weights[118] * x118;
                s += weights[119] * x119;
                s += weights[120] * x120;
                s += weights[121] * x121;
                s += weights[122] * x122;
                s += weights[123] * x123;
                
                // Calculate contribution to gradient
                gradient[0] += (s - c_d[t]) * 1.0f;
                
                gradient[1] += (s - c_d[t]) * x1;
                gradient[2] += (s - c_d[t]) * x2;
                gradient[3] += (s - c_d[t]) * x3;
                gradient[4] += (s - c_d[t]) * x4;
                gradient[5] += (s - c_d[t]) * x5;
                gradient[6] += (s - c_d[t]) * x6;
                gradient[7] += (s - c_d[t]) * x7;
                gradient[8] += (s - c_d[t]) * x8;
                gradient[9] += (s - c_d[t]) * x9;
                gradient[10] += (s - c_d[t]) * x10;
                gradient[11] += (s - c_d[t]) * x11;
                gradient[12] += (s - c_d[t]) * x12;
                gradient[13] += (s - c_d[t]) * x13;
                gradient[14] += (s - c_d[t]) * x14;
                gradient[15] += (s - c_d[t]) * x15;
                gradient[16] += (s - c_d[t]) * x16;
                gradient[17] += (s - c_d[t]) * x17;
                gradient[18] += (s - c_d[t]) * x18;
                gradient[19] += (s - c_d[t]) * x19;
                gradient[20] += (s - c_d[t]) * x20;
                gradient[21] += (s - c_d[t]) * x21;
                gradient[22] += (s - c_d[t]) * x22;
                gradient[23] += (s - c_d[t]) * x23;
                gradient[24] += (s - c_d[t]) * x24;
                gradient[25] += (s - c_d[t]) * x25;
                gradient[26] += (s - c_d[t]) * x26;
                gradient[27] += (s - c_d[t]) * x27;
                gradient[28] += (s - c_d[t]) * x28;
                gradient[29] += (s - c_d[t]) * x29;
                gradient[30] += (s - c_d[t]) * x30;
                gradient[31] += (s - c_d[t]) * x31;
                gradient[32] += (s - c_d[t]) * x32;
                gradient[33] += (s - c_d[t]) * x33;
                gradient[34] += (s - c_d[t]) * x34;
                gradient[35] += (s - c_d[t]) * x35;
                gradient[36] += (s - c_d[t]) * x36;
                gradient[37] += (s - c_d[t]) * x37;
                gradient[38] += (s - c_d[t]) * x38;
                gradient[39] += (s - c_d[t]) * x39;
                gradient[40] += (s - c_d[t]) * x40;
                gradient[41] += (s - c_d[t]) * x41;
                gradient[42] += (s - c_d[t]) * x42;
                gradient[43] += (s - c_d[t]) * x43;
                gradient[44] += (s - c_d[t]) * x44;
                gradient[45] += (s - c_d[t]) * x45;
                gradient[46] += (s - c_d[t]) * x46;
                gradient[47] += (s - c_d[t]) * x47;
                gradient[48] += (s - c_d[t]) * x48;
                gradient[49] += (s - c_d[t]) * x49;
                gradient[50] += (s - c_d[t]) * x50;
                gradient[51] += (s - c_d[t]) * x51;
                gradient[52] += (s - c_d[t]) * x52;
                gradient[53] += (s - c_d[t]) * x53;
                gradient[54] += (s - c_d[t]) * x54;
                gradient[55] += (s - c_d[t]) * x55;
                gradient[56] += (s - c_d[t]) * x56;
                gradient[57] += (s - c_d[t]) * x57;
                gradient[58] += (s - c_d[t]) * x58;
                gradient[59] += (s - c_d[t]) * x59;
                gradient[60] += (s - c_d[t]) * x60;
                gradient[61] += (s - c_d[t]) * x61;
                gradient[62] += (s - c_d[t]) * x62;
                gradient[63] += (s - c_d[t]) * x63;
                gradient[64] += (s - c_d[t]) * x64;
                gradient[65] += (s - c_d[t]) * x65;
                gradient[66] += (s - c_d[t]) * x66;
                gradient[67] += (s - c_d[t]) * x67;
                gradient[68] += (s - c_d[t]) * x68;
                gradient[69] += (s - c_d[t]) * x69;
                gradient[70] += (s - c_d[t]) * x70;
                gradient[71] += (s - c_d[t]) * x71;
                gradient[72] += (s - c_d[t]) * x72;
                gradient[73] += (s - c_d[t]) * x73;
                gradient[74] += (s - c_d[t]) * x74;
                gradient[75] += (s - c_d[t]) * x75;
                gradient[76] += (s - c_d[t]) * x76;
                gradient[77] += (s - c_d[t]) * x77;
                gradient[78] += (s - c_d[t]) * x78;
                gradient[79] += (s - c_d[t]) * x79;
                gradient[80] += (s - c_d[t]) * x80;
                gradient[81] += (s - c_d[t]) * x81;
                gradient[82] += (s - c_d[t]) * x82;
                gradient[83] += (s - c_d[t]) * x83;
                gradient[84] += (s - c_d[t]) * x84;
                gradient[85] += (s - c_d[t]) * x85;
                gradient[86] += (s - c_d[t]) * x86;
                gradient[87] += (s - c_d[t]) * x87;
                gradient[88] += (s - c_d[t]) * x88;
                gradient[89] += (s - c_d[t]) * x89;
                gradient[90] += (s - c_d[t]) * x90;
                gradient[91] += (s - c_d[t]) * x91;
                gradient[92] += (s - c_d[t]) * x92;
                gradient[93] += (s - c_d[t]) * x93;
                gradient[94] += (s - c_d[t]) * x94;
                gradient[95] += (s - c_d[t]) * x95;
                gradient[96] += (s - c_d[t]) * x96;
                gradient[97] += (s - c_d[t]) * x97;
                gradient[98] += (s - c_d[t]) * x98;
                gradient[99] += (s - c_d[t]) * x99;
                gradient[100] += (s - c_d[t]) * x100;
                gradient[101] += (s - c_d[t]) * x101;
                gradient[102] += (s - c_d[t]) * x102;
                gradient[103] += (s - c_d[t]) * x103;
                gradient[104] += (s - c_d[t]) * x104;
                gradient[105] += (s - c_d[t]) * x105;
                gradient[106] += (s - c_d[t]) * x106;
                gradient[107] += (s - c_d[t]) * x107;
                gradient[108] += (s - c_d[t]) * x108;
                gradient[109] += (s - c_d[t]) * x109;
                gradient[110] += (s - c_d[t]) * x110;
                gradient[111] += (s - c_d[t]) * x111;
                gradient[112] += (s - c_d[t]) * x112;
                gradient[113] += (s - c_d[t]) * x113;
                gradient[114] += (s - c_d[t]) * x114;
                gradient[115] += (s - c_d[t]) * x115;
                gradient[116] += (s - c_d[t]) * x116;
                gradient[117] += (s - c_d[t]) * x117;
                gradient[118] += (s - c_d[t]) * x118;
                gradient[119] += (s - c_d[t]) * x119;
                gradient[120] += (s - c_d[t]) * x120;
                gradient[121] += (s - c_d[t]) * x121;
                gradient[122] += (s - c_d[t]) * x122;
                gradient[123] += (s - c_d[t]) * x123;
                
                
                // end for t
            }
            
            // Update weights
            weights[0] -= n/(float)NUMBER_OF_VOLUMES * gradient[0];
            weights[1] -= n/(float)NUMBER_OF_VOLUMES * gradient[1];
            weights[2] -= n/(float)NUMBER_OF_VOLUMES * gradient[2];
            weights[3] -= n/(float)NUMBER_OF_VOLUMES * gradient[3];
            weights[4] -= n/(float)NUMBER_OF_VOLUMES * gradient[4];
            weights[5] -= n/(float)NUMBER_OF_VOLUMES * gradient[5];
            weights[6] -= n/(float)NUMBER_OF_VOLUMES * gradient[6];
            weights[7] -= n/(float)NUMBER_OF_VOLUMES * gradient[7];
            weights[8] -= n/(float)NUMBER_OF_VOLUMES * gradient[8];
            weights[9] -= n/(float)NUMBER_OF_VOLUMES * gradient[9];
            weights[10] -= n/(float)NUMBER_OF_VOLUMES * gradient[10];
            weights[11] -= n/(float)NUMBER_OF_VOLUMES * gradient[11];
            weights[12] -= n/(float)NUMBER_OF_VOLUMES * gradient[12];
            weights[13] -= n/(float)NUMBER_OF_VOLUMES * gradient[13];
            weights[14] -= n/(float)NUMBER_OF_VOLUMES * gradient[14];
            weights[15] -= n/(float)NUMBER_OF_VOLUMES * gradient[15];
            weights[16] -= n/(float)NUMBER_OF_VOLUMES * gradient[16];
            weights[17] -= n/(float)NUMBER_OF_VOLUMES * gradient[17];
            weights[18] -= n/(float)NUMBER_OF_VOLUMES * gradient[18];
            weights[19] -= n/(float)NUMBER_OF_VOLUMES * gradient[19];
            weights[20] -= n/(float)NUMBER_OF_VOLUMES * gradient[20];
            weights[21] -= n/(float)NUMBER_OF_VOLUMES * gradient[21];
            weights[22] -= n/(float)NUMBER_OF_VOLUMES * gradient[22];
            weights[23] -= n/(float)NUMBER_OF_VOLUMES * gradient[23];
            weights[24] -= n/(float)NUMBER_OF_VOLUMES * gradient[24];
            weights[25] -= n/(float)NUMBER_OF_VOLUMES * gradient[25];
            weights[26] -= n/(float)NUMBER_OF_VOLUMES * gradient[26];
            weights[27] -= n/(float)NUMBER_OF_VOLUMES * gradient[27];
            weights[28] -= n/(float)NUMBER_OF_VOLUMES * gradient[28];
            weights[29] -= n/(float)NUMBER_OF_VOLUMES * gradient[29];
            weights[30] -= n/(float)NUMBER_OF_VOLUMES * gradient[30];
            weights[31] -= n/(float)NUMBER_OF_VOLUMES * gradient[31];
            weights[32] -= n/(float)NUMBER_OF_VOLUMES * gradient[32];
            weights[33] -= n/(float)NUMBER_OF_VOLUMES * gradient[33];
            weights[34] -= n/(float)NUMBER_OF_VOLUMES * gradient[34];
            weights[35] -= n/(float)NUMBER_OF_VOLUMES * gradient[35];
            weights[36] -= n/(float)NUMBER_OF_VOLUMES * gradient[36];
            weights[37] -= n/(float)NUMBER_OF_VOLUMES * gradient[37];
            weights[38] -= n/(float)NUMBER_OF_VOLUMES * gradient[38];
            weights[39] -= n/(float)NUMBER_OF_VOLUMES * gradient[39];
            weights[40] -= n/(float)NUMBER_OF_VOLUMES * gradient[40];
            weights[41] -= n/(float)NUMBER_OF_VOLUMES * gradient[41];
            weights[42] -= n/(float)NUMBER_OF_VOLUMES * gradient[42];
            weights[43] -= n/(float)NUMBER_OF_VOLUMES * gradient[43];
            weights[44] -= n/(float)NUMBER_OF_VOLUMES * gradient[44];
            weights[45] -= n/(float)NUMBER_OF_VOLUMES * gradient[45];
            weights[46] -= n/(float)NUMBER_OF_VOLUMES * gradient[46];
            weights[47] -= n/(float)NUMBER_OF_VOLUMES * gradient[47];
            weights[48] -= n/(float)NUMBER_OF_VOLUMES * gradient[48];
            weights[49] -= n/(float)NUMBER_OF_VOLUMES * gradient[49];
            weights[50] -= n/(float)NUMBER_OF_VOLUMES * gradient[50];
            weights[51] -= n/(float)NUMBER_OF_VOLUMES * gradient[51];
            weights[52] -= n/(float)NUMBER_OF_VOLUMES * gradient[52];
            weights[53] -= n/(float)NUMBER_OF_VOLUMES * gradient[53];
            weights[54] -= n/(float)NUMBER_OF_VOLUMES * gradient[54];
            weights[55] -= n/(float)NUMBER_OF_VOLUMES * gradient[55];
            weights[56] -= n/(float)NUMBER_OF_VOLUMES * gradient[56];
            weights[57] -= n/(float)NUMBER_OF_VOLUMES * gradient[57];
            weights[58] -= n/(float)NUMBER_OF_VOLUMES * gradient[58];
            weights[59] -= n/(float)NUMBER_OF_VOLUMES * gradient[59];
            weights[60] -= n/(float)NUMBER_OF_VOLUMES * gradient[60];
            weights[61] -= n/(float)NUMBER_OF_VOLUMES * gradient[61];
            weights[62] -= n/(float)NUMBER_OF_VOLUMES * gradient[62];
            weights[63] -= n/(float)NUMBER_OF_VOLUMES * gradient[63];
            weights[64] -= n/(float)NUMBER_OF_VOLUMES * gradient[64];
            weights[65] -= n/(float)NUMBER_OF_VOLUMES * gradient[65];
            weights[66] -= n/(float)NUMBER_OF_VOLUMES * gradient[66];
            weights[67] -= n/(float)NUMBER_OF_VOLUMES * gradient[67];
            weights[68] -= n/(float)NUMBER_OF_VOLUMES * gradient[68];
            weights[69] -= n/(float)NUMBER_OF_VOLUMES * gradient[69];
            weights[70] -= n/(float)NUMBER_OF_VOLUMES * gradient[70];
            weights[71] -= n/(float)NUMBER_OF_VOLUMES * gradient[71];
            weights[72] -= n/(float)NUMBER_OF_VOLUMES * gradient[72];
            weights[73] -= n/(float)NUMBER_OF_VOLUMES * gradient[73];
            weights[74] -= n/(float)NUMBER_OF_VOLUMES * gradient[74];
            weights[75] -= n/(float)NUMBER_OF_VOLUMES * gradient[75];
            weights[76] -= n/(float)NUMBER_OF_VOLUMES * gradient[76];
            weights[77] -= n/(float)NUMBER_OF_VOLUMES * gradient[77];
            weights[78] -= n/(float)NUMBER_OF_VOLUMES * gradient[78];
            weights[79] -= n/(float)NUMBER_OF_VOLUMES * gradient[79];
            weights[80] -= n/(float)NUMBER_OF_VOLUMES * gradient[80];
            weights[81] -= n/(float)NUMBER_OF_VOLUMES * gradient[81];
            weights[82] -= n/(float)NUMBER_OF_VOLUMES * gradient[82];
            weights[83] -= n/(float)NUMBER_OF_VOLUMES * gradient[83];
            weights[84] -= n/(float)NUMBER_OF_VOLUMES * gradient[84];
            weights[85] -= n/(float)NUMBER_OF_VOLUMES * gradient[85];
            weights[86] -= n/(float)NUMBER_OF_VOLUMES * gradient[86];
            weights[87] -= n/(float)NUMBER_OF_VOLUMES * gradient[87];
            weights[88] -= n/(float)NUMBER_OF_VOLUMES * gradient[88];
            weights[89] -= n/(float)NUMBER_OF_VOLUMES * gradient[89];
            weights[90] -= n/(float)NUMBER_OF_VOLUMES * gradient[90];
            weights[91] -= n/(float)NUMBER_OF_VOLUMES * gradient[91];
            weights[92] -= n/(float)NUMBER_OF_VOLUMES * gradient[92];
            weights[93] -= n/(float)NUMBER_OF_VOLUMES * gradient[93];
            weights[94] -= n/(float)NUMBER_OF_VOLUMES * gradient[94];
            weights[95] -= n/(float)NUMBER_OF_VOLUMES * gradient[95];
            weights[96] -= n/(float)NUMBER_OF_VOLUMES * gradient[96];
            weights[97] -= n/(float)NUMBER_OF_VOLUMES * gradient[97];
            weights[98] -= n/(float)NUMBER_OF_VOLUMES * gradient[98];
            weights[99] -= n/(float)NUMBER_OF_VOLUMES * gradient[99];
            weights[100] -= n/(float)NUMBER_OF_VOLUMES * gradient[100];
            weights[101] -= n/(float)NUMBER_OF_VOLUMES * gradient[101];
            weights[102] -= n/(float)NUMBER_OF_VOLUMES * gradient[102];
            weights[103] -= n/(float)NUMBER_OF_VOLUMES * gradient[103];
            weights[104] -= n/(float)NUMBER_OF_VOLUMES * gradient[104];
            weights[105] -= n/(float)NUMBER_OF_VOLUMES * gradient[105];
            weights[106] -= n/(float)NUMBER_OF_VOLUMES * gradient[106];
            weights[107] -= n/(float)NUMBER_OF_VOLUMES * gradient[107];
            weights[108] -= n/(float)NUMBER_OF_VOLUMES * gradient[108];
            weights[109] -= n/(float)NUMBER_OF_VOLUMES * gradient[109];
            weights[110] -= n/(float)NUMBER_OF_VOLUMES * gradient[110];
            weights[111] -= n/(float)NUMBER_OF_VOLUMES * gradient[111];
            weights[112] -= n/(float)NUMBER_OF_VOLUMES * gradient[112];
            weights[113] -= n/(float)NUMBER_OF_VOLUMES * gradient[113];
            weights[114] -= n/(float)NUMBER_OF_VOLUMES * gradient[114];
            weights[115] -= n/(float)NUMBER_OF_VOLUMES * gradient[115];
            weights[116] -= n/(float)NUMBER_OF_VOLUMES * gradient[116];
            weights[117] -= n/(float)NUMBER_OF_VOLUMES * gradient[117];
            weights[118] -= n/(float)NUMBER_OF_VOLUMES * gradient[118];
            weights[119] -= n/(float)NUMBER_OF_VOLUMES * gradient[119];
            weights[120] -= n/(float)NUMBER_OF_VOLUMES * gradient[120];
            weights[121] -= n/(float)NUMBER_OF_VOLUMES * gradient[121];
            weights[122] -= n/(float)NUMBER_OF_VOLUMES * gradient[122];
            weights[123] -= n/(float)NUMBER_OF_VOLUMES * gradient[123];
            
            
            // end for epocs
        }
        
        // Make classification
        float s;
        s =  weights[0] * 1.0f;
        
        float x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20;
        float x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38, x39, x40;
        float x41, x42, x43, x44, x45, x46, x47, x48, x49, x50, x51, x52, x53, x54, x55, x56, x57, x58, x59, x60;
        float x61, x62, x63, x64, x65, x66, x67, x68, x69, x70, x71, x72, x73, x74, x75, x76, x77, x78, x79, x80;
        float x81, x82, x83, x84, x85, x86, x87, x88, x89, x90, x91, x92, x93, x94, x95, x96, x97, x98, x99, x100;
        float x101, x102, x103, x104, x105, x106, x107, x108, x109, x110, x111, x112, x113, x114, x115, x116, x117, x118, x119, x120, x121, x122, x123;
        
        x1 = Volumes[Calculate4DIndex(x+0,y+0,z-3,validation,DATA_W,DATA_H,DATA_D)];
        x2 = Volumes[Calculate4DIndex(x-2,y-1,z-2,validation,DATA_W,DATA_H,DATA_D)];
        x3 = Volumes[Calculate4DIndex(x-2,y+0,z-2,validation,DATA_W,DATA_H,DATA_D)];
        x4 = Volumes[Calculate4DIndex(x-2,y+1,z-2,validation,DATA_W,DATA_H,DATA_D)];
        x5 = Volumes[Calculate4DIndex(x-1,y-2,z-2,validation,DATA_W,DATA_H,DATA_D)];
        x6 = Volumes[Calculate4DIndex(x-1,y-1,z-2,validation,DATA_W,DATA_H,DATA_D)];
        x7 = Volumes[Calculate4DIndex(x-1,y+0,z-2,validation,DATA_W,DATA_H,DATA_D)];
        x8 = Volumes[Calculate4DIndex(x-1,y+1,z-2,validation,DATA_W,DATA_H,DATA_D)];
        x9 = Volumes[Calculate4DIndex(x-1,y+2,z-2,validation,DATA_W,DATA_H,DATA_D)];
        x10 = Volumes[Calculate4DIndex(x+0,y-2,z-2,validation,DATA_W,DATA_H,DATA_D)];
        x11 = Volumes[Calculate4DIndex(x+0,y-1,z-2,validation,DATA_W,DATA_H,DATA_D)];
        x12 = Volumes[Calculate4DIndex(x+0,y+0,z-2,validation,DATA_W,DATA_H,DATA_D)];
        x13 = Volumes[Calculate4DIndex(x+0,y+1,z-2,validation,DATA_W,DATA_H,DATA_D)];
        x14 = Volumes[Calculate4DIndex(x+0,y+2,z-2,validation,DATA_W,DATA_H,DATA_D)];
        x15 = Volumes[Calculate4DIndex(x+1,y-2,z-2,validation,DATA_W,DATA_H,DATA_D)];
        x16 = Volumes[Calculate4DIndex(x+1,y-1,z-2,validation,DATA_W,DATA_H,DATA_D)];
        x17 = Volumes[Calculate4DIndex(x+1,y+0,z-2,validation,DATA_W,DATA_H,DATA_D)];
        x18 = Volumes[Calculate4DIndex(x+1,y+1,z-2,validation,DATA_W,DATA_H,DATA_D)];
        x19 = Volumes[Calculate4DIndex(x+1,y+2,z-2,validation,DATA_W,DATA_H,DATA_D)];
        x20 = Volumes[Calculate4DIndex(x+2,y-1,z-2,validation,DATA_W,DATA_H,DATA_D)];
        x21 = Volumes[Calculate4DIndex(x+2,y+0,z-2,validation,DATA_W,DATA_H,DATA_D)];
        x22 = Volumes[Calculate4DIndex(x+2,y+1,z-2,validation,DATA_W,DATA_H,DATA_D)];
        x23 = Volumes[Calculate4DIndex(x-2,y-2,z-1,validation,DATA_W,DATA_H,DATA_D)];
        x24 = Volumes[Calculate4DIndex(x-2,y-1,z-1,validation,DATA_W,DATA_H,DATA_D)];
        x25 = Volumes[Calculate4DIndex(x-2,y+0,z-1,validation,DATA_W,DATA_H,DATA_D)];
        x26 = Volumes[Calculate4DIndex(x-2,y+1,z-1,validation,DATA_W,DATA_H,DATA_D)];
        x27 = Volumes[Calculate4DIndex(x-2,y+2,z-1,validation,DATA_W,DATA_H,DATA_D)];
        x28 = Volumes[Calculate4DIndex(x-1,y-2,z-1,validation,DATA_W,DATA_H,DATA_D)];
        x29 = Volumes[Calculate4DIndex(x-1,y-1,z-1,validation,DATA_W,DATA_H,DATA_D)];
        x30 = Volumes[Calculate4DIndex(x-1,y+0,z-1,validation,DATA_W,DATA_H,DATA_D)];
        x31 = Volumes[Calculate4DIndex(x-1,y+1,z-1,validation,DATA_W,DATA_H,DATA_D)];
        x32 = Volumes[Calculate4DIndex(x-1,y+2,z-1,validation,DATA_W,DATA_H,DATA_D)];
        x33 = Volumes[Calculate4DIndex(x+0,y-2,z-1,validation,DATA_W,DATA_H,DATA_D)];
        x34 = Volumes[Calculate4DIndex(x+0,y-1,z-1,validation,DATA_W,DATA_H,DATA_D)];
        x35 = Volumes[Calculate4DIndex(x+0,y+0,z-1,validation,DATA_W,DATA_H,DATA_D)];
        x36 = Volumes[Calculate4DIndex(x+0,y+1,z-1,validation,DATA_W,DATA_H,DATA_D)];
        x37 = Volumes[Calculate4DIndex(x+0,y+2,z-1,validation,DATA_W,DATA_H,DATA_D)];
        x38 = Volumes[Calculate4DIndex(x+1,y-2,z-1,validation,DATA_W,DATA_H,DATA_D)];
        x39 = Volumes[Calculate4DIndex(x+1,y-1,z-1,validation,DATA_W,DATA_H,DATA_D)];
        x40 = Volumes[Calculate4DIndex(x+1,y+0,z-1,validation,DATA_W,DATA_H,DATA_D)];
        x41 = Volumes[Calculate4DIndex(x+1,y+1,z-1,validation,DATA_W,DATA_H,DATA_D)];
        x42 = Volumes[Calculate4DIndex(x+1,y+2,z-1,validation,DATA_W,DATA_H,DATA_D)];
        x43 = Volumes[Calculate4DIndex(x+2,y-2,z-1,validation,DATA_W,DATA_H,DATA_D)];
        x44 = Volumes[Calculate4DIndex(x+2,y-1,z-1,validation,DATA_W,DATA_H,DATA_D)];
        x45 = Volumes[Calculate4DIndex(x+2,y+0,z-1,validation,DATA_W,DATA_H,DATA_D)];
        x46 = Volumes[Calculate4DIndex(x+2,y+1,z-1,validation,DATA_W,DATA_H,DATA_D)];
        x47 = Volumes[Calculate4DIndex(x+2,y+2,z-1,validation,DATA_W,DATA_H,DATA_D)];
        x48 = Volumes[Calculate4DIndex(x-3,y+0,z+0,validation,DATA_W,DATA_H,DATA_D)];
        x49 = Volumes[Calculate4DIndex(x-2,y-2,z+0,validation,DATA_W,DATA_H,DATA_D)];
        x50 = Volumes[Calculate4DIndex(x-2,y-1,z+0,validation,DATA_W,DATA_H,DATA_D)];
        x51 = Volumes[Calculate4DIndex(x-2,y+0,z+0,validation,DATA_W,DATA_H,DATA_D)];
        x52 = Volumes[Calculate4DIndex(x-2,y+1,z+0,validation,DATA_W,DATA_H,DATA_D)];
        x53 = Volumes[Calculate4DIndex(x-2,y+2,z+0,validation,DATA_W,DATA_H,DATA_D)];
        x54 = Volumes[Calculate4DIndex(x-1,y-2,z+0,validation,DATA_W,DATA_H,DATA_D)];
        x55 = Volumes[Calculate4DIndex(x-1,y-1,z+0,validation,DATA_W,DATA_H,DATA_D)];
        x56 = Volumes[Calculate4DIndex(x-1,y+0,z+0,validation,DATA_W,DATA_H,DATA_D)];
        x57 = Volumes[Calculate4DIndex(x-1,y+1,z+0,validation,DATA_W,DATA_H,DATA_D)];
        x58 = Volumes[Calculate4DIndex(x-1,y+2,z+0,validation,DATA_W,DATA_H,DATA_D)];
        x59 = Volumes[Calculate4DIndex(x+0,y-3,z+0,validation,DATA_W,DATA_H,DATA_D)];
        x60 = Volumes[Calculate4DIndex(x+0,y-2,z+0,validation,DATA_W,DATA_H,DATA_D)];
        x61 = Volumes[Calculate4DIndex(x+0,y-1,z+0,validation,DATA_W,DATA_H,DATA_D)];
        x62 = Volumes[Calculate4DIndex(x+0,y+0,z+0,validation,DATA_W,DATA_H,DATA_D)];
        x63 = Volumes[Calculate4DIndex(x+0,y+1,z+0,validation,DATA_W,DATA_H,DATA_D)];
        x64 = Volumes[Calculate4DIndex(x+0,y+2,z+0,validation,DATA_W,DATA_H,DATA_D)];
        x65 = Volumes[Calculate4DIndex(x+0,y+3,z+0,validation,DATA_W,DATA_H,DATA_D)];
        x66 = Volumes[Calculate4DIndex(x+1,y-2,z+0,validation,DATA_W,DATA_H,DATA_D)];
        x67 = Volumes[Calculate4DIndex(x+1,y-1,z+0,validation,DATA_W,DATA_H,DATA_D)];
        x68 = Volumes[Calculate4DIndex(x+1,y+0,z+0,validation,DATA_W,DATA_H,DATA_D)];
        x69 = Volumes[Calculate4DIndex(x+1,y+1,z+0,validation,DATA_W,DATA_H,DATA_D)];
        x70 = Volumes[Calculate4DIndex(x+1,y+2,z+0,validation,DATA_W,DATA_H,DATA_D)];
        x71 = Volumes[Calculate4DIndex(x+2,y-2,z+0,validation,DATA_W,DATA_H,DATA_D)];
        x72 = Volumes[Calculate4DIndex(x+2,y-1,z+0,validation,DATA_W,DATA_H,DATA_D)];
        x73 = Volumes[Calculate4DIndex(x+2,y+0,z+0,validation,DATA_W,DATA_H,DATA_D)];
        x74 = Volumes[Calculate4DIndex(x+2,y+1,z+0,validation,DATA_W,DATA_H,DATA_D)];
        x75 = Volumes[Calculate4DIndex(x+2,y+2,z+0,validation,DATA_W,DATA_H,DATA_D)];
        x76 = Volumes[Calculate4DIndex(x+3,y+0,z+0,validation,DATA_W,DATA_H,DATA_D)];
        x77 = Volumes[Calculate4DIndex(x-2,y-2,z+1,validation,DATA_W,DATA_H,DATA_D)];
        x78 = Volumes[Calculate4DIndex(x-2,y-1,z+1,validation,DATA_W,DATA_H,DATA_D)];
        x79 = Volumes[Calculate4DIndex(x-2,y+0,z+1,validation,DATA_W,DATA_H,DATA_D)];
        x80 = Volumes[Calculate4DIndex(x-2,y+1,z+1,validation,DATA_W,DATA_H,DATA_D)];
        x81 = Volumes[Calculate4DIndex(x-2,y+2,z+1,validation,DATA_W,DATA_H,DATA_D)];
        x82 = Volumes[Calculate4DIndex(x-1,y-2,z+1,validation,DATA_W,DATA_H,DATA_D)];
        x83 = Volumes[Calculate4DIndex(x-1,y-1,z+1,validation,DATA_W,DATA_H,DATA_D)];
        x84 = Volumes[Calculate4DIndex(x-1,y+0,z+1,validation,DATA_W,DATA_H,DATA_D)];
        x85 = Volumes[Calculate4DIndex(x-1,y+1,z+1,validation,DATA_W,DATA_H,DATA_D)];
        x86 = Volumes[Calculate4DIndex(x-1,y+2,z+1,validation,DATA_W,DATA_H,DATA_D)];
        x87 = Volumes[Calculate4DIndex(x+0,y-2,z+1,validation,DATA_W,DATA_H,DATA_D)];
        x88 = Volumes[Calculate4DIndex(x+0,y-1,z+1,validation,DATA_W,DATA_H,DATA_D)];
        x89 = Volumes[Calculate4DIndex(x+0,y+0,z+1,validation,DATA_W,DATA_H,DATA_D)];
        x90 = Volumes[Calculate4DIndex(x+0,y+1,z+1,validation,DATA_W,DATA_H,DATA_D)];
        x91 = Volumes[Calculate4DIndex(x+0,y+2,z+1,validation,DATA_W,DATA_H,DATA_D)];
        x92 = Volumes[Calculate4DIndex(x+1,y-2,z+1,validation,DATA_W,DATA_H,DATA_D)];
        x93 = Volumes[Calculate4DIndex(x+1,y-1,z+1,validation,DATA_W,DATA_H,DATA_D)];
        x94 = Volumes[Calculate4DIndex(x+1,y+0,z+1,validation,DATA_W,DATA_H,DATA_D)];
        x95 = Volumes[Calculate4DIndex(x+1,y+1,z+1,validation,DATA_W,DATA_H,DATA_D)];
        x96 = Volumes[Calculate4DIndex(x+1,y+2,z+1,validation,DATA_W,DATA_H,DATA_D)];
        x97 = Volumes[Calculate4DIndex(x+2,y-2,z+1,validation,DATA_W,DATA_H,DATA_D)];
        x98 = Volumes[Calculate4DIndex(x+2,y-1,z+1,validation,DATA_W,DATA_H,DATA_D)];
        x99 = Volumes[Calculate4DIndex(x+2,y+0,z+1,validation,DATA_W,DATA_H,DATA_D)];
        x100 = Volumes[Calculate4DIndex(x+2,y+1,z+1,validation,DATA_W,DATA_H,DATA_D)];
        x101 = Volumes[Calculate4DIndex(x+2,y+2,z+1,validation,DATA_W,DATA_H,DATA_D)];
        x102 = Volumes[Calculate4DIndex(x-2,y-1,z+2,validation,DATA_W,DATA_H,DATA_D)];
        x103 = Volumes[Calculate4DIndex(x-2,y+0,z+2,validation,DATA_W,DATA_H,DATA_D)];
        x104 = Volumes[Calculate4DIndex(x-2,y+1,z+2,validation,DATA_W,DATA_H,DATA_D)];
        x105 = Volumes[Calculate4DIndex(x-1,y-2,z+2,validation,DATA_W,DATA_H,DATA_D)];
        x106 = Volumes[Calculate4DIndex(x-1,y-1,z+2,validation,DATA_W,DATA_H,DATA_D)];
        x107 = Volumes[Calculate4DIndex(x-1,y+0,z+2,validation,DATA_W,DATA_H,DATA_D)];
        x108 = Volumes[Calculate4DIndex(x-1,y+1,z+2,validation,DATA_W,DATA_H,DATA_D)];
        x109 = Volumes[Calculate4DIndex(x-1,y+2,z+2,validation,DATA_W,DATA_H,DATA_D)];
        x110 = Volumes[Calculate4DIndex(x+0,y-2,z+2,validation,DATA_W,DATA_H,DATA_D)];
        x111 = Volumes[Calculate4DIndex(x+0,y-1,z+2,validation,DATA_W,DATA_H,DATA_D)];
        x112 = Volumes[Calculate4DIndex(x+0,y+0,z+2,validation,DATA_W,DATA_H,DATA_D)];
        x113 = Volumes[Calculate4DIndex(x+0,y+1,z+2,validation,DATA_W,DATA_H,DATA_D)];
        x114 = Volumes[Calculate4DIndex(x+0,y+2,z+2,validation,DATA_W,DATA_H,DATA_D)];
        x115 = Volumes[Calculate4DIndex(x+1,y-2,z+2,validation,DATA_W,DATA_H,DATA_D)];
        x116 = Volumes[Calculate4DIndex(x+1,y-1,z+2,validation,DATA_W,DATA_H,DATA_D)];
        x117 = Volumes[Calculate4DIndex(x+1,y+0,z+2,validation,DATA_W,DATA_H,DATA_D)];
        x118 = Volumes[Calculate4DIndex(x+1,y+1,z+2,validation,DATA_W,DATA_H,DATA_D)];
        x119 = Volumes[Calculate4DIndex(x+1,y+2,z+2,validation,DATA_W,DATA_H,DATA_D)];
        x120 = Volumes[Calculate4DIndex(x+2,y-1,z+2,validation,DATA_W,DATA_H,DATA_D)];
        x121 = Volumes[Calculate4DIndex(x+2,y+0,z+2,validation,DATA_W,DATA_H,DATA_D)];
        x122 = Volumes[Calculate4DIndex(x+2,y+1,z+2,validation,DATA_W,DATA_H,DATA_D)];
        x123 = Volumes[Calculate4DIndex(x+0,y+0,z+3,validation,DATA_W,DATA_H,DATA_D)];
        
        
       /*
        if (x==9 && y==22 && z==26)
        //if (!printed)
        {
            printf("classifying %i\n", validation);

          	printf("1:%f 2:%f 3:%f 4:%f 5:%f 6:%f 7:%f 8:%f 9:%f 10:%f 11:%f 12:%f 13:%f 14:%f 15:%f 16:%f 17:%f 18:%f 19:%f 20:%f 21:%f 22:%f 23:%f 24:%f 25:%f 26:%f 27:%f 28:%f 29:%f 30:%f 31:%f 32:%f ",x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32);
            printf(" 33:%f 34:%f 35:%f 36:%f 37:%f 38:%f 39:%f 40:%f 41:%f 42:%f 43:%f 44:%f 45:%f 46:%f 47:%f 48:%f 49:%f 50:%f 51:%f 52:%f 53:%f 54:%f 55:%f 56:%f 57:%f 58:%f 59:%f 60:%f 61:%f 62:%f 63:%f 64:%f", x33, x34, x35, x36, x37, x38, x39, x40, x41, x42, x43, x44, x45, x46, x47, x48, x49, x50, x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, x61, x62, x63, x64);
            printf(" 65:%f 66:%f 67:%f 68:%f 69:%f 70:%f 71:%f 72:%f 73:%f 74:%f 75:%f 76:%f 77:%f 78:%f 79:%f 80:%f 81:%f 82:%f 83:%f 84:%f 85:%f 86:%f 87:%f 88:%f 89:%f 90:%f 91:%f 92:%f 93:%f 94:%f 95:%f 96:%f", x65, x66, x67, x68, x69, x70, x71, x72, x73, x74, x75, x76, x77, x78, x79, x80, x81, x82, x83, x84, x85, x86, x87, x88, x89, x90, x91, x92, x93, x94, x95, x96);
            printf(" 97:%f 98:%f 99:%f 100:%f 101:%f 102:%f 103:%f 104:%f 105:%f 106:%f 107:%f 108:%f 109:%f 110:%f 111:%f 112:%f 113:%f 114:%f 115:%f 116:%f 117:%f 118:%f 119:%f 120:%f 121:%f 122:%f 123:%f ", x97, x98, x99, x100, x101, x102, x103, x104, x105, x106, x107, x108, x109, x110, x111, x112, x113, x114, x115, x116, x117, x118, x119, x120, x121, x122, x123);
            printf("\n");
            printed = true;
               
        }*/
        
        // weights sum
        s += weights[1] * x1;
        s += weights[2] * x2;
        s += weights[3] * x3;
        s += weights[4] * x4;
        s += weights[5] * x5;
        s += weights[6] * x6;
        s += weights[7] * x7;
        s += weights[8] * x8;
        s += weights[9] * x9;
        s += weights[10] * x10;
        s += weights[11] * x11;
        s += weights[12] * x12;
        s += weights[13] * x13;
        s += weights[14] * x14;
        s += weights[15] * x15;
        s += weights[16] * x16;
        s += weights[17] * x17;
        s += weights[18] * x18;
        s += weights[19] * x19;
        s += weights[20] * x20;
        s += weights[21] * x21;
        s += weights[22] * x22;
        s += weights[23] * x23;
        s += weights[24] * x24;
        s += weights[25] * x25;
        s += weights[26] * x26;
        s += weights[27] * x27;
        s += weights[28] * x28;
        s += weights[29] * x29;
        s += weights[30] * x30;
        s += weights[31] * x31;
        s += weights[32] * x32;
        s += weights[33] * x33;
        s += weights[34] * x34;
        s += weights[35] * x35;
        s += weights[36] * x36;
        s += weights[37] * x37;
        s += weights[38] * x38;
        s += weights[39] * x39;
        s += weights[40] * x40;
        s += weights[41] * x41;
        s += weights[42] * x42;
        s += weights[43] * x43;
        s += weights[44] * x44;
        s += weights[45] * x45;
        s += weights[46] * x46;
        s += weights[47] * x47;
        s += weights[48] * x48;
        s += weights[49] * x49;
        s += weights[50] * x50;
        s += weights[51] * x51;
        s += weights[52] * x52;
        s += weights[53] * x53;
        s += weights[54] * x54;
        s += weights[55] * x55;
        s += weights[56] * x56;
        s += weights[57] * x57;
        s += weights[58] * x58;
        s += weights[59] * x59;
        s += weights[60] * x60;
        s += weights[61] * x61;
        s += weights[62] * x62;
        s += weights[63] * x63;
        s += weights[64] * x64;
        s += weights[65] * x65;
        s += weights[66] * x66;
        s += weights[67] * x67;
        s += weights[68] * x68;
        s += weights[69] * x69;
        s += weights[70] * x70;
        s += weights[71] * x71;
        s += weights[72] * x72;
        s += weights[73] * x73;
        s += weights[74] * x74;
        s += weights[75] * x75;
        s += weights[76] * x76;
        s += weights[77] * x77;
        s += weights[78] * x78;
        s += weights[79] * x79;
        s += weights[80] * x80;
        s += weights[81] * x81;
        s += weights[82] * x82;
        s += weights[83] * x83;
        s += weights[84] * x84;
        s += weights[85] * x85;
        s += weights[86] * x86;
        s += weights[87] * x87;
        s += weights[88] * x88;
        s += weights[89] * x89;
        s += weights[90] * x90;
        s += weights[91] * x91;
        s += weights[92] * x92;
        s += weights[93] * x93;
        s += weights[94] * x94;
        s += weights[95] * x95;
        s += weights[96] * x96;
        s += weights[97] * x97;
        s += weights[98] * x98;
        s += weights[99] * x99;
        s += weights[100] * x100;
        s += weights[101] * x101;
        s += weights[102] * x102;
        s += weights[103] * x103;
        s += weights[104] * x104;
        s += weights[105] * x105;
        s += weights[106] * x106;
        s += weights[107] * x107;
        s += weights[108] * x108;
        s += weights[109] * x109;
        s += weights[110] * x110;
        s += weights[111] * x111;
        s += weights[112] * x112;
        s += weights[113] * x113;
        s += weights[114] * x114;
        s += weights[115] * x115;
        s += weights[116] * x116;
        s += weights[117] * x117;
        s += weights[118] * x118;
        s += weights[119] * x119;
        s += weights[120] * x120;
        s += weights[121] * x121;
        s += weights[122] * x122;
        s += weights[123] * x123;
        
        float classification;
        if (s > 0.0f)
        {
            classification = 0.0f;
        }
        else
        {
            classification = 1.0f;
        }
        
        if (classification == c_Correct_Classes[validation])
        {
            classification_performance++;
        }
        
        // end for validation
    }
    
    Classifier_Performance[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = Classifier_Performance[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] + (float)classification_performance / (float)NUMBER_OF_VOLUMES;
}


__kernel void CalculateStatisticalMapSearchlight___(__global float* Classifier_Performance,
                                                  __global const float* Volumes,
                                                  __global const float* Mask,
                                                  __constant float* c_d,
                                                  __constant float* c_Correct_Classes,
                                                  __private int DATA_W,
                                                  __private int DATA_H,
                                                  __private int DATA_D,
                                                  __private int NUMBER_OF_VOLUMES,

                                                  __private float n,
                                                  __private int EPOCS)

{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    
    if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
        return;
    
    if ( Mask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] != 1.0f )
    {
        Classifier_Performance[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = 0.0f;
        return;
    }
    
    if ( ((x + 1) >= DATA_W) || ((y + 1) >= DATA_H) || ((z + 1) >= DATA_D) )
    {
        Classifier_Performance[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = 0.0f;
        return;
    }
    
    
    if ( ((x - 1) < 0) || ((y - 1) < 0) || ((z - 1) < 0) )
    {
        Classifier_Performance[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = 0.0f;
        return;
    }
    
    
    int classification_performance = 0;
    

	// 
	// Training
	//

    float weights[20];
       
    weights[0]  = 0.0f;
    weights[1]  = 0.0f;
    weights[2]  = 0.0f;
    weights[3]  = 0.0f;
    weights[4]  = 0.0f;
    weights[5]  = 0.0f;
    weights[6]  = 0.0f;
    weights[7]  = 0.0f;
    weights[8]  = 0.0f;
    weights[9]  = 0.0f;
    weights[10] = 0.0f;
    weights[11] = 0.0f;
    weights[12] = 0.0f;
    weights[13] = 0.0f;
    weights[14] = 0.0f;
    weights[15] = 0.0f;
    weights[16] = 0.0f;
    weights[17] = 0.0f;
    weights[18] = 0.0f;
    weights[19] = 0.0f;
        
    // Do training for a number of iterations
    for (int epoc = 0; epoc < EPOCS; epoc++)
    {
        float gradient[20];
            
        gradient[0] = 0.0f;
        gradient[1] = 0.0f;
        gradient[2] = 0.0f;
        gradient[3] = 0.0f;
        gradient[4] = 0.0f;
        gradient[5] = 0.0f;
        gradient[6] = 0.0f;
        gradient[7] = 0.0f;
        gradient[8] = 0.0f;
        gradient[9] = 0.0f;
        gradient[10] = 0.0f;
        gradient[11] = 0.0f;
        gradient[12] = 0.0f;
        gradient[13] = 0.0f;
        gradient[14] = 0.0f;
        gradient[15] = 0.0f;
        gradient[16] = 0.0f;
        gradient[17] = 0.0f;
        gradient[18] = 0.0f;
        gradient[19] = 0.0f;
            
        for (int t = 0; t < NUMBER_OF_VOLUMES / 2; t++)
        {
   			// Ignore censored volumes
			if (c_Correct_Classes[t] == 9999.0f)
			{
				continue;
			}
             
            // Make classification
            float s;
            s =  weights[0] * 1.0f;
                
            float x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19;
                
            x1 = Volumes[Calculate4DIndex(x-1,y,z-1,t,DATA_W,DATA_H,DATA_D)];
            x2 = Volumes[Calculate4DIndex(x,y-1,z-1,t,DATA_W,DATA_H,DATA_D)];
            x3 = Volumes[Calculate4DIndex(x,y,z-1,t,DATA_W,DATA_H,DATA_D)];
            x4 = Volumes[Calculate4DIndex(x,y+1,z-1,t,DATA_W,DATA_H,DATA_D)];
            x5 = Volumes[Calculate4DIndex(x+1,y,z-1,t,DATA_W,DATA_H,DATA_D)];
                
            x6 = Volumes[Calculate4DIndex(x-1,y-1,z,t,DATA_W,DATA_H,DATA_D)];
            x7 = Volumes[Calculate4DIndex(x-1,y,z,t,DATA_W,DATA_H,DATA_D)];
            x8 = Volumes[Calculate4DIndex(x-1,y+1,z,t,DATA_W,DATA_H,DATA_D)];
            x9 = Volumes[Calculate4DIndex(x,y-1,z,t,DATA_W,DATA_H,DATA_D)];
            x10 = Volumes[Calculate4DIndex(x,y,z,t,DATA_W,DATA_H,DATA_D)];
            x11 = Volumes[Calculate4DIndex(x,y+1,z,t,DATA_W,DATA_H,DATA_D)];
            x12 = Volumes[Calculate4DIndex(x+1,y-1,z,t,DATA_W,DATA_H,DATA_D)];
            x13 = Volumes[Calculate4DIndex(x+1,y,z,t,DATA_W,DATA_H,DATA_D)];
            x14 = Volumes[Calculate4DIndex(x+1,y+1,z,t,DATA_W,DATA_H,DATA_D)];
                
            x15 = Volumes[Calculate4DIndex(x-1,y,z+1,t,DATA_W,DATA_H,DATA_D)];
            x16 = Volumes[Calculate4DIndex(x,y-1,z+1,t,DATA_W,DATA_H,DATA_D)];
            x17 = Volumes[Calculate4DIndex(x,y,z+1,t,DATA_W,DATA_H,DATA_D)];
            x18 = Volumes[Calculate4DIndex(x,y+1,z+1,t,DATA_W,DATA_H,DATA_D)];
            x19 = Volumes[Calculate4DIndex(x+1,y,z+1,t,DATA_W,DATA_H,DATA_D)];
                
            // z - 1
            s += weights[1] * x1;
            s += weights[2] * x2;
            s += weights[3] * x3;
            s += weights[4] * x4;
            s += weights[5] * x5;
                
            // z
            s += weights[6] * x6;
            s += weights[7] * x7;
            s += weights[8] * x8;
            s += weights[9] * x9;
            s += weights[10] * x10;
            s += weights[11] * x11;
            s += weights[12] * x12;
            s += weights[13] * x13;
            s += weights[14] * x14;
                
            // z + 1
            s += weights[15] * x15;
            s += weights[16] * x16;
            s += weights[17] * x17;
            s += weights[18] * x18;
            s += weights[19] * x19;
                
            // Calculate contribution to gradient
            gradient[0] += (s - c_d[t]) * 1.0f;
                
            // z - 1
            gradient[1]  += (s - c_d[t]) * x1;
            gradient[2]  += (s - c_d[t]) * x2;
            gradient[3]  += (s - c_d[t]) * x3;
            gradient[4]  += (s - c_d[t]) * x4;
            gradient[5]  += (s - c_d[t]) * x5;
                
            // z
            gradient[6]  += (s - c_d[t]) * x6;
            gradient[7]  += (s - c_d[t]) * x7;
            gradient[8]  += (s - c_d[t]) * x8;
            gradient[9]  += (s - c_d[t]) * x9;
            gradient[10] += (s - c_d[t]) * x10;
            gradient[11] += (s - c_d[t]) * x11;
            gradient[12] += (s - c_d[t]) * x12;
            gradient[13] += (s - c_d[t]) * x13;
            gradient[14] += (s - c_d[t]) * x14;
                
            // z + 1
            gradient[15] += (s - c_d[t]) * x15;
            gradient[16] += (s - c_d[t]) * x16;
            gradient[17] += (s - c_d[t]) * x17;
            gradient[18] += (s - c_d[t]) * x18;
            gradient[19] += (s - c_d[t]) * x19;
                
            // end for t
        }
            
        // Update weights
        weights[0] -= n/(float)NUMBER_OF_VOLUMES * gradient[0];
        weights[1] -= n/(float)NUMBER_OF_VOLUMES * gradient[1];
        weights[2] -= n/(float)NUMBER_OF_VOLUMES * gradient[2];
        weights[3] -= n/(float)NUMBER_OF_VOLUMES * gradient[3];
        weights[4] -= n/(float)NUMBER_OF_VOLUMES * gradient[4];
        weights[5] -= n/(float)NUMBER_OF_VOLUMES * gradient[5];
        weights[6] -= n/(float)NUMBER_OF_VOLUMES * gradient[6];
        weights[7] -= n/(float)NUMBER_OF_VOLUMES * gradient[7];
        weights[8] -= n/(float)NUMBER_OF_VOLUMES * gradient[8];
        weights[9] -= n/(float)NUMBER_OF_VOLUMES * gradient[9];
        weights[10] -= n/(float)NUMBER_OF_VOLUMES * gradient[10];
        weights[11] -= n/(float)NUMBER_OF_VOLUMES * gradient[11];
        weights[12] -= n/(float)NUMBER_OF_VOLUMES * gradient[12];
        weights[13] -= n/(float)NUMBER_OF_VOLUMES * gradient[13];
        weights[14] -= n/(float)NUMBER_OF_VOLUMES * gradient[14];
        weights[15] -= n/(float)NUMBER_OF_VOLUMES * gradient[15];
        weights[16] -= n/(float)NUMBER_OF_VOLUMES * gradient[16];
        weights[17] -= n/(float)NUMBER_OF_VOLUMES * gradient[17];
        weights[18] -= n/(float)NUMBER_OF_VOLUMES * gradient[18];
        weights[19] -= n/(float)NUMBER_OF_VOLUMES * gradient[19];
        
        // end for epocs
    }


	//
	// Testing
	//
        
	float s;

	int uncensoredVolumes = 0;

    // Make classifications
    for (int t = NUMBER_OF_VOLUMES / 2 + 1; t < NUMBER_OF_VOLUMES; t++)
    {
		// Ignore censored volumes
		if (c_Correct_Classes[t] == 9999.0f)
		{
			continue;
		}

		uncensoredVolumes++;

	    s =  weights[0] * 1.0f;
        
	    float x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19;
        
	    x1 = Volumes[Calculate4DIndex(x-1,y,z-1,t,DATA_W,DATA_H,DATA_D)];
	    x2 = Volumes[Calculate4DIndex(x,y-1,z-1,t,DATA_W,DATA_H,DATA_D)];
	    x3 = Volumes[Calculate4DIndex(x,y,z-1,t,DATA_W,DATA_H,DATA_D)];
	    x4 = Volumes[Calculate4DIndex(x,y+1,z-1,t,DATA_W,DATA_H,DATA_D)];
		x5 = Volumes[Calculate4DIndex(x+1,y,z-1,t,DATA_W,DATA_H,DATA_D)];
        
	    x6 = Volumes[Calculate4DIndex(x-1,y-1,z,t,DATA_W,DATA_H,DATA_D)];
	    x7 = Volumes[Calculate4DIndex(x-1,y,z,t,DATA_W,DATA_H,DATA_D)];
	    x8 = Volumes[Calculate4DIndex(x-1,y+1,z,t,DATA_W,DATA_H,DATA_D)];
	    x9 = Volumes[Calculate4DIndex(x,y-1,z,t,DATA_W,DATA_H,DATA_D)];
	    x10 = Volumes[Calculate4DIndex(x,y,z,t,DATA_W,DATA_H,DATA_D)];

    	x11 = Volumes[Calculate4DIndex(x,y+1,z,t,DATA_W,DATA_H,DATA_D)];
    	x12 = Volumes[Calculate4DIndex(x+1,y-1,z,t,DATA_W,DATA_H,DATA_D)];
    	x13 = Volumes[Calculate4DIndex(x+1,y,z,t,DATA_W,DATA_H,DATA_D)];
    	x14 = Volumes[Calculate4DIndex(x+1,y+1,z,t,DATA_W,DATA_H,DATA_D)];
    	    
    	x15 = Volumes[Calculate4DIndex(x-1,y,z+1,t,DATA_W,DATA_H,DATA_D)];
    	x16 = Volumes[Calculate4DIndex(x,y-1,z+1,t,DATA_W,DATA_H,DATA_D)];
    	x17 = Volumes[Calculate4DIndex(x,y,z+1,t,DATA_W,DATA_H,DATA_D)];
    	x18 = Volumes[Calculate4DIndex(x,y+1,z+1,t,DATA_W,DATA_H,DATA_D)];
    	x19 = Volumes[Calculate4DIndex(x+1,y,z+1,t,DATA_W,DATA_H,DATA_D)];
        
    	// z - 1
    	s += weights[1] * x1;
    	s += weights[2] * x2;
    	s += weights[3] * x3;
    	s += weights[4] * x4;
    	s += weights[5] * x5;
    	    
	    // z
 	    s += weights[6] * x6;
   	    s += weights[7] * x7;
   	    s += weights[8] * x8;
   	    s += weights[9] * x9;
   	    s += weights[10] * x10;
   	    s += weights[11] * x11;
   	    s += weights[12] * x12;
   	    s += weights[13] * x13;
   	    s += weights[14] * x14;
    	    
   	    // z + 1
   	    s += weights[15] * x15;
   	    s += weights[16] * x16;
   	    s += weights[17] * x17;
   	    s += weights[18] * x18;
   	    s += weights[19] * x19;
        
   	    float classification;
   	    if (s > 0.0f)
   	    {
   	        classification = 0.0f;
   	    }
   	    else
   	    {
   	        classification = 1.0f;
   	    }
       
   	    if (classification == c_Correct_Classes[t])
        {
   	        classification_performance++;
   	    }           
    }
  
    Classifier_Performance[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = (float)classification_performance / (float)uncensoredVolumes;
}
