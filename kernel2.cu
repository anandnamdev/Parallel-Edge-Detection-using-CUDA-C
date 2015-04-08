#include "cuda_runtime.h"
#include <stdio.h>
#include <stdint.h>
#include "device_launch_parameters.h"

__global__ void refineGradients(double* out,double* in,int row,int col)
{
	//Get individual indices
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	
	//check bounds and process
	if(i < col-2 && j < row-2)
	{
		//get X gradient
		float GX = in[(i+2)*row + j] + in[(i+2)*row + j+1]*2 +  in[(i+2)*row + j+2]-(in[(i)*row + j] + in[(i)*row + j+1]*2 + in[(i)*row + j+2]);
		
		//get Y gradient
		float GY = in[(i)*row + j+2]+in[(i+1)*row + j+2]*2+in[(i+2)*row + j+2]-(in[(i)*row + j]+in[(i+1)*row + j]*2+in[(i+2)*row + j]);
		
		//calculate final Gradient
		float Gfinal = sqrt((GX*GX)+(GY*GY));
		
		//replace in output matrix
		out[(i)*row + j] = Gfinal;
	}
	return ;
}
