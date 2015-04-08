#include "cuda_runtime.h"
#include <stdio.h>
#include <stdint.h>
#include "device_launch_parameters.h"

__global__ void RGBtoGray (double *gray,double *R,double *G,double *B,int row,int col)
{
	int j=(blockIdx.x)*(blockDim.x)+(threadIdx.x);
	int i=(blockIdx.y)*(blockDim.y)+(threadIdx.y);
	
    if( i<col && j<row )
	   gray[i+j*row]=R[i+j*row]*0.299 + G[i+j*row]*0.587 + B[i+j*row]*0.114;
	//if( i<row && j<col )
	   //gray[i*col+j]=R[i*col+j]*0.299 + G[i*col+j]*0.587 + B[i*col+j]*0.114;
}
