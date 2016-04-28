
#include <cuda.h>
#include "cuda_runtime_api.h"
#include "device_functions.h"

#include <iostream>
#include <cmath>
#include <ostream>
#include <fstream>
#include <math.h>

using namespace std;

// Define Given Parameters.  NOTE TIMELIMIT is approximate.
#define DIVISIONS  1024.
#define LENX       50.
#define TS         .5
#define TIMELIMIT  5000.
#define REAL       float
#define TH_DIFF    8.418e-5

__constant__ REAL fo;

__global__ void upTriangle(REAL *IC, REAL *right, REAL *left)
{

	__shared__ REAL temper[64];
	__shared__ REAL sR[32];
	__shared__ REAL sL[32];

	unsigned int gid = blockDim.x * blockIdx.x + threadIdx; //Global Thread ID
	unsigned int tid = threadIdx.x;

	Temper[tid] = IC[gid];
	__syncthreads();
	int itr = -1;

	for (unsigned int k = 1; k<16; k++)
	{
		int sw = (k & 1);
		int shft = sw*2-1;

		if (tid <= (32-k-1) && >= k)
		{
			temper[tid + (32*sw)] = fo * (IC[tid+(32*shft)-1] + IC[tid+(32*shft)+1]) + (1-2.*fo) * IC[tid+(32*shft)];
		}

		__syncthreads();
		if (sw)
		{
			sL[k+itr] = temper[k-1];
			sL[k+itr+1] = temper[k];
			sL[k+itr+2] = temper[32+k];
			sL[k+itr+3] = temper[33+k];
			sR[k+itr] = temper[31-k];
			sR[k+itr+1] = temper[32-k];
			sR[k+itr+2] = temper[61-k];
			sR[k+itr+3] = temper[62-k];
			itr += 2;
		}

	}

	right[gid] = sR[tid];
	left[gid] = sL[tid];
	__syncthreads();

	}

}

__global__ void downTriangle(REAL *IC)
{



}


int main()
{
	const int dv = int(DIVISIONS);
	const int bks = dv/32;
	const REAL ds = LENX/(DIVISIONS-1);
	REAL fou = TS*TH_DIFF/(ds*ds);

	REAL IC[dv];
	REAL *d_IC[dv], *d_right[dv], *d_left[dv];

	for (int k = 0; k<dv; k++)
	{
		IC[k] = 500.f*expf((-ds*k)/LENX);
	}

	// Call out the file before the loop and write out the initial condition.
	ofstream fwr;
	fwr.open("1DHeatEQResult.dat");
	// Write out x length and then delta x and then delta t.  First item of each line is timestamp.
	filewrite << LENX << " " << ds << " " << TS << " " << 0 <<endl;

	for (int k = 0; k<dv; k++)
	{
		fwr << IC[k] << " ";
	}

	fwr << endl;

	// Put what you need on the GPU.

	cudaMemcpyToSymbol(fo,&fou,sizeof(REAL));

	cudaMalloc((void **)&d_IC, sizeof(REAL)*dv);
	cudaMalloc((void **)&d_right, sizeof(REAL)*dv);
	cudaMalloc((void **)&d_left, sizeof(REAL)*dv);

	cudaMemcpy(d_IC,IC,sizeof(REAL)*dv,cudaMemcpyHostToDevice);

	// Some for loop

	upTriangle <<< blk,32 >>>(d_IC,d_right,d_left);


	downTriangle <<< blk,32 >>>(d_IC2,d_IC);

	// Some condition about when to stop or copy memory.



	// End loop and write out data.



	cudaFree(d_IC);
	cudaFree(d_IC2);
	cudaFree(d_coll);

	return 1;

}
