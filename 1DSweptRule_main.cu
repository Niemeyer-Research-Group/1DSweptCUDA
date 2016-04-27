
#include <cuda.h>
#include "cuda_runtime_api.h"
#include "device_functions.h"

#include <iostream>
#include <cmath>
#include <ostream>
#include <fstream>
#include <math.h>

using namespace std;

// Define Given Parameters
#define DIVISIONS  1024.
#define LENX       50.
#define TS         5.
#define TIMELIMIT  5000.
#define REAL       float
#define TH_DIFF    8.418e-5

__constant__ REAL fo;
__constant__ int tlen;


__global__ void upTriangle(REAL *IC, REAL *co)
{

	int tid = blockDim.x * blockIdx.x + threadIdx;
	// Some for loop to be unrolled.  Perhaps an if statment based on block size

	co[tid] = fo * (IC[tid-1] + IC[tid+1]) + (1-2.*fo) * IC[tid];



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
	int sz = int(DIVISIONS*DIVISIONS)/2

	REAL x[dv],
	REAL IC[dv];
	REAL *d_IC[dv];
	REAL *d_IC2[dv];
	REAL *d_coll[dv];

	for (int k = 0; k<dv; k++)
	{
		x[k] = ds*k;
		IC[k] = 500.f*expf((-x[k])/LENX);
	}

	cudaMemcpyToSymbol(fo,&fou,sizeof(REAL));
	cudaMemcpyToSymbol(tlen,&sz,sizeof(int));

	cudaMalloc((void **)&d_IC, sizeof(REAL)*dv);
	cudaMalloc((void **)&d_IC2, sizeof(REAL)*dv);
	cudaMalloc((void **)&d_coll, sizeof(REAL)*sz);

	cudaMemcpy(d_IC,IC,sizeof(REAL)*dv,cudaMemcpyHostToDevice);

	// Some for loop


	upTriangle <<< blk,32 >>>(d_IC,d_coll);


	downTriangle <<< blk,32 >>>(d_IC2,d_coll);

	// Some condition about when to stop or copy memory.



	// End loop and write out data.

	cudaFree(d_IC);
	cudaFree(d_IC2);
	cudaFree(d_coll);

	return 1;

}
