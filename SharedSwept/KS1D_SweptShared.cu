//I just really need to do these versions in one file before I really try to modularize
//the code.

//K-S involves no splitting.  And clearly a CPU diamond would be a waste.
//But it does need to pass it back and forth so it needs different passing versions.
//It would also be fruitful to test out streams on this one.
//The heat equation requires much more because of the boundary conditions.
//Also, we could test putting the whole thing on the GPU here.  So it never leaves shared memory.

//COMPILE LINE!
// nvcc -o ./bin/KSOut KS1D_SweptShared.cu -gencode arch=compute_35,code=sm_35 -lm -w -std=c++11

//RUN LINE!
// ./bin/KSOut 256 2048 .01 10 0

//DO:
//Use malloc or cudaHostAlloc for all initial and final conditions.
//Write the ability to pull out time snapshots.
//The ability to feed in initial conditions.
//Input paths to output as argc?  Input?
//Ask about just making up my own conventions.  Like dat vs txt.  Use that as flag?

#include <cuda.h>
#include "cuda_runtime_api.h"
#include "device_functions.h"

#include <iostream>
#include <ostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>


#ifndef REAL
#define REAL  float2
#endif

//-----------For testing --------------

using namespace std;

// disc.x is dx, disc.y is dt.
__constant__ REAL disc;

const float dx = .5;

__host__ __device__ REAL initFun(float xnode)
{
	REAL result;
	result.x = 2.f * cos(19.f*xnode*M_PI/128.f);
	result.y = -361.f/4096.f*M_PI * cos(19.f*xnode*M_PI/128.f);
	return result;
}


__device__ REAL execFunc(REAL tLeft, REAL tRight, REAL tCenter)
{
	REAL tempC;
	REAL step; //step.x is conv .y is diff
	REAL tOut;
	//First full step. (Predictor)
	tempC.y = (tLeft.x + tRight.x - 2.f * tCenter.x) / (disc.x*disc.x);
	step.x = (tLeft.x*tLeft.x - tRight.x*tRight.x)/(4.f*disc.x);
	step.y = ((tLeft.x + tLeft.y) + (tRight.x + tRight.y) - 2.f*(tCenter.x+tempC.y))/(disc.x*disc.x);
	tempC.x = tCenter.x - 0.5f * disc.y * (step.y + step.x);

	//Second step (Corrector)
	tOut.y = (tLeft.x + tRight.x - 2.f * tempC.x) / (disc.x*disc.x);
	step.x = (tLeft.x*tLeft.x - tRight.x*tRight.x)/(4.f*disc.x);
	step.y = ((tLeft.x + tLeft.y) + (tRight.x + tRight.y) - 2.f*(tempC.x+tOut.y))/(disc.x*disc.x);
	tOut.x = tCenter.x - 0.5f * disc.y * (step.y + step.x);

	return tOut;

}

//-----------For testing --------------

//Up is the same.
__global__
void
upTriangle(REAL *IC, REAL *right, REAL *left)
{

	extern __shared__ REAL temper[];

	int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
	int tid = threadIdx.x; //Block Thread ID
    int tidp = tid + 1;
	int tidm = tid - 1;
	int shft_wr; //Initialize the shift to the written row of temper.
	int shft_rd; //Initialize the shift to the read row (opposite of written)
	int leftidx = tid/2 + ((tid/2 & 1) * blockDim.x) + (tid & 1);
	int rightidx = (blockDim.x - 2) + ((tid/2 & 1) * blockDim.x) + (tid & 1) -  tid/2;

    //Assign the initial values to the first row in temper, each warp (in this
	//case each block) has it's own version of temper shared among its threads.
	temper[tid] = IC[gid];

	//The initial conditions are timslice 0 so start k at 1.
	for (int k = 1; k<(blockDim.x/2); k++)
	{
		//Bitwise even odd. On even iterations write to first row.
		shft_wr = blockDim.x * (k & 1);
		//On even iterations write to second row (starts at element 32)
		shft_rd = blockDim.x * ((k + 1) & 1);

		//Each iteration the triangle narrows.  When k = 1, 30 points are
		//computed, k = 2, 28 points.
		if (tid < (blockDim.x-k) && tid >= k)
		{
			temper[tid + shft_wr] = execFunc(temper[tidm+shft_rd], temper[tidp+shft_rd], temper[tid+shft_rd]);
		}

		//Make sure the threads are synced
		__syncthreads();

	}

	//After the triangle has been computed, the right and left shared arrays are
	//stored in global memory by the global thread ID since (conveniently),
	//they're the same size as a warp!
	right[gid] = temper[rightidx];
	left[gid] = temper[leftidx];

}

__global__
void
downTriangle(REAL *IC, REAL *right, REAL *left)
{
	extern __shared__ REAL temper[];

	//Same as upTriangle
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	int tid1 = tid + 1;
	int tid2 = tid + 2;
	int base = blockDim.x + 2;
	int height = base/2;
	int shft_rd;
	int shft_wr;
	int leftidx = height - tid/2 + ((tid/2 & 1) * base) + (tid & 1) - 2;
	int rightidx = height + tid/2 + ((tid/2 & 1) * base) + (tid & 1);
	int gidin = (gid + blockDim.x) & ((blockDim.x*gridDim.x)-1);

	// Initialize temper. Kind of an unrolled for loop.  This is actually at
	// Timestep 0.

	temper[leftidx] = right[gid];
	temper[rightidx] = left[gidin];

    //k needs to insert the relevant left right values around the computed values
	//every timestep.  Since it grows larger the loop is reversed.
	for (int k = height-1; k>0; k--)
	{
		// This tells you if the current row is the first or second.
		shft_wr = base * ((k+1) & 1);
		// Read and write are opposite rows.
		shft_rd = base * (k & 1);

		if (tid1 < (base-k) && tid1 >= k)
		{
			temper[tid1 + shft_wr] = execFunc(temper[tid+shft_rd], temper[tid2+shft_rd], temper[tid1+shft_rd]);
		}
        __syncthreads();
	}

    IC[gid] = temper[tid1];
}

// Pass to split is false.  Pass to whole is true.
__global__
void
wholeDiamond(REAL *right, REAL *left, int pass)
{
    extern __shared__ REAL temper[];

	//Same as downTriangle.
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
    int lastidx = ((blockDim.x*gridDim.x)-1);
	int tid1 = tid + 1;
	int tid2 = tid + 2;
	int base = blockDim.x + 2;
	int height = base/2;
	int shft_rd;
	int shft_wr;
	int leftidx = height - tid/2 + ((tid/2 & 1) * base) + (tid & 1) - 2;
	int rightidx = height + tid/2 + ((tid/2 & 1) * base) + (tid & 1);
	int gidin = (gid + pass*blockDim.x) & lastidx;

	if (pass < 0)
	{
    	temper[leftidx] = right[gidin];
    	temper[rightidx] = left[gid];
	}
	else
	{
		temper[leftidx] = right[gid];
		temper[rightidx] = left[gidin];
	}


	for (int k = (height-1); k>0; k--)
	{
        // This tells you if the current row is the first or second.
		shft_wr = base * ((k + 1) & 1);
		// Read and write are opposite rows.
		shft_rd = base * (k & 1);

        if (tid1 < (base-k) && tid1 >= k)
		{
			temper[tid1 + shft_wr] = execFunc(temper[tid+shft_rd], temper[tid2+shft_rd], temper[tid1+shft_rd]);
		}
        __syncthreads();
	}

	//Shift the last row to justify it at 0.
	temper[tid] = temper[tid1];
    //-------------------TOP PART------------------------------------------

    leftidx = tid/2 + ((tid/2 & 1) * base) + (tid & 1);
    rightidx = (base - 4) + ((tid/2 & 1) * base) + (tid & 1) -  tid/2;

    int tidm = tid - 1;

	//The initial conditions are timeslice 0 so start k at 1.
    for (int k = 1; k<(height-1); k++)
	{
		//Bitwise even odd. On even iterations write to first row.
		shft_wr = base * (k & 1);
		//On even iterations write to second row (starts at element 32)
		shft_rd = base * ((k + 1) & 1);

		//Each iteration the triangle narrows.  When k = 1, 30 points are
		//computed, k = 2, 28 points.
		if (tid < (blockDim.x-k) && tid >= k)
		{
			temper[tid + shft_wr] = execFunc(temper[tidm+shft_rd], temper[tid1+shft_rd], temper[tid+shft_rd]);
		}

		//Make sure the threads are synced
		__syncthreads();

	}

    right[gid] = temper[rightidx];
	left[gid] = temper[leftidx];

}

//The host routine.
double
sweptWrapper(const int bks, int tpb, const int dv, REAL dt, const int t_end, REAL *IC, REAL *T_f)
{

	REAL *d_IC, *d_right, *d_left;
	cudaMalloc((void **)&d_IC, sizeof(REAL)*dv);
	cudaMalloc((void **)&d_right, sizeof(REAL)*dv);
	cudaMalloc((void **)&d_left, sizeof(REAL)*dv);

	// Copy the initial conditions to the device array.
	cudaMemcpy(d_IC,IC,sizeof(REAL)*dv,cudaMemcpyHostToDevice);
	// Start the counter and start the clock.
	const double t_fullstep = dt.y*(double)tpb;

	const size_t smem1 = 2*tpb*sizeof(REAL);
	const size_t smem2 = (2*tpb+4)*sizeof(REAL);

	upTriangle <<< bks,tpb,smem1 >>>(d_IC,d_right,d_left);
	wholeDiamond <<< bks,tpb,smem2 >>>(d_right,d_left,-1);
	double t_eq = t_fullstep;

	// Call the kernels until you reach the iteration limit.
	while(t_eq < t_end)
	{

		wholeDiamond <<< bks,tpb,smem2 >>>(d_right,d_left,1);
		//So it always ends on a left pass since the down triangle is a right pass.
		wholeDiamond <<< bks,tpb,smem2 >>>(d_right,d_left,-1);

		t_eq += t_fullstep;

		/*
	 	if (true)
		{
			downTriangle <<< bks,tpb,smem2 >>>(d_IC,d_right,d_left);
			cudaMemcpy(T_final, d_IC, sizeof(REAL)*dv, cudaMemcpyDeviceToHost);
			fwr << t_eq << " ";

			for (int k = 0; k<dv; k++)
			{
					fwr << T_final.x[k] << " ";
			}
				fwr << endl;

			upTriangle <<< bks,tpb,smem1 >>>(d_IC,d_right,d_left);
			wholeDiamond <<< bks,tpb,smem2 >>>(d_right,d_left,-1);
		}
		-------------------------------------
		*/
	}

	downTriangle <<< bks,tpb,smem2 >>>(d_IC,d_right,d_left);

	cudaMemcpy(T_f, d_IC, sizeof(REAL)*dv, cudaMemcpyDeviceToHost);

	cudaFree(d_IC);
	cudaFree(d_right);
	cudaFree(d_left);

	return t_eq;

}

int main( int argc, char *argv[])
{

	if (argc != 6)
	{
		cout << "The Program takes four inputs, #Divisions, #Threads/block, dt, finish time, and test or not" << endl;
		exit(-1);
	}

	// Choose the GPGPU.  This is device 0 in my machine which has 2 devices.
	cudaSetDevice(0);

	int dv = atoi(argv[1]); //Setting it to an int helps with arrays
	const int tpb = atoi(argv[2]);
	const int tf = atoi(argv[4]);
	const int bks = dv/tpb; //The number of blocks since threads/block = 32.
	const int tst = atoi(argv[5]);

	const float lx = dv*dx;

	//Conditions for main input.  Unit testing kinda.
	//dv and tpb must be powers of two.  dv must be larger than tpb and divisible by
	//tpb.

	//if ((dv & (tpb-1) !=0) || tpb&31 != 0)

	REAL dsc;
	dsc.x = dx;
	dsc.y = atof(argv[3]);

	// Initialize arrays.  Should malloc instead of stack.
	REAL IC[dv];
	REAL *IC_p;
	REAL T_final[dv];
	REAL *Tfin_p;

	Tfin_p = T_final;

	// Some initial condition for the bar temperature, an exponential decay
	// function.
	for (int k = 0; k<dv; k++)
	{
		IC[k] = initFun((float)k*dsc.x/2);
	}

	// Call out the file before the loop and write out the initial condition.
	ofstream fwr, ftime;
	fwr.open("Results/KS1D_Result.dat",ios::trunc);
	if (tst) ftime.open("Results/KS1D_Timing.txt",ios::app);
	// Write out x length and then delta x and then delta t.
	// First item of each line is timestamp.
	fwr << lx << " " << dv << " " << dsc.x << " " << endl << 0 << " ";

	for (int k = 0; k<dv; k++)
	{
		fwr << IC[k].x << " ";
	}

	fwr << endl;

	IC_p = IC;

	// Transfer data to GPU.

	// This puts the Fourier number in constant memory.
	cudaMemcpyToSymbol(disc,&dsc,sizeof(REAL));

	// This initializes the device arrays on the device in global memory.
	// They're all the same size.  Conveniently.

	// Start the counter and start the clock.
	cudaEvent_t start, stop;
	float timed;
	cudaEventCreate( &start );
	cudaEventCreate( &stop );
	cudaEventRecord( start, 0);

	// Call the kernels until you reach the iteration limit.
	double tfm;

	tfm = sweptWrapper(bks,tpb,dv,dsc,tf,IC_p,Tfin_p);

	cout << dsc.x << " " << dsc.x*dsc.x << endl;

	// Show the time and write out the final condition.
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime( &timed, start, stop);

	timed *= 1.e-3;

	cout << "That took: " << timed << " seconds" << endl;
	if (tst)
	{
		ftime << dv << " " << tpb << " " << timed << endl;
		ftime.close();
	}

	fwr << tfm << " ";
	for (int k = 0; k<dv; k++)
	{
		fwr << Tfin_p[k].x << " ";
	}

	fwr.close();

	// Free the memory and reset the device.
	cudaDeviceReset();
	cudaEventDestroy( start );
	cudaEventDestroy( stop );

	return 0;

}
