//I just really need to do these versions in one file before I really try to modularize
//the code.

//K-S involves no splitting.  And clearly a CPU diamond would be a waste.
//But it does need to pass it back and forth so it needs different passing versions.
//It would also be fruitful to test out streams on this one.
//The heat equation requires much more because of the boundary conditions.
//Also, we could test putting the whole thing on the GPU here.  So it never leaves shared memory.


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

const double PI = 3.141592653589793238463;

const float lx = 50.0;

__host__ __device__ REAL initFun(float xnode)
{
	REAL result;
	result.x = 2.f * cos(19.f*xnode*PI/128.f);
	result.y = -361.f/8192.f * cos(19.f*xnode*PI/128.f);
	return result;
}


__host__ __device__ REAL execFunc(REAL tLeft, REAL tRight, REAL tCenter)
{
	REAL tempC;
	REAL step; //step.x is conv .y is diff
	REAL tOut;
	//First full step. (Predictor)
	tempC.y = (tLeft.x + tRight.x - 2.f * tCenter.x) / (disc.x*disc.x);
	step.x = (tLeft.x*tLeft.x - tRight.x*tRight.x)/(4.f*disc.x);
	step.y = ((tLeft.x + tLeft.y) + (tRight.x + tRight.y) - 2.f*(tCenter.x+tempC.y))/(disc.x*disc.x);
	tempC.x = tCenter.x - 0.5 * disc.y * (step.y + step.x);

	//Second step (Corrector)
	tOut.y = (tLeft.x + tRight.x - 2.f * tempC.x) / (disc.x*disc.x);
	step.x = (tLeft.x*tLeft.x - tRight.x*tRight.x)/(4.f*disc.x);
	step.y = ((tLeft.x + tLeft.y) + (tRight.x + tRight.y) - 2.f*(tempC.x+tOut.y))/(disc.x*disc.x);
	tOut.x = tCenter.x - 0.5 * disc.y * (step.y + step.x);

	return tOut;

}

//-----------For testing --------------

//Up is the same.
__global__ void upTriangle(const REAL *IC, REAL * __restrict__ right, REAL * __restrict__ left)
{
	/*
	Initialize shared variables.  Each node (warp) will store 32 values on the
	right and left sides of their triangle, 2 on each side for each timeshLeftice.
	Since the base of the triangle is 32 numbers for each node, 16 timeshLeftices
	are evaluated per kernel call.
	Temper stores the temperatures at each timeshLeftice.  Since only the current
	and previous timeshLeftice results need to be held at each iteration.  This
	variable has 64 values, or two rows of 32, linearized.  The current and
	previous value alternate rows at each timeslice.
	*/
	extern __shared__ REAL share[];

	REAL *temper = (REAL*) share;
	REAL *shRight = (REAL*) &share[2*blockDim.x];
	REAL *shLeft = (REAL*) &share[3*blockDim.x];

	int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
	int tid = threadIdx.x; //Block Thread ID
	int tid1 = tid + 1;
	int tid2 = tid + 2;
	//int height = blockDim.x/2;
	int shft_wr; //Initialize the shift to the written row of temper.
	int shft_rd; //Initialize the shift to the read row (opposite of written)
	int logic_position;
	int itr = 0;
	int right_pick[4] = {2,1,-1,0}; //Template!
	//Assign the initial values to the first row in temper, each warp (in this
	//case each block) has it's own version of temper shared among its threads.
	temper[tid] = IC[gid];
	__syncthreads(); // Then make sure each block of threads are synced.

	for (int k = (blockDim.x-2); k>1 ; k-=2)
	{
		//Bitwise even odd. On even iterations write to first row.
		logic_position = (k/2 & 1);
		shft_wr = blockDim.x*logic_position;
		//On even iterations write to second row (starts at element 32)
		shft_rd = blockDim.x*((logic_position+1) & 1);

		//Each iteration the triangle narrows.  When k = 1, 30 points are
		//computed, k = 2, 28 points.
		if (tid <= k)
		{
			temper[tid + shft_wr] = execFunc(temper[tid+shft_rd], temper[tid2+shft_rd], temper[tid1+shft_rd]);
		}

		//Make sure the threads are synced
		__syncthreads();

		//Really tricky to get unique values with threads.
		if (shft_wr && tid < 4)
		{
			shLeft[tid+itr] = temper[(tid & 1) + (tid/2 * blockDim.x)]; // Still baroque.
			shRight[tid+itr] = temper[(right_pick[tid] + k) + (tid/2 * blockDim.x)];
			itr += 4;
		}

		__syncthreads();

	}

	//After the triangle has been computed, the right and left shared arrays are
	//stored in global memory by the global thread ID since (conveniently),
	//they're the same size as a warp!
	right[gid] = shRight[tid];
	left[gid] = shLeft[tid];

}

//The upside down triangle.  This function essentially takes right and left and
//returns IC.

//IMPORTANT note: k and tid were in sync in the first function, but here they're
//out of sync in the loop.  This is because we can't use tid = 33 or 32 and the
//shared temperature array is that long.  BUT in order to fill the arrays, these
//elements must be accessed.  So each element in each row is shifted by +1.
//For instance, thread tid = 16 refers to temper[17].  That being said, tid is
//unique and k is NOT so the index must be referenced by tid.

//
__global__
void
downTriangle(REAL *IC, REAL * __restrict__ right, REAL *__restrict__ left)
{

	extern __shared__ REAL share[];

	REAL *temper = (REAL *) share;
	REAL *shRight = (REAL *) &share[2*blockDim.x+4];
	REAL *shLeft = (REAL *) &share[3*blockDim.x+4];

	int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
	int tid = threadIdx.x; //Block Thread ID
    //Something like this.
    int gid_plus = (gid + blockDim.x) & (blockDim.x * gridDim.x - 1);

	int tid1 = tid + 1;
	int tid2 = tid + 2;
	//int height = blockDim.x/2;
	int shft_wr; //Initialize the shift to the written row of temper.
	int shft_rd; //Initialize the shift to the read row (opposite of written)
	int logic_position;
	int base = blockDim.x + 2;

	//Assign the initial values to the first row in temper, each warp (in this
	//case each block) has it's own version of temper shared among its threads.

	shLeft[tid] = right[gid];

	if (blockIdx.x == (gridDim.x-1))
	{
		shRight[tid] = left[tid];
	}
	else
	{
		shRight[tid] = left[gid+blockDim.x];
	}

	if (tid < 2)
	{
		temper[tid] = shLeft[tid];
		temper[tid2] = shRight[tid];
	}

    __syncthreads();

	for (int k = 2; k < blockDim.x; k+=2)
	{
		logic_position = (k/2 & 1);
		shft_wr = base*logic_position;
		//On even iterations write to second row (starts at element 32)
		shft_rd = base*((logic_position+1) & 1);

		if (tid < 2)
		{
			temper[tid + shft_wr] = shLeft[tid+k];
			temper[tid2 + k + shft_wr] = shRight[tid+k];
		}

		if (tid < k)
		{
			temper[tid2 + shft_wr] = execFunc(temper[tid + shft_rd], temper[tid2 + shft_rd], temper[tid1 + shft_rd]);
		}
        __syncthreads();
	}


    temper[tid] = execFunc(temper[tid+base], temper[tid2+base], temper[tid1+base]);

    IC[gid] = temper[tid];
}


__global__ void wholeDiamond(REAL * __restrict__ right, REAL * __restrict__ left, int swap)
{
    //Swap is either -1 or 1.  Since it either passes back or forward.
    extern __shared__ REAL share[];

	REAL *temper = (REAL*) share;
	REAL *shRight = (REAL*) &share[2*blockDim.x+4];
	REAL *shLeft = (REAL*) &share[3*blockDim.x+4];

    int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
	int tid = threadIdx.x; //Block Thread ID
    //Something like this.
    int gid_swap = (gid + swap*blockDim.x) & (blockDim.x * gridDim.x - 1);

	int tid1 = tid + 1;
	int tid2 = tid + 2;
	//int height = blockDim.x/2;
	int shft_wr; //Initialize the shift to the written row of temper.
	int shft_rd; //Initialize the shift to the read row (opposite of written)
	int logic_position;
	int right_pick[4] = {2,1,-1,0}; //Template!
	int itr = 0;
	int base = blockDim.x + 2;

	//int height = THREADBLK/2;
	//
	if (swap > 0)
	{

		shLeft[tid] = right[gid];
		shRight[tid] = left[gid_swap];

	}
	else
	{

		shRight[tid] = left[gid];
		shLeft[tid] = right[gid_swap];

	}

	if (tid < 2)
	{
		temper[tid] = shLeft[tid];
		temper[tid2] = shRight[tid];
	}


    for (int k = 2; k < blockDim.x; k+=2)
	{
		logic_position = (k/2 & 1);
		shft_wr = base*logic_position;
		//On even iterations write to second row (starts at element 32)
		shft_rd = base*((logic_position+1) & 1);

		if (tid < 2)
		{
			temper[tid + shft_wr] = shLeft[tid+k];
			temper[tid2 + k + shft_wr] = shRight[tid+k];
		}

		if (tid < k)
		{
			temper[tid2 + shft_wr] = execFunc(temper[tid + shft_rd], temper[tid2+shft_rd], temper[tid1 + shft_rd]);
		}
        __syncthreads();
	}

    temper[tid] = execFunc(temper[tid+base], temper[tid2+base], temper[tid1+base]);

    __syncthreads(); // Then make sure each block of threads are synced.

    //-------------------TOP PART------------------------------------------


	//The initial conditions are timeslice 0 so start k at 1.
	for (int k = (blockDim.x-2); k>1; k-=2)
	{
		//Bitwise even odd. On even iterations write to first row.
		logic_position = (k/2 & 1);
		shft_wr = base*logic_position;
		//On even iterations write to second row (starts at element 32)
		shft_rd = base*((logic_position+1) & 1);

		//Each iteration the triangle narrows.  When k = 1, 30 points are
		//computed, k = 2, 28 points.
		if (tid <= k)
		{
			temper[tid + shft_wr] = execFunc(temper[tid+shft_rd], temper[tid2+shft_rd], temper[tid1+shft_rd]);
		}

		//Make sure the threads are synced
		__syncthreads();

		//Really tricky to get unique values with threads.
		if (shft_wr && tid < 4)
		{
			shLeft[tid+itr] = temper[(tid & 1) + (tid/2 * base)]; // Still baroque.
			shRight[tid+itr] = temper[(right_pick[tid] + k) + (tid/2 * base)];
			itr += 4;
		}

		__syncthreads();

	}

	//After the triangle has been computed, the right and left shared arrays are
	//stored in global memory by the global thread ID since (conveniently),
	//they're the same size as a warp!
	right[gid] = shRight[tid];
	left[gid] = shLeft[tid];

}

//The host routine.
void sweptWrapper(const int bks, const int tpb, const int dv, REAL dt, const int t_end, REAL *IC, REAL *T_f)
{

	REAL *d_IC, *d_right, *d_left;
	cudaMalloc((void **)&d_IC, sizeof(REAL)*dv);
	cudaMalloc((void **)&d_right, sizeof(REAL)*dv);
	cudaMalloc((void **)&d_left, sizeof(REAL)*dv);

	// Copy the initial conditions to the device array.
	cudaMemcpy(d_IC,IC,sizeof(REAL)*dv,cudaMemcpyHostToDevice);
	// Start the counter and start the clock.
	const double t_fullstep = dt.y*(double)tpb;

	const size_t smem1 = 4*tpb*sizeof(REAL);
	const size_t smem2 = (4*tpb+4)*sizeof(REAL);

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

		/* Since the procedure does not store the temperature values, the user
		could input some time interval for which they want the temperature
		values and this loop could copy the values over from the device and
		write them out.  This way the user could see the progression of the
		solution over time, identify an area to be investigated and re-run a
		shorter version of the simulation starting with those intiial conditions.

		-------------------------------------
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

	//Conditions for main input.  Unit testing kinda.
	//dv and tpb must be powers of two.  dv must be larger than tpb and divisible by
	//tpb.

	//if ((dv & (tpb-1) !=0) || tpb&31 != 0)


	REAL dsc;
	dsc.x = lx/((float)dv-1.f);
	dsc.y = atof(argv[3]);

	// Initialize arrays.
	REAL IC[dv];
	REAL *IC_p;
	REAL T_final[dv];
	REAL *Tfin_p;

	Tfin_p = T_final;


	// Some initial condition for the bar temperature, an exponential decay
	// function.
	for (int k = 0; k<dv; k++)
	{
		IC[k] = initFun((float)k*dsc.x);
		cout << (float) k * dsc.x << " ";
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
	sweptWrapper(bks,tpb,dv,dsc,tf,IC_p,Tfin_p);

	// Show the time and write out the final condition.
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime( &timed, start, stop);

	timed = timed * 1.e-3;

	cout << "That took: " << timed << " seconds" << endl;
	if (tst)
	{
		ftime << dv << " " << tpb << " " << timed << endl;
		ftime.close();
	}

	fwr << tf << " ";
	for (int k = 0; k<dv; k++)
	{
		fwr << Tfin_p[k].x << " ";
	}

	fwr.close();

	// Free the memory and reset the device.

	cudaDeviceReset();
	cudaEventDestroy( start );
	cudaEventDestroy( start );

	return 0;

}
