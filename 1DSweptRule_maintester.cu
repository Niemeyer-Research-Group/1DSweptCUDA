/* This file is the current iteration of research being done to implement the
swept rule for Partial differential equations in one dimension.  This research
is a collaborative effort between teams at MIT, Oregon State University, and
Purdue University.

Copyright (C) 2015 Kyle Niemeyer, niemeyek@oregonstate.edu AND
Daniel Magee, mageed@oregonstate.edu

This program is free software: you can redistribute it and/or modify
it under the terms of the MIT license.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

You should have received a copy of the MIT license
along with this program.  If not, see <https://opensource.org/licenses/MIT>.
*/

//This is to test different triangle version for shared memory.

#include <cuda.h>
#include "cuda_runtime_api.h"
#include "device_functions.h"

//NEW EDGE COLLECTION ALGORITHM!!!!

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ostream>
#include <cstring>
#include <fstream>

using namespace std;

//#define ITERLIMIT  50000
#ifndef REAL
#define REAL       float
#endif
// Declare constant Fourier number that will go in Device constant memory.
__constant__ REAL fo;

const REAL lx = 50.0;

const REAL th_diff = 8.418e-5;

_device__ REAL execFunc(REAL tLeft, REAL tRight, REAL tCenter)
{

    return fo*(tLeft+tRight) + (1.f-2.f*fo)*tCenter;

}

//FILLING POST LOOP AND GLOBAL ONLY EDGES.
__global__ void upTriangle_GA(REAL *IC, REAL *right, REAL *left)
{

	extern __shared__ REAL temper[];

	int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
	int tid = threadIdx.x; //Warp or node ID
	int tidp = tid + 1;
	int tidm = tid - 1;
	int shft_wr; //Initialize the shift to the written row of temper.
	int shft_rd; //Initialize the shift to the read row (opposite of written)
	int leftidx = tid/2 + ((tid/2 & 1) * blockDim.x) + (tid & 1);
	int rightidx = (blockDim.x - 1) + ((tid/2 & 1) * blockDim.x) + (tid & 1) -  tid/2;

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
		if (tid < ((blockDim.x-k) && tid => k)
		{
			temper[tid + shft_wr] = fo * (temper[tidm+shft_rd] + temper[tidp+shft_rd]) + (1.f-2.f*fo) * temper[tid+shft_rd];
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

__global__ void downTriangle_GA(REAL *IC, REAL *right, REAL *left)
{

	//Now temper needs to accommodate a longer row by 2, one on each side.
	//since it has two rows that's 4 extra floats.  The last row will still be
	//32 numbers long.

	extern __shared__ REAL temper[];
	int tidp = tid + 1;
	int tidm = tid - 1;

	//Same as upTriangle
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	int tidp = tid + 1;
	int tidm = tid - 1;
	int base = blockDim.x + 2;
	int height = base/2;
	int shft_rd;
	int shft_wr;
	int leftidx = base/2 - tid/2 ((tid/2 & 1) * base) + (tid & 1) - 2;
	int rightidx = base/2 + tid/2 ((tid/2 & 1) * base) + (tid & 1);
	int gidout = (gid - blockDim.x/2) & ((blockDim.x*gridDim.x)-1)
	int gidin = (gid - blockDim.x) & ((blockDim.x*gridDim.x)-1)

	// Pass to the left so all checks are for block 0 (this reduces arithmetic).
	// The left ridge is always kept by the block.

	// The right ridge is passed, each block 1-end gets the right of 0-end-1
	// Block 0 gets the right of the last block.


	// Initialize temper. Kind of an unrolled for loop.  This is actually at
	// Timestep 0.

	__syncthreads();
	temper[leftidx] = right[gidin]
	temper[rightidx] = left[gid]

	//k needs to insert the relevant left right values around the computed values
	//every timestep.  Since it grows larger the loop is reversed.
	for (int k = height-1 k>1; k--)
	{
		// This tells you if the current row is the first or second.
		shft_wr = base * ((k+1) & 1);
		// Read and write are opposite rows.
		shft_rd = base * ((k & 1);

		//Block 0 is split so it needs a different algorithm.  This algorithm
		//is slightly different than top triangle as described in the note above.
		if (blockIdx.x > 0)
		{
			if (tidp <= (blockDim.x-k) && tidp >= k)
			{
				temper[tidp + shft_wr)] = fo * (temper[tid+shft_rd] + temper[tid+shft_rd+2]) + (1.f-2.f*fo) * temper[tidp+shft_rd];
			}

		}

		//Split part.  This exhibits thread divergence and is suboptimal.
		//So it's ripe to be improved.

		else
		{
			if (tidp <= ((THREADBLK+1)-k) && tidp >= k)
			{
				if (tid == (height-1))
				{
					temper[tidp + (base*shft_wr)] = 2.f * fo * (temper[tid+shft_rd]-temper[tid+shft_rd+1]) + temper[tidp + shft_rd];
				}
				else if (tid == height)
				{
					temper[tidp + (base*shft_wr)] = 2.f * fo * (temper[tid+shft_rd+2]-temper[tid+shft_rd+1]) + temper[tidp + shft_rd];
				}
				else
				{
					temper[tidp + (base*shft_wr)] = fo * (temper[tid+shft_rd] + temper[tid+shft_rd+2]) + (1.f-2.f*fo) * temper[tidp +shft_rd];
				}
			}

		}

		__syncthreads();
	}

	IC[gidout] = temper[gid]

}

// MIDDLE CODE.  INSERTS EDGES IN LOOP BUT USES RIGHT TRIANGLES.
__global__ void upTriangle_SRight(REAL *IC, REAL *right, REAL *left)
{

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
	const int right_pick[4] = {2,1,-1,0}; //Template!

	//Assign the initial values to the first row in temper, each warp (in this
	//case each block) has it's own version of temper shared among its threads.
	temper[tid] = IC[gid];
	__syncthreads(); // Then make sure each block of threads are synced.

	//The initial conditions are timeslice 0 so start k at 1.
	for (int k = (blockDim.x-2); k>1; k-=2)
	{
		//Bitwise even odd. On even iterations write to first row.
		logic_position = (k/2 & 1);
		shft_wr = blockDim.x*logic_position;
		//On even iterations write to second row (starts at element 32)
		shft_rd = blockDim.x*((logic_position+1) & 1);

		//Each iteration the triangle narrows.  When k = 1, 30 points are
		//computed, k = 2, 28 points.
		if (tid < k)
		{
			temper[tid + shft_wr] = fo * (temper[tid+shft_rd] + temper[tid2+shft_rd]) + (1.f-2.f*fo) * temper[tid1+shft_rd];
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

	}

	//After the triangle has been computed, the right and left shared arrays are
	//stored in global memory by the global thread ID since (conveniently),
	//they're the same size as a warp!
	right[gid] = shRight[tid];
	left[gid] = shLeft[tid];

}

__global__ void downTriangle_SRight(REAL *IC, REAL *right, REAL *left)
{

	extern __shared__ REAL share[];

	REAL *temper = (REAL*) share;
	REAL *shRight = (REAL*) &share[2*blockDim.x+4];
	REAL *shLeft = (REAL*) &share[3*blockDim.x+4];

	int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
	int tid = threadIdx.x; //Block Thread ID
	int tid1 = tid + 1;
	int tid2 = tid + 2;
	//int height = blockDim.x/2;
	int shft_wr; //Initialize the shift to the written row of temper.
	int shft_rd; //Initialize the shift to the read row (opposite of written)
	int logic_position;
	const int right_pick[4] = {2,1,-1,0}; //Template!
	const int base = blockDim.x + 2;
	const int height = blockDim.x/2;
	int gidin = (gid - blockDim.x) & ((blockDim.x*gridDim.x)-1)
	int gidout = (gid - blockDim.x/2) & ((blockDim.x*gridDim.x)-1)

	shRight[tid] = left[gid];
	shLeft[tid] = right[gidin]

	// Initialize temper. Kind of an unrolled for loop.  This is actually at

	if (tid < 2)
	{
		temper[tid] = shLeft[tid];
		temper[tid2] = shRight[tid];
	}
	//Wind it up!

	for (int k = 2; k < blockDim.x; k+=2)
	{
		logic_position = (k/2 & 1);
		shft_wr = base * logic_position;
		//On even iterations write to second row (starts at element 32)
		shft_rd = base * ((logic_position+1) & 1);

		if (tid < 2)
		{
			temper[tid + shft_wr] = shLeft[tid+k];
			temper[tid2 + k + shft_wr] = shRight[tid+k];
		}

		if (tid < k)
		{
			if (blockIdx.x > 0)
			{
				temper[tid2 + shft_wr] = execFunc(temper[tid + shft_rd], temper[tid2+shft_rd], temper[tid1 + shft_rd]);
			}
			else
			{
				if (tid2 == (k/2+1))
				{
					temper[tid2 + shft_wr] = execFunc(temper[tid + shft_rd], temper[tid+shft_rd], temper[tid1 + shft_rd]);
				}
				else if (tid2 == (k/2+2))
				{
					temper[tid2 + shft_wr] = execFunc(temper[tid2 + shft_rd], temper[tid2+shft_rd], temper[tid1 + shft_rd]);
				}
				else
				{
					temper[tid2 + shft_wr] = execFunc(temper[tid + shft_rd], temper[tid2+shft_rd], temper[tid1 + shft_rd]);
				}
			}
		}
		__syncthreads();
	}

	if (gid == (height-1))
	{
		temper[tid] = execFunc(temper[tid+base], temper[tid+base], temper[tid1+base]);
	}
	else if (gid == height)
	{
		temper[tid] = execFunc(temper[tid2+base], temper[tid2+base], temper[tid1+base]);
	}
	else
	{
		temper[tid] = execFunc(temper[tid+base], temper[tid2+base], temper[tid1+base]);
	}

    IC[gidout] = temper[tid];
}

// OLDEST CODE.  GETS AND INSERTS EDGES INSIDE LOOP
__global__ void upTriangle_SI(REAL *IC, REAL *right, REAL *left)
{

	extern __shared__ REAL share[];

	REAL *temper = (REAL *) share;
	REAL *shRight = (REAL *) &share[2*blockDim.x+4];
	REAL *shLeft = (REAL *) &share[3*blockDim.x+4];

	int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
	int tid = threadIdx.x; //Warp or node ID
	int shft_wr; //Initialize the shift to the written row of temper.
	int shft_rd; //Initialize the shift to the read row (opposite of written)

	//Assign the initial values to the first row in temper, each warp (in this
	//case each block) has it's own version of temper shared among its threads.
	temper[tid] = IC[gid];
	__syncthreads(); // Then make sure each block of threads are synced.

	int itr = -1;

	//The initial conditions are timslice 0 so start k at 1.
	for (int k = 1; k<(blockDim.x/2); k++)
	{
		//Bitwise even odd. On even iterations write to first row.
		shft_wr = blockDim.x * (k & 1);
		//On even iterations write to second row (starts at element 32)
		shft_rd = blockDim.x * ((k + 1) & 1);

		//Each iteration the triangle narrows.  When k = 1, 30 points are
		//computed, k = 2, 28 points.
		if (tid < ((blockDim.x-k) && tid => k)
		{
			temper[tid + shft_wr] = fo * (temper[tidm+shft_rd] + temper[tidp+shft_rd]) + (1.f-2.f*fo) * temper[tid+shft_rd];
		}

		//Make sure the threads are synced
		__syncthreads();
		if (shft_wr && tid < 4)
		{
			shLeft[k+itr+tid] = temper[(tid/2*(THREADBLK-1))+(tid-1)+k];
			shRight[k+itr+tid] = temper[((tid+2)/2*(THREADBLK-1))+(tid&1)-k];
			itr += 2;
		}

		__syncthreads();

	}

	//After the triangle has been computed, the right and left shared arrays are
	//stored in global memory by the global thread ID since (conveniently),
	//they're the same size as a warp!
	right[gid] = shRight[tid];
	left[gid] = shLeft[tid];

}

__global__ void downTriangle_SI(REAL *IC, REAL *right, REAL *left)
{

	extern __shared__ REAL share[];

	REAL *temper = (REAL *) share;
	REAL *shRight = (REAL *) &share[2*blockDim.x+4];
	REAL *shLeft = (REAL *) &share[3*blockDim.x+4];

	int base = blockDim.x + 2;

	//Same as upTriangle
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	int tidp = tid + 1;
	int height = blockDim.x/2;
	int shft_rd;
	int shft_wr;

	// Pass to the left so all checks are for block 0 (this reduces arithmetic).
	// The left ridge is always kept by the block.
	shRight[tid] = left[gid];

	// The right ridge is passed, each block 1-end gets the right of 0-end-1
	// Block 0 gets the right of the last block.
	if (blockIdx.x > 0)
	{
		shLeft[tid] = right[gid-blockDim.x];
	}
	else
	{
		shLeft[tid] = right[blockDim.x*(gridDim.x-1) + tid];
	}

	// Initialize temper. Kind of an unrolled for loop.  This is actually at
	// Timestep 0.
	if (tid < 2)
	{
		temper[tid+height-1] = shLeft[tid];
		temper[tidp+height] = shRight[tid];
	}
	__syncthreads();
	//Now we need two counters since we need to use shLeft and shRight EVERY iteration
	//instead of every other iteration and instead of growing smaller with every
	//iteration this grows larger.
	int itr = 2;

	//k needs to insert the relevant left right values around the computed values
	//every timestep.  Since it grows larger the loop is reversed.
	for (int k = height; k>1; k--)
	{
		// This tells you if the current row is the first or second.
		shft_wr = ((k+1) & 1);
		// Read and write are opposite rows.
		shft_rd = base*((shft_wr+1) & 1);

		//Block 0 is split so it needs a different algorithm.  This algorithm
		//is slightly different than top triangle as described in the note above.
		if (blockIdx.x > 0)
		{
			if (tidp <= ((THREADBLK+1)-k) && tidp >= k)
			{
				temper[tidp + (base*shft_wr)] = fo * (temper[tid+shft_rd] + temper[tid+shft_rd+2]) + (1.f-2.f*fo) * temper[tidp+shft_rd];
			}

		}

		//Split part.  This exhibits thread divergence and is suboptimal.
		//So it's ripe to be improved.

		else
		{
			if (tidp <= ((THREADBLK+1)-k) && tidp >= k)
			{
				if (tid == (height-1))
				{
					temper[tidp + (base*shft_wr)] = 2.f * fo * (temper[tid+shft_rd]-temper[tid+shft_rd+1]) + temper[tidp + shft_rd];
				}
				else if (tid == height)
				{
					temper[tidp + (base*shft_wr)] = 2.f * fo * (temper[tid+shft_rd+2]-temper[tid+shft_rd+1]) + temper[tidp + shft_rd];
				}
				else
				{
					temper[tidp + (base*shft_wr)] = fo * (temper[tid+shft_rd] + temper[tid+shft_rd+2]) + (1.f-2.f*fo) * temper[tidp +shft_rd];
				}
			}

		}

		if (tid < 2)
		{
			temper[tid+(k-2)+shft_wr*base] = shLeft[itr+tid];
			temper[tid+(base-k)+shft_wr*base] = shRight[itr+tid];
			itr += 2;
		}

		__syncthreads();

	}

	if (blockIdx.x > 0)
	{
		temper[tidp] = fo * (temper[tid+base] + temper[tid+base+2]) + (1.f-2.f*fo) * temper[tidp+base];
	}
	else
	{
		if (tid == (height-1))
		{
			temper[tidp] = 2.f * fo * (temper[tid+base]-temper[tid+base+1]) + temper[tidp+base];
		}
		else if (tid == height)
		{
			temper[tidp] = 2.f * fo * (temper[tid+base+2]-temper[tid+base+1]) + temper[tidp+base];
		}
		else
		{
			temper[tidp] = fo * (temper[tid+base] + temper[tid+base+2]) + (1.f-2.f*fo) * temper[tidp+base];
		}

	}

	//Now fill the global unified timestep variable with the final calculated
	//temperatures.

	//Blocks 1 to end hold values 16 to end-16.
	if (blockIdx.x > 0)
	{
		//True if it ends on the first row! The first and last of temper on the final row are empty.
		IC[gid - height] = temper[tidp];
	}
	//Block 0 holds values 0 to 15 and end-15 to end.  In that order.
	else
	{
		if (tid >= height)
		{
			IC[gid - (height)] = temper[tidp];
		}
		else
		{
			IC[(blockDim.x * gridDim.x) + (tid - height) ] = temper[tidp];
		}
	}
}

//The host routine.
int main( int argc, char *argv[])
{
	if (argc != 6)
	{
		cout << "The Program takes five inputs, #Divisions, #Threads/block, dt, finish time, and which algorithm. " << endl;
		exit(-1);
	}

	// Choose the GPGPU.  This is device 0 in my machine which has 2 devices.
	cudaSetDevice(0);

	int dv = atoi(argv[1]); //Setting it to an int helps with arrays
	const int tpb = atoi(argv[2]);
	const int tf = atoi(argv[4]);
	const int bks = dv/tpb; //The number of blocks since threads/block = 32.
    const REAL ds = lx/((float)dv-1.f);
    REAL dt = atof(argv[3]);
    const int ALG = atoi(argv[5])
    REAL fou = dt*th_diff/(ds*ds);

	//Initialize arrays.
	REAL IC[dv];
	REAL T_final[dv];
	REAL *d_IC, *d_right, *d_left;

	// Some initial condition for the bar temperature, an exponential decay
	// function.
	for (int k = 0; k<dv; k++)
	{
		IC[k] = 500.f*expf((-ds*k)/lx);
	}

	//cout << fou << endl;
	// Call out the file before the loop and write out the initial condition.
	ofstream fwr;
	ofstream ftime;
	ftime.open("Result/TriangleAlgTestTime.txt",ios::app);
	fwr.open("Result/1DHeatEQResult.dat",ios::trunc);
	// Write out x length and then delta x and then delta t.
	// First item of each line is timestamp.
	fwr << lx << " " << DIVISIONS << " " << TS << " " << endl << 0 << " ";

	for (int k = 0; k<dv; k++)
	{
		fwr << IC[k] << " ";
	}

	fwr << endl;

	// Transfer data to GPU.

	// This puts the Fourier number in constant memory.
	cudaMemcpyToSymbol(fo,&fou,sizeof(REAL));

	// This initializes the device arrays on the device in global memory.
	// They're all the same size.  Conveniently.
	cudaMalloc((void **)&d_IC, sizeof(REAL)*dv);
	cudaMalloc((void **)&d_right, sizeof(REAL)*dv);
	cudaMalloc((void **)&d_left, sizeof(REAL)*dv);

	//Copy the initial conditions to the device array.
	cudaMemcpy(d_IC,IC,sizeof(REAL)*dv,cudaMemcpyHostToDevice);

	// Start the counter and start the clock.
	REAL t_eq = 0.f;
	REAL t_fullstep = dt*tpb;
	cudaEvent_t start, stop;
	float timed;
	cudaEventCreate( &start );
	cudaEventCreate( &stop );
	cudaEventRecord( start, 0);

	// Call the kernels until you reach the iteration limit.
	if (ALG == 0)
	{
		while(t_eq < tf)
		{
			upTriangle_SI <<< bks,THREADBLK >>>(d_IC,d_right,d_left);
			downTriangle_SI <<< bks,THREADBLK >>>(d_IC,d_right,d_left);
			t_eq += t_fullstep;
		}
	}

	else if (ALG == 1)
	{
		while(t_eq < tf)
		{
			upTriangle_SRight <<< bks,THREADBLK >>>(d_IC,d_right,d_left);
			downTriangle_SRight <<< bks,THREADBLK >>>(d_IC,d_right,d_left);
			t_eq += t_fullstep;
		}
	}

	else
	{
		while(t_eq < tf)
		{
			upTriangle_GA <<< bks,THREADBLK >>>(d_IC,d_right,d_left);
			downTriangle_GA <<< bks,THREADBLK >>>(d_IC,d_right,d_left);
			t_eq += t_fullstep;
		}
	}
	//Show the time and write out the final condition.
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime( &timed, start, stop);

	ftime << timed << endl;
	cout << "That took: " << timed << " seconds" << endl;


	cudaMemcpy(T_final, d_IC, sizeof(REAL)*dv, cudaMemcpyDeviceToHost);
	fwr << t_eq << " ";
	for (int k = 0; k<dv; k++)
	{
		fwr << T_final[k] << " ";
	}

	fwr.close();

	//Free the memory and reset the device.
	cudaFree(d_IC);
	cudaFree(d_right);
	cudaFree(d_left);
	cudaDeviceReset();

	return 0;
}
