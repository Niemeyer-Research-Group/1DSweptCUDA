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

//COMPILE LINE!
// nvcc -o ./bin/TriTestOut 1DSweptRule_maintester.cu -gencode arch=compute_35,code=sm_35 -lm -w -std=c++11

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

#ifndef REAL
#define REAL       float
#endif

// Declare constant Fourier number that will go in Device constant memory.
__constant__ REAL fo;

const REAL th_diff = 8.418e-5;

__host__ REAL initFun(int xnode, REAL ds, REAL lx)
{

    return 500.f*expf((-ds*(REAL)xnode)/lx ) + 50.f*sinf(-ds*2.f*(REAL)xnode);

}

__device__ REAL execFunc(REAL tLeft, REAL tRight, REAL tCenter)
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
	int rightidx = (blockDim.x - 1) + ((tid/2 & 1) * blockDim.x) + (tid & 1) -  tid/2 - 1;

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

__global__ void downTriangle_GA(REAL *IC, REAL *right, REAL *left)
{

	//Now temper needs to accommodate a longer row by 2, one on each side.
	//since it has two rows that's 4 extra floats.  The last row will still be
	//32 numbers long.

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
	int leftidx = base/2 - tid/2 + ((tid/2 & 1) * base) + (tid & 1) - 2;
	int rightidx = base/2 + tid/2 + ((tid/2 & 1) * base) + (tid & 1);
	int gidout = (gid - blockDim.x/2) & ((blockDim.x*gridDim.x)-1);
	int gidin = (gid - blockDim.x) & ((blockDim.x*gridDim.x)-1);

	// Initialize temper. Kind of an unrolled for loop.  This is actually at
	// Timestep 0.

	temper[leftidx] = right[gidin];
	temper[rightidx] = left[gid];

	//k needs to insert the relevant left right values around the computed values
	//every timestep.  Since it grows larger the loop is reversed.
	for (int k = height-1; k>0; k--)
	{
		// This tells you if the current row is the first or second.
		shft_wr = base * ((k+1) & 1);
		// Read and write are opposite rows.
		shft_rd = base * (k & 1);

		//Block 0 is split so it needs a different algorithm.  This algorithm
		//is slightly different than top triangle as described in the note above.
		if (blockIdx.x > 0)
		{
			if (tid1 < (base-k) && tid1 >= k)
			{
				temper[tid1 + shft_wr] = execFunc(temper[tid+shft_rd], temper[tid2+shft_rd], temper[tid1+shft_rd]);
			}

		}


		else
		{
			if (tid1 < (base-k) && tid1 >= k)
			{
				if (tid1 == (height-1))
				{
					temper[tid1 + shft_wr] =execFunc(temper[tid+shft_rd], temper[tid+shft_rd], temper[tid1+shft_rd]);
				}
				else if (tid1 == height)
				{
					temper[tid1 + shft_wr] = execFunc(temper[tid2+shft_rd], temper[tid2+shft_rd], temper[tid1+shft_rd]);
				}
				else
				{
					temper[tid1 + shft_wr] = execFunc(temper[tid+shft_rd], temper[tid2+shft_rd], temper[tid1+shft_rd]);
				}
			}

		}

		__syncthreads();
	}

	IC[gidout] = temper[tid1];
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
	int shft_wr; //Initialize the shift to the written row of temper.
	int shft_rd; //Initialize the shift to the read row (opposite of written)
	int logic_position;
	int itr = 2;

	//Assign the initial values to the first row in temper, each warp (in this
	//case each block) has it's own version of temper shared among its threads.
	temper[tid] = IC[gid];
	__syncthreads(); // Then make sure each block of threads are synced.

    shLeft[0] = temper[0];
    shLeft[1] = temper[1];
    shRight[0] = temper[blockDim.x-1];
    shRight[1] = temper[blockDim.x-2];

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
			temper[tid + shft_wr] = execFunc(temper[tid+shft_rd], temper[tid2+shft_rd], temper[tid1+shft_rd]);
		}

		//Make sure the threads are synced
		__syncthreads();

		//Really tricky to get unique values with threads.
		if (tid < 2)
		{
			shLeft[tid+itr] = temper[tid+shft_wr]; // Still baroque.
			shRight[tid+itr] = temper[(k+tid-2) + shft_wr];
            itr +=2;

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
	int shft_wr; //Initialize the shift to the written row of temper.
	int shft_rd; //Initialize the shift to the read row (opposite of written)
    const int base = blockDim.x + 2;
	int logic_position;
	const int height = blockDim.x/2;
	int gidin = (gid - blockDim.x) & ((blockDim.x*gridDim.x)-1);
	int gidout = (gid - blockDim.x/2) & ((blockDim.x*gridDim.x)-1);

	shRight[tid] = left[gid];
	shLeft[tid] = right[gidin];

	// Initialize temper. Kind of an unrolled for loop.  This is actually at

	if (tid < 2)
	{
		temper[tid] = shLeft[tid];
		temper[tid2] = shRight[tid];
	}
	//Wind it up!

	__syncthreads();

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
	REAL *shRight = (REAL *) &share[2*blockDim.x];
	REAL *shLeft = (REAL *) &share[3*blockDim.x];

	int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
	int tid = threadIdx.x; //Warp or node ID
	int shft_wr; //Initialize the shift to the written row of temper.
	int shft_rd; //Initialize the shift to the read row (opposite of written)
    int tidp = tid + 1;
    int tidm = tid - 1;

	//Assign the initial values to the first row in temper, each warp (in this
	//case each block) has it's own version of temper shared among its threads.
	temper[tid] = IC[gid];
	__syncthreads(); // Then make sure each block of threads are synced.


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
			temper[tid + shft_wr] = execFunc(temper[tidm + shft_rd], temper[tidp+shft_rd], temper[tid+shft_rd]);
		}

		//Make sure the threads are synced
		__syncthreads();
		if (shft_wr && tid < 4)
		{
			shLeft[2*(k-1)+tid] = temper[(tid/2*(blockDim.x))+(tidm+k)-tid/2];
			shRight[2*(k-1)+tid] = temper[((tid+2)/2*(blockDim.x-1))+(tid&1)-k];

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


	//Same as upTriangle
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	int tid1 = tid + 1;
    int tid2 = tid + 2;

	int shft_rd;
	int shft_wr;

    const int base = blockDim.x + 2;
	const int height = blockDim.x/2;
	int gidin = (gid - blockDim.x) & ((blockDim.x*gridDim.x)-1);
	int gidout = (gid - blockDim.x/2) & ((blockDim.x*gridDim.x)-1);

	shRight[tid] = left[gid];
	shLeft[tid] = right[gidin];


	// Initialize temper. Kind of an unrolled for loop.  This is actually at
	// Timestep 0.
	if (tid < 2)
	{
		temper[tid+height-1] = shLeft[tid];
		temper[tid1+height] = shRight[tid];
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
		shft_wr = base*((k+1) & 1);
		// Read and write are opposite rows.
		shft_rd = base*(k & 1);

		//Block 0 is split so it needs a different algorithm.  This algorithm
		//is slightly different than top triangle as described in the note above.
		if (blockIdx.x > 0)
		{
			if ((tid1 <= ((blockDim.x+1)-k)) && tid1 >= k)
			{
				temper[tid1 + shft_wr] = execFunc(temper[tid+shft_rd], temper[tid2+shft_rd], temper[tid1+shft_rd]);
			}

		}

		//Split part.  This exhibits thread divergence and is suboptimal.
		//So it's ripe to be improved.

		else
		{
			if ((tid1 <= ((blockDim.x+1)-k)) && tid1 >= k)
			{
				if (tid1 == (height))
				{
					temper[tid1 + shft_wr] = execFunc(temper[tid+shft_rd], temper[tid+shft_rd], temper[tid1+shft_rd]);
				}
				else if (tid1 == (height+1))
				{
					temper[tid1 + shft_wr] = execFunc(temper[tid2+shft_rd], temper[tid2+shft_rd], temper[tid1+shft_rd]);
				}
				else
				{
					temper[tid1 + shft_wr] = execFunc(temper[tid+shft_rd], temper[tid2+shft_rd], temper[tid1+shft_rd]);
				}
			}

		}

		if (tid < 2)
		{
			temper[tid+(k-2)+shft_wr] = shLeft[itr+tid];
			temper[tid+(base-k)+shft_wr] = shRight[itr+tid];
			itr += 2;
		}

		__syncthreads();

	}

	if (blockIdx.x > 0)
	{
		temper[tid1] = execFunc(temper[tid+base], temper[tid2+base], temper[tid1+base]);
	}
	else
	{
		if (tid1 == (height))
		{
			temper[tid1] = execFunc(temper[tid+base], temper[tid+base], temper[tid1+base]);
		}
		else if (tid1 == (height+1))
		{
			temper[tid1] = execFunc(temper[tid2+base], temper[tid2+base], temper[tid1+base]);
		}
		else
		{
			temper[tid1] = execFunc(temper[tid+base], temper[tid2+base], temper[tid1+base]);
		}

	}

	//Now fill the global unified timestep variable with the final calculated
	//temperatures.

    IC[gidout] = temper[tid1];
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
    REAL fou = .05;
    REAL dt = atof(argv[3]);
    const int ALG = atoi(argv[5]);
    const REAL ds = sqrtf(dt*th_diff/fou);
    REAL lx = ds*((float)dv-1.f);

    cout << ALG << " " << dv << " " << tpb << endl;

	//Initialize arrays.
	REAL IC[dv];
	REAL T_final[dv];
	REAL *d_IC, *d_right, *d_left;

	// Some initial condition for the bar temperature, an exponential decay
	// function.
	for (int k = 0; k<dv; k++)
	{
		IC[k] = initFun(k, ds, lx);
	}

	//cout << fou << endl;
	// Call out the file before the loop and write out the initial condition.
	ofstream fwr;
	ofstream ftime;
	ftime.open("Results/TriangleAlgTestTime.txt",ios::app);
	fwr.open("Results/1DHeatEQResult.dat",ios::trunc);
	// Write out x length and then delta x and then delta t.
	// First item of each line is timestamp.
	fwr << lx << " " << dv << " " << dt << " " << endl << 0 << " ";

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

    size_t smem1u = 2*tpb*sizeof(REAL);
    size_t smem1d = (2*tpb+4)*sizeof(REAL);
    size_t smem2u = 4*tpb*sizeof(REAL);
    size_t smem2d = (4*tpb+4)*sizeof(REAL);

	// Call the kernels until you reach the iteration limit.
	if (ALG == 0)
	{
		while(t_eq < tf)
		{
			upTriangle_SI <<< bks,tpb,smem2u >>>(d_IC,d_right,d_left);
			downTriangle_SI <<< bks,tpb,smem2d >>>(d_IC,d_right,d_left);
			t_eq += t_fullstep;
		}
	}

	else if (ALG == 1)
	{
		while(t_eq < tf)
		{
			upTriangle_SRight <<< bks,tpb,smem2u >>>(d_IC,d_right,d_left);
			downTriangle_SRight <<< bks,tpb,smem2d >>>(d_IC,d_right,d_left);
			t_eq += t_fullstep;
		}
	}

	else
	{
		while(t_eq < tf)
		{
			upTriangle_GA <<< bks,tpb,smem1u >>>(d_IC,d_right,d_left);
			downTriangle_GA <<< bks,tpb,smem1d >>>(d_IC,d_right,d_left);
			t_eq += t_fullstep;
		}
	}
	//Show the time and write out the final condition.
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime( &timed, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    timed *= 1e-3;

	ftime << dv << " " << tpb << " " <<  timed << endl;
	cout << "That took: " << timed << " seconds" << endl;

    ftime.close();
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
