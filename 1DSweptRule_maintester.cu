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

/*
Note that this code alters the original scheme. Paper available here:
http://www.sciencedirect.com/science/article/pii/S0021999115007664
The nodes never calculate a full diamond in a single kernel call and the boundary
values only get passed one direction, right.  This is a slightly simpler
application that passes the shared values in each node to the GPU global memory
more often.  This cuts down on some of the logic required in the full scheme and
makes results easier to output at various points in the solution.
*/

#include <cuda.h>
#include "cuda_runtime_api.h"
#include "device_functions.h"

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <ostream>
#include <cstring>
#include <fstream>

using namespace std;

// Define Given Parameters.  Material is aluminum.
//#define DIVISIONS  1024.
#define TS         .01
//#define ITERLIMIT  50000
#define REAL       float
#define TH_DIFF    8.418e-5
#ifndef FINISH
#define FINISH		1e4
#endif

//#define THREADBLK  32

// Declare constant Fourier number that will go in Device constant memory.
__constant__ REAL fo;

__global__ void upTriangle(REAL *IC, REAL *right, REAL *left)
{
	/*
	Initialize shared variables.  Each node (warp) will store 32 values on the
	right and left sides of their triangle, 2 on each side for each timeslice.
	Since the base of the triangle is 32 numbers for each node, 16 timeslices
	are evaluated per kernel call.
	Temper stores the temperatures at each timeslice.  Since only the current
	and previous timeslice results need to be held at each iteration.  This
	variable has 64 values, or two rows of 32, linearized.  The current and
	previous value alternate rows at each timeslice.
	*/
	__shared__ REAL temper[2*THREADBLK];
	__shared__ REAL sR[THREADBLK];
	__shared__ REAL sL[THREADBLK];

	int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
	int tid = threadIdx.x; //Warp or node ID
	int shft_wr; //Initialize the shift to the written row of temper.
	int shft_rd; //Initialize the shift to the read row (opposite of written)

	//Assign the initial values to the first row in temper, each warp (in this
	//case each block) has it's own version of temper shared among its threads.
	temper[tid] = IC[gid];
	__syncthreads(); // Then make sure each block of threads are synced.

	//This counter facilitates the transfer of the relevant values into the
	//right and left saved arrays.
	int itr = -1;

	//The initial conditions are timslice 0 so start k at 1.
	for (int k = 1; k<(THREADBLK/2); k++)
	{
		//Bitwise even odd. On even iterations write to first row.
		shft_wr = (k & 1);
		//On even iterations write to second row (starts at element 32)
		shft_rd = THREADBLK*((shft_wr+1) & 1);

		//Each iteration the triangle narrows.  When k = 1, 30 points are
		//computed, k = 2, 28 points.
		if (tid <= ((THREADBLK-1)-k) && tid >= k)
		{
			temper[tid + (THREADBLK*shft_wr)] = fo * (temper[tid+shft_rd-1] + temper[tid+shft_rd+1]) + (1.f-2.f*fo) * temper[tid+shft_rd];
		}

		//Make sure the threads are synced
		__syncthreads();

		//Now thread 0 in each block (which never computes a value) is used to
		//fill the shared right and left arrays with the relevant values.
		//This grabs the top and bottom edges on the iteration when the top
		//row is written.
		if (shft_wr && tid < 4)
		{
			sL[k+itr+tid] = temper[(tid/2*(THREADBLK-1))+(tid-1)+k];
			sR[k+itr+tid] = temper[((tid+2)/2*(THREADBLK-1))+(tid&1)-k];
			itr += 2;
		}

		__syncthreads();

	}


	//After the triangle has been computed, the right and left shared arrays are
	//stored in global memory by the global thread ID since (conveniently),
	//they're the same size as a warp!
	right[gid] = sR[tid];
	left[gid] = sL[tid];

}

//The upside down triangle.  This function essentially takes right and left and
//returns IC.

//IMPORTANT note: k and tid were in sync in the first function, but here they're
//out of sync in the loop.  This is because we can't use tid = 33 or 32 and the
//shared temperature array is that long.  BUT in order to fill the arrays, these
//elements must be accessed.  So each element in each row is shifted by +1.
//For instance, thread tid = 16 refers to temper[17].  That being said, tid is
//unique and k is NOT so the index must be referenced by tid.


// SPLIT
__global__ void downTriangle(REAL *IC, REAL *right, REAL *left)
{

	//Now temper needs to accommodate a longer row by 2, one on each side.
	//since it has two rows that's 4 extra floats.  The last row will still be
	//32 numbers long.
	int base = THREADBLK + 2;
	__shared__ REAL temper[(2*THREADBLK)+4];
	__shared__ REAL sR[THREADBLK];
	__shared__ REAL sL[THREADBLK];

	//Same as upTriangle
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	int tidp = tid + 1;
	int height = THREADBLK/2;
	int shft_rd;
	int shft_wr;

	// Pass to the left so all checks are for block 0 (this reduces arithmetic).
	// The left ridge is always kept by the block.
	sR[tid] = left[gid];

	// The right ridge is passed, each block 1-end gets the right of 0-end-1
	// Block 0 gets the right of the last block.
	if (blockIdx.x > 0)
	{
		sL[tid] = right[gid-blockDim.x];
	}
	else
	{
		sL[tid] = right[blockDim.x*(gridDim.x-1) + tid];
	}

	// Initialize temper. Kind of an unrolled for loop.  This is actually at
	// Timestep 0.
	if (tid < 2)
	{
		temper[tid+height-1] = sL[tid];
		temper[tidp+height] = sR[tid];
	}
	__syncthreads();
	//Now we need two counters since we need to use sL and sR EVERY iteration
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
			temper[tid+(k-2)+shft_wr*base] = sL[itr+tid];
			temper[tid+(base-k)+shft_wr*base] = sR[itr+tid];
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

__global__ void wholeDiamond(REAL *right, REAL *left)
{

	const int base = THREADBLK + 2;
	__shared__ REAL temper[2 * base];
	__shared__ REAL sR[THREADBLK];
	__shared__ REAL sL[THREADBLK];

	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	int tidp = tid + 1;
	int height = THREADBLK/2;
	int shft_rd;
	int shft_wr;

	sL[tid] = right[gid];

	// The right ridge is passed, each block 1-end gets the right of 0-end-1
	// Block 0 gets the right of the last block.
	if (blockIdx.x > 0)
	{
		sR[tid] = left[gid-blockDim.x];
	}
	else
	{
		sR[tid] = left[blockDim.x*(gridDim.x-1) + tid];
	}

	// Initialize temper. Kind of an unrolled for loop.  This is actually at
	// Timestep 0.

	if (tid < 2)
	{
		temper[tid+height-1] = sL[tid];
		temper[tidp+height] = sR[tid];
	}
	//Wind it up!

	int itr = 2;

	__syncthreads();

	for (int k = height; k>1; k--)
	{
		// This tells you if the current row is the first or second.
		shft_wr = ((k + 1) & 1);
		// Read and write are opposite rows.
		shft_rd = base*((shft_wr+1) & 1);
		//Block 0 is split so it needs a different algorithm.  This algorithm
		//is slightly different than top triangle as described in the note above.

		if (tid <= ((THREADBLK+1)-k) && tid >= k)
		{
			temper[tidp + ((base)*shft_wr)] = fo * (temper[tid+shft_rd] + temper[tid+shft_rd+2]) + (1.f-2.f*fo) * temper[tidp+shft_rd];
		}

		//Add the next values in.
		if (tid < 2)
		{
			temper[tid+(k-2)+shft_wr*base] = sL[itr+tid];
			temper[tidp+k+shft_wr*base] = sR[itr+tid];
			itr += 2;
		}

	}

	//DO THE MIDDLE ONE.
	if (blockIdx.x == (gridDim.x-1) && tid == 0 )
	{
		temper[2*base-1] = 0;
	}
	else if (blockIdx.x == 0 && tid == 0)
	{
		temper[base] = 0;
	}

	temper[tidp] = fo * (temper[tid+base] + temper[tid+base+2]) + (1.f-2.f*fo) * temper[tidp+base];

	itr = -1;

	//Wind it down!
	for (int k = 1; k<height; k++)
	{
		// This tells you if the current row is the first or second.
		shft_wr = (k & 1);
		// Read and write are opposite rows.
		shft_rd = base*((shft_wr+1) & 1);

		if (tid < (THREADBLK-k) && tid > k)
		{
			temper[tidp + (base*shft_wr)] = fo * (temper[tid+shft_rd] + temper[tid+shft_rd+2]) + (1.f-2.f*fo) * temper[tidp+shft_rd];
		}


	//Make sure the threads are synced
	__syncthreads();

	//Now thread 0 in each block (which never computes a value) is used to
	//fill the shared right and left arrays with the relevant values.
	//This grabs the top and bottom edges on the iteration when the top
	//row is written.
		if (shft_wr && tid < 4)
		{
			sL[k+itr+tid] = temper[(tid/2*(base-1))+(tid-1)+k];
			sR[k+itr+tid] = temper[(((tid/2)+1)*(base-1))+(tid&1)-k];
			itr += 2;
		}

	}
	__syncthreads();

	right[gid] = sR[tid];
	left[gid] = sL[tid];

}

//Split one is always first.  Passing left like the downTriangle.  downTriangle
//should be rewritten so it isn't split.  Only write on a non split pass.
__global__ void splitDiamond(REAL *right, REAL *left)
{

	const int base = THREADBLK + 2;
	__shared__ REAL temper[2 * base];
	__shared__ REAL sR[THREADBLK];
	__shared__ REAL sL[THREADBLK];

	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	int tidp = tid + 1;
	int height = THREADBLK/2;
	int shft_rd;
	int shft_wr;

	sR[tid] = left[gid];

	if (blockIdx.x > 0)
	{
		sL[tid] = right[gid-blockDim.x];
	}
	else
	{
		sL[tid] = right[blockDim.x*(gridDim.x-1) + tid];
	}

	// Initialize temper. Kind of an unrolled for loop.  This is actually at

	if (tid < 2)
	{
		temper[tid+height-1] = sL[tid];
		temper[tidp+height] = sR[tid];
	}
	//Wind it up!

	int itr = 2;

	__syncthreads();

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

		else
		{
			if (tidp <= ((THREADBLK+1)-k) && tidp >= k)
			{
				if (tid == (height-1))
				{
					temper[tidp + (base*shft_wr)] = 2.f * fo * (temper[tid+shft_rd]-temper[tid+shft_rd+1]) + temper[tidp+shft_rd];
				}
				else if (tid == height)
				{
					temper[tidp + (base*shft_wr)] = 2.f * fo * (temper[tid+shft_rd+2]-temper[tid+shft_rd+1]) + temper[tidp+shft_rd];
				}
				else
				{
					temper[tidp + (base*shft_wr)] = fo * (temper[tid+shft_rd] + temper[tid+shft_rd+2]) + (1.f-2.f*fo) * temper[tidp+shft_rd];
				}
			}
		}

		//Add the next values in.
		if (tid < 2)
		{
			temper[tid+(k-2)+shft_wr*base] = sL[itr+tid];
			temper[tid+(base-k)+shft_wr*base] = sR[itr+tid];
			itr += 2;
		}

	}

	__syncthreads();

	itr = -1;
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

	//Wind it down!
	for (int k = 1; k<height; k++)
	{
		// This tells you if the current row is the first or second.
		shft_wr = (k & 1);
		// Read and write are opposite rows.
		shft_rd = base*((shft_wr+1) & 1);
		//Block 0 is split so it needs a different algorithm.  This algorithm
		//is slightly different than top triangle as described in the note above.

		if (tid < (THREADBLK-k) && tid > k)
		{
			temper[tidp + (base*shft_wr)] = fo * (temper[tid+shft_rd] + temper[tid+shft_rd+2]) + (1.f-2.f*fo) * temper[tidp+shft_rd];
		}


	//Make sure the threads are synced
	__syncthreads();

	//Now thread 0 in each block (which never computes a value) is used to
	//fill the shared right and left arrays with the relevant values.
	//This grabs the top and bottom edges on the iteration when the top
	//row is written.
	if (shft_wr && tid < 4)
	{
		sL[k+itr+tid] = temper[(tid/2*(base-1))+(tid-1)+k];
		sR[k+itr+tid] = temper[(((tid/2)+1)*(base-1))+(tid&1)-k];
		itr += 2;
	}
	}
	__syncthreads();

	right[gid] = sR[tid];
	left[gid] = sL[tid];


}


//The host routine.
int main()
{
	//Choose the GPGPU.  This is device 0 in my machine which has 2 devices.
	cudaSetDevice(0);
	const int dv = int(DIVISIONS); //Setting it to an int helps with arrays
	const int bks = dv/THREADBLK; //The number of blocks since threads/block = 32.
	//Threads/block will be experimented on.
	const REAL lx = 5.0*DIVISIONS/1024;
	const REAL ds = lx/((double)DIVISIONS-1.0); //The x division length.
	REAL fou = TS*TH_DIFF/(ds*ds); //The Fourier number.

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
	ftime.open("1DSweptTiming.txt",ios::app);
	fwr.open("1DHeatEQResult.dat",ios::trunc);
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
	REAL t_eq = 0.;
	REAL t_fullstep = TS*(THREADBLK);
	double wall0 = clock();

	// Call the kernels until you reach the iteration limit.
	while(t_eq < FINISH)
	{

		upTriangle <<< bks,THREADBLK >>>(d_IC,d_right,d_left);

		downTriangle <<< bks,THREADBLK >>>(d_IC,d_right,d_left);


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
		cudaMemcpy(T_final, d_IC, sizeof(REAL)*dv, cudaMemcpyDeviceToHost);
		fwr << t_eq << " ";

		for (int k = 0; k<dv; k++)
		{
				fwr << T_final[k] << " ";
			}
			fwr << endl;
		}
		-------------------------------------
		*/


	}

	//Show the time and write out the final condition.
	double wall1 = clock();
	double timed = (wall1-wall0)/CLOCKS_PER_SEC;

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
