
#include "SwR_1DShared.cuh"
#include <cstdio>
#include <cstdilib>
#include <cuda.h>

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

__global__ void downTriangle(REAL *IC, REAL *right, REAL *left)
{

	//Now temper needs to accommodate a longer row by 2, one on each side.
	//since it has two rows that's 4 extra floats.  The last row will still be
	//32 numbers long.
	__shared__ REAL temper[(2*THREADBLK)+4];
	__shared__ REAL sR[THREADBLK];
	__shared__ REAL sL[THREADBLK];

	//Same as upTriangle
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	int shft_rd;
	int shft_wr;

	// Pass to the left so all checks are for block 0 (this reduces arithmetic).
	// The left ridge is always kept by the block.
	sL[tid] = right[gid];

	// The right ridge is passed, each block 1-end gets the right of 0-end-1
	// Block 0 gets the right of the last block.
	if (blockIdx.x == (gridDim.x-1))
	{
		sR[tid] = left[tid];
	}
	else
	{
		sR[tid] = left[gid+blockDim.x];
	}

	// Initialize temper. Kind of an unrolled for loop.  This is actually at
	// Timestep 0.
	if (tid < 2)
	{
		temper[tid+THREADBLK/2-1] = sL[tid];
		temper[tid+THREADBLK/2+1] = sR[tid];
	}

	//Now we need two counters since we need to use sL and sR EVERY iteration
	//instead of every other iteration and instead of growing smaller with every
	//iteration this grows larger.
	int itr = 2;
	int itr2 = THREADBLK/2+2;
	//k needs to insert the relevant left right values around the computed values
	//every timestep.  Since it grows larger the loop is reversed.
	for (int k = THREADBLK/2+1; k>1; k--)
	{
		// This tells you if the current row is the first or second.
		shft_wr = (k & 1);
		// Read and write are opposite rows.
		shft_rd = (THREADBLK+2)*((shft_wr+1) & 1);

		//Block 0 is split so it needs a different algorithm.  This algorithm
		//is slightly different than top triangle as described in the note above.
		if (blockIdx.x > 0)
		{
			if (tid <= ((THREADBLK+1)-k) && tid >= (k-2))
			{
				temper[tid + 1 + ((THREADBLK+2)*shft_wr)] = fo * (temper[tid+shft_rd] + temper[tid+shft_rd+2]) + (1.f-2.f*fo) * temper[tid+shft_rd+1];
			}

		}

		//Split part.  This exhibits thread divergence and is suboptimal.
		//So it's ripe to be improved.

		else
		{
			if (tid <= ((THREADBLK+1)-k) && tid >= (k-2))
			{
				if (tid == (THREADBLK/2-1))
				{
					temper[tid + 1 + ((THREADBLK+2)*shft_wr)] = 2.f * fo * (temper[tid+shft_rd]-temper[tid+shft_rd+1]) + temper[tid+shft_rd+1];
				}
				else if (tid == THREADBLK/2)
				{
					temper[tid + 1 + ((THREADBLK+2)*shft_wr)] = 2.f * fo * (temper[tid+shft_rd+2]-temper[tid+shft_rd+1]) + temper[tid+shft_rd+1];
				}
				else
				{
					temper[tid + 1 + ((THREADBLK+2)*shft_wr)] = fo * (temper[tid+shft_rd] + temper[tid+shft_rd+2]) + (1.f-2.f*fo) * temper[tid+shft_rd+1];
				}
			}

		}

		//Fill edges.  Thread 0 never gets used for both operations so the
		//calculation and the filling are conceptually coincident.
		//Threads are synced afterward here because the next timestep is
		//reliant on the entire loop.
		if (k>2 && tid == 0)
		{
			temper[(k-3)+((THREADBLK+2)*shft_wr)] = sL[itr];
			temper[(k-2)+((THREADBLK+2)*shft_wr)] = sL[itr+1];
			temper[itr2+((THREADBLK+2)*shft_wr)] = sR[itr];
			itr2++;
			temper[itr2+((THREADBLK+2)*shft_wr)] = sR[itr+1];
			itr+=2;

		}
		__syncthreads();

	}

	//Now fill the global unified timestep variable with the final calculated
	//temperatures.

	//Blocks 1 to end hold values 16 to end-16.
	if (blockIdx.x > 0)
	{
		//True if it ends on the first row! The first and last of temper on the final row are empty.
		IC[gid - (THREADBLK/2)] = temper[tid+1];
	}
	//Block 0 holds values 0 to 15 and end-15 to end.  In that order.
	else
	{
		if (tid >= THREADBLK/2)
		{
			IC[gid - (THREADBLK/2)] = temper[tid+1];
		}
		else
		{
			IC[(blockDim.x * gridDim.x) + (tid - THREADBLK/2) ] = temper[tid+1];
		}
	}
}

__global__ void wholeDiamond(REAL *right, REAL *left)
{

	int base = THREADBLK + 2;
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

		//Add the next values in.  Fix this shit.
		if (tid < 2)
		{
			temper[tid+(k-2)+shft_wr*base] = sL[itr+tid];
			temper[tidp+k+shft_wr*base] = sR[itr+tid];
		}

	}

	//DO THE MIDDLE ONE.
	if (blockIdx.x == (gridDim.x-1) && tid == 0 )
	{
		temper[2*base-1] = 0;
	}
	elseif (blockIdx.x == 0 && tid == 0)
	{
		temper[base] = 0;
	}

	temper[tidp] = fo * (temper[tid+base] + temper[tid+base+2]) + (1.f-2.f*fo) * temper[tidp+base];


	//Wind it down!
	for (int k = 0; k<height; k++)
	{
		// This tells you if the current row is the first or second.
		shft_wr = (k & 1);
		// Read and write are opposite rows.
		shft_rd = base*((shft_wr+1) & 1);
		//Block 0 is split so it needs a different algorithm.  This algorithm
		//is slightly different than top triangle as described in the note above.

		if (tid < (THREADBLK-k) && tid > k)
		{
			temper[tidp + ((THREADBLK+2)*shft_wr)] = fo * (temper[tid+shft_rd] + temper[tid+shft_rd+2]) + (1.f-2.f*fo) * temper[tidp+shft_rd];
		}
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

//Split one is always first.  Passing left like the downTriangle.  downTriangle
//should be rewritten so it isn't split.  Only write on a non split pass.
__global__ void splitDiamond(REAL *right, REAL *left)
{

	int base = THREADBLK + 2;
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

		//Add the next values in.  Fix this shit.
		if (tid < 2)
		{
			temper[tid+(k-2)+shft_wr*base] = sL[itr+tid];
			temper[tidp+k+shft_wr*base] = sR[itr+tid];
		}

	}

	//DO THE MIDDLE ONE.
	if (blockIdx.x == (gridDim.x-1) && tid == 0 )
	{
		temper[2*base-1] = 0;
	}
	elseif (blockIdx.x == 0 && tid == 0)
	{
		temper[base] = 0;
	}

	temper[tidp] = fo * (temper[tid+base] + temper[tid+base+2]) + (1.f-2.f*fo) * temper[tidp+base];


	//Wind it down!
	for (int k = 0; k<height; k++)
	{
		// This tells you if the current row is the first or second.
		shft_wr = (k & 1);
		// Read and write are opposite rows.
		shft_rd = base*((shft_wr+1) & 1);
		//Block 0 is split so it needs a different algorithm.  This algorithm
		//is slightly different than top triangle as described in the note above.

		if (tid < (THREADBLK-k) && tid > k)
		{
			temper[tidp + ((THREADBLK+2)*shft_wr)] = fo * (temper[tid+shft_rd] + temper[tid+shft_rd+2]) + (1.f-2.f*fo) * temper[tidp+shft_rd];
		}
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
