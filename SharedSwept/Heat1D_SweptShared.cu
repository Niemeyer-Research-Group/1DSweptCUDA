
#include <cuda.h>
#include "cuda_runtime_api.h"
#include "device_functions.h"

#include <ostream>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>

//#include "SwR_1DShared.h"

//-----------For testing --------------

__host__ __device__ void initFun(int xnode, REAL ds, REAL lx,REAL result)
{

    result = 500.f*expf((-ds*(REAL)xnode)/lx);

}

__host__ __device__ REAL execFunc(REAL tLeft, REAL tRight, REAL tCenter)
{

    return fo*(tLeft+tRight) + (1.f-2.f*fo)*tCenter;

}

//-----------For testing --------------

__global__ void upTriangle(REAL *IC, REAL *right, REAL *left)
{
	/*
	Initialize shared variables.  Each node (warp) will store 32 values on the
	right and left sides of their triangle, 2 on each side for each timeshLeftice.
	Since the base of the triangle is 32 numbers for each node, 16 timeshLeftices
	are evaluated per kernel call.
	Temper stores the temperatures at each timeshLeftice.  Since only the current
	and previous timeshLeftice results need to be held at each iteration.  This
	variable has 64 values, or two rows of 32, linearized.  The current and
	previous value alternate rows at each timeshLeftice.
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

	//The initial conditions are timeslice 0 so start k at 1.
	for (int k = 30; k>1 ; k-=2)
	{
		//Bitwise even odd. On even iterations write to first row.
		logic_position = (k/2 & 1);
		shift_wr = blockDim.x*logic_position;
		//On even iterations write to second row (starts at element 32)
		shft_rd = blockDim.x*((logic_position+1) & 1);

		//Each iteration the triangle narrows.  When k = 1, 30 points are
		//computed, k = 2, 28 points.
		if (tid <= k)
		{
			temper[tid + shft_wr] = execFunc(temper[tid+shift_rd], temper[tid2+shift_rd], temper[tid1+shift_rd]);
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
__global__ void downTriangle(REAL *IC, REAL *right, REAL *left)
{

	extern __shared__ REAL share[];

	REAL *temper = (REAL *) share;
	REAL *shRight = (REAL *) &share[2*blockDim.x+4];
	REAL *shLeft = (REAL *) &share[3*blockDim.x+4];

	int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
	int tid = threadIdx.x; //Block Thread ID
	int tid1 = tid + 1;
	int tid2 = tid + 2;
	//int height = blockDim.x/2;
	int shft_wr; //Initialize the shift to the written row of temper.
	int shft_rd; //Initialize the shift to the read row (opposite of written)
	int logic_position;
	int itr = 0;

	//Assign the initial values to the first row in temper, each warp (in this
	//case each block) has it's own version of temper shared among its threads.

	// Pass to the left so all checks are for block 0 (this reduces arithmetic).
	// The left ridge is always kept by the block.
	shLeft[tid] = right[gid];

	if (blockIdx.x == (gridDim.x-1))
	{
		shRight[tid] = left[tid];
        __syncthreads();
        if (tid1 == blockDim.x) shRight[tid] = shRight[tid-2];
	}
	else
	{
		shRight[tid] = left[gid+blockDim.x];
        if (gid == 0) shLeft[tid] = shLeft[tid2];
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
		shift_wr = blockDim.x*logic_position;
		//On even iterations write to second row (starts at element 32)
		shft_rd = blockDim.x*((logic_position+1) & 1);

		if (tid < 2)
		{
			temper[tid + shft_wr] = shLeft[tid+k];
			temper[tid2 + k + shft_wr] = shRight[tid+k];
		}

		if (tid < (k+2) && tid > 1)
		{
			temper[tid + shft_wr] = execFunc(temper[tid+shift_rd], temper[(tid-2)+shift_rd], temper[(tid-1)+shift_rd]);
		}
        __syncthreads();
	}


    temper[tid] = execFunc(temper[tid+blockDim.x], temper[tid2+blockDim.x], temper[tid1+blockDim.x]);

    IC[gid] = temper[tid];
}


__global__ void downSplitTriangle(REAL *IC, REAL *right, REAL *left)
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
	int itr = 0;

	//Assign the initial values to the first row in temper, each warp (in this
	//case each block) has it's own version of temper shared among its threads.

	// Pass to the left so all checks are for block 0 (this reduces arithmetic).
	// The left ridge is always kept by the block.
	shLeft[tid] = right[gid];

	// The right ridge is passed, each block 1-end gets the right of 0-end-1
	// Block 0 gets the right of the last block.
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
		temper[tid+2] = shRight[tid];
	}

	//Now we need two counters since we need to use shLeft and shRight EVERY iteration
	//instead of every other iteration and instead of growing smaller with every
	//iteration this grows larger.
	int itr = 2;

	for (int k = 4; k>1; k--)
	{
		// This tells you if the current row is the first or second.
		shft_wr = (k & 1);
		// Read and write are opposite rows.
		shft_rd = (THREADBLK+2)*((shft_wr+1) & 1);

		//Block 0 is split so it needs a different algorithm.  This algorithm
		//is shLeftightly different than top triangle as described in the note above.
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
			temper[(k-3)+((THREADBLK+2)*shft_wr)] = shLeft[itr];
			temper[(k-2)+((THREADBLK+2)*shft_wr)] = shLeft[itr+1];
			temper[itr2+((THREADBLK+2)*shft_wr)] = shRight[itr];
			itr2++;
			temper[itr2+((THREADBLK+2)*shft_wr)] = shRight[itr+1];
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
	int itr = 0;

	//int base = THREADBLK + 2;
	//int height = THREADBLK/2;

    shLeft[tid] = right[gid];

	if (blockIdx.x == (gridDim.x-1))
	{
		shRight[tid] = left[tid];
        __syncthreads();
        if (tid1 == blockDim.x) shRight[tid] = shRight[tid-1];
	}
	else
	{
		shRight[tid] = left[gid+blockDim.x];
        if (gid == 0) shLeft[tid] = shLeft[tid2];
	}

	if (tid < 2)
	{
		temper[tid] = shLeft[tid];
		temper[tid+2] = shRight[tid];
	}


    for (int k = 2; k < blockDim.x; k+=2)
	{
		logic_position = (k/2 & 1);
		shift_wr = blockDim.x*logic_position;
		//On even iterations write to second row (starts at element 32)
		shft_rd = blockDim.x*((logic_position+1) & 1);

		if (tid < 2)
		{
			temper[tid + shft_wr] = shLeft[tid+k];
			temper[tid2 + k + shft_wr] = shRight[tid+k];
		}

		if (tid < (k+2) && tid > 1)
		{
			temper[tid + shft_wr] = execFunc(temper[tid+shift_rd], temper[(tid-2)+shift_rd], temper[(tid-1)+shift_rd]);
		}
        __syncthreads();
	}

    temper[tid] = execFunc(temper[tid+blockDim.x], temper[tid2+blockDim.x], temper[tid1+blockDim.x]);

    __syncthreads(); // Then make sure each block of threads are synced.

    //-------------------TOP PART------------------------------------------

    int itr = 0;

	//The initial conditions are timeslice 0 so start k at 1.
	for (int k = 30; k>1 ; k-=2)
	{
		//Bitwise even odd. On even iterations write to first row.
		logic_position = (k/2 & 1);
		shift_wr = blockDim.x*logic_position;
		//On even iterations write to second row (starts at element 32)
		shft_rd = blockDim.x*((logic_position+1) & 1);

		//Each iteration the triangle narrows.  When k = 1, 30 points are
		//computed, k = 2, 28 points.
		if (tid <= k)
		{
			temper[tid + shft_wr] = execFunc(temper[tid+shift_rd], temper[tid2+shift_rd], temper[tid1+shift_rd]);
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

//Split one is always first.  Passing left like the downTriangle.  downTriangle
//should be rewritten so it isn't split.  Only write on a non split pass.
__global__ void splitDiamond(REAL *right, REAL *left)
{

	extern __shared__ REAL share[];

	REAL *temper = (REAL*) share;
	REAL *shRight = (REAL*) &share[2*blockDim.x+4];
	REAL *shLeft = (REAL*) &share[3*blockDim.x+4];

	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	int tidp = tid + 1;

	//Could put them in constant memory.
	int base = THREADBLK + 2;
	int height = THREADBLK/2;
	int shft_rd;
	int shft_wr;

	shRight[tid] = left[gid];

	if (blockIdx.x > 0)
	{
		shLeft[tid] = right[gid-blockDim.x];
	}
	else
	{
		shLeft[tid] = right[blockDim.x*(gridDim.x-1) + tid];
	}

	// Initialize temper. Kind of an unrolled for loop.  This is actually at

	if (tid < 2)
	{
		temper[tid+height-1] = shLeft[tid];
		temper[tidp+height] = shRight[tid];
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
		//is shLeftightly different than top triangle as described in the note above.
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
			temper[tid+(k-2)+shft_wr*base] = shLeft[itr+tid];
			temper[tid+(base-k)+shft_wr*base] = shRight[itr+tid];
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
		//is shLeftightly different than top triangle as described in the note above.

		if (tid < (THREADBLK-k) && tid > k)
		{
			temper[tidp + (base*shft_wr)] = fo * (temper[tid+shft_rd] + temper[tid+shft_rd+2]) + (1.f-2.f*fo) * temper[tidp+shft_rd];
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
		shLeft[k+itr+tid] = temper[(tid/2*(base-1))+(tid-1)+k];
		shRight[k+itr+tid] = temper[(((tid/2)+1)*(base-1))+(tid&1)-k];
		itr += 2;
	}

	__syncthreads();

	right[gid] = shRight[tid];
	left[gid] = shLeft[tid];

}

//Do the split diamond on the CPU?
__host__ void CPU_diamond(REAL right, REAL left, int tpb)
{
    int idx;
    int ht = tpb/2;
    REAL temper[tpb+1][tpb+2];
    temper[0][0] = left[0];
    temper[0][1] = left[1];
    temper[0][2] = right[0];
    temper[0][3] = right[1];

    //Splitting it is the whole point!

    for (int k = 1; k < ht; k++)
    {

        temper[k][0] = left[2*k];
        temper[k][1] = left[2*(k+1)];
        temper[k][(k+1)*2] = right[2*k];
        temper[k][(k+1)*2+1] = right[2*(k+1)];

        for(int n = 2; n<(k+1)*2; n++)
        {
            //Double trailing index.
            if(n == k+1)
            {
                temper[k][n] = execFunc(temper[k-1][n-2], temper[k-1][n-2], temper[k-1][n-1]);
            }
            //Double leading index.
            else if(n==k+2)
            {
                temper[k][n] = execFunc(temper[k-1][n], temper[k-1][n], temper[k-1][n-1]);
            }
            else
            {
                temper[k][n] = execFunc(temper[k-1][n-2], temper[k-1][n], temper[k-1][n-1]);
            }

        }

    }

    for(int n = 0; n < tpb; n++)
    {
        //Double trailing index.
        if(n == ht-1)
        {
            temper[ht][n] = execFunc(temper[k-1][n], temper[k-1][n+2], temper[k-1][n+1]);
        }
        //Double leading index.
        else if(n == ht)
        {
            temper[ht][n] = execFunc(temper[k-1][n], temper[k-1][n], temper[k-1][n-1]);
        }
        else
        {
            temper[ht][n] = execFunc(temper[k-1][n-2], temper[k-1][n], temper[k-1][n-1]);
        }
    }

    left[0] = temper[ht][0];
    left[1] = temper[ht][1];
    right[0] = temper[ht][tpb-2];
    right[1] = temper[ht][tpb-1];

    //Top part.
    for (int k = 1; k<ht; k++)
    {

        for (int n = 0; n<(tpb-2*k); n++)
        {
            if(n == ht-1)
            {
                temper[k+ht][n] = execFunc(temper[k-1+ht][n], temper[k-1+ht][n+2], temper[k-1+ht][n+1]);
            }
            //Double leading index.
            else if(n == ht)
            {
                temper[k+ht][n] = execFunc(temper[k-1+ht][n], temper[k-1+ht][n+2], temper[k-1+ht][n+1]);
            }
            else
            {
                temper[k+ht][n] = execFunc(temper[k-1+ht][n], temper[k-1+ht][n+2], temper[k-1+ht][n+1]);
            }
        }

        right[2*k] = temper[k+ht][0];
        right[2*k+1] = temper[k+ht][1];
        left[2*k] = temper[k+ht][(tpb-2) - 2*k];
        left[2*k+1] = temper[k+ht][(tpb-2) - 2*k + 1];

    }
}

//The host routine.
void sweptWrapper(const int bks, const int tpb, const int dv, REAL dt, const int t_end, const int cpu, REAL *IC, REAL *T_f)
{

	REAL *d_IC, *d_right, *d_left;
    REAL right, left;

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

	double t_eq = t_fullstep;

	// Call the kernels until you reach the iteration limit.
    if (cpu == 1)
    {
    	while(t_eq < t_end)
    	{

            cudaMemcpy(right,d_right,tpb*sizeof(REAL),cudaMemcpyDeviceToHost);
            cudaMemcpy(left,d_left,tpb*sizeof(REAL),cudaMemcpyDeviceToHost);

            CPU_diamond(right, left, tpb);

            cudaMemcpy(d_right, right, tpb*sizeof(REAL), cudaMemcpyHostToDevice);
            cudaMemcpy(d_left, left, tpb*sizeof(REAL), cudaMemcpyHostToDevice);

            wholeDiamond <<< bks,tpb,smem2 >>>(d_right,d_left,-1);

		    //So it always ends on a left pass since the down triangle is a right pass.

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
	}
    else
    {
        while(t_eq < t_end)
        {

            splitDiamond <<< bks,tpb,smem2 >>>(d_right,d_left,-1);

            wholeDiamond <<< bks,tpb,smem2 >>>(d_right,d_left,1);
            //So it always ends on a left pass since the down triangle is a right pass.

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
    }

	downTriangle <<< bks,tpb,smem2 >>>(d_IC,d_right,d_left);

	cudaMemcpy(T_f, d_IC, sizeof(REAL)*dv, cudaMemcpyDeviceToHost);

	cudaFree(d_IC);
	cudaFree(d_right);
	cudaFree(d_left);

}

int main( int argc, char *argv[] )
{
	if (argc != 6)
	{
		cout << "The Program takes five inputs: #Divisions, #Threads/block, dt, finish time, and GPU/CPU or all GPU" << endl;
		exit(-1);
	}
	// Choose the GPGPU.  This is device 0 in my machine which has 2 devices.
	cudaSetDevice(0);

	const int dv = atoi(argv[1]); //Setting it to an int helps with arrays
	const int tpb = atoi(argv[2]);
	const int tf = atoi(argv[4]);
	const int bks = dv/tpb; //The number of blocks since threads/block = 32.

	//Conditions for main input.  Unit testing kinda.
	//dv and tpb must be powers of two.  dv must be larger than tpb and divisible by
	//tpb.

	//if ((dv & (tpb-1) !=0) || tpb&31 != 0)

	REAL dsc;
	dsc.x = lx/(dv-1);
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
		initFun((float)k*dsc.x,IC[k]);
	}

	// Call out the file before the loop and write out the initial condition.
	ofstream fwr, ftime;
	fwr.open("Results/Heat1D_Result.dat",ios::trunc);
	ftime.open("Results/Heat1D_Timing.txt",ios::app);
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

	ftime << dv << " " << tpb << " " << timed << endl;

	ftime.close();

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
