
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

#ifndef REAL
#define REAL  float
#endif

using namespace std;

__constant__ REAL fo;

const double PI = 3.141592653589793238463;

const REAL lx = 50.0;

const REAL th_diff = 8.418e-5;

//-----------For testing --------------

__host__ __device__ REAL initFun(int xnode, REAL ds, REAL lx)
{

    return 500.f*expf((-ds*(REAL)xnode)/lx);

}

__host__ __device__ REAL execFunc(REAL tLeft, REAL tRight, REAL tCenter)
{

    return fo*(tLeft+tRight) + (1.f-2.f*fo)*tCenter;

}
//-----------For testing --------------

// I think we still need the shared memory.
// We still need to get it into registers.
// Aha!  It can only have a base of 32!  So we could do it at compile time!
// Can't share values across warps!
__global__ void upTriangle(REAL *IC, REAL *right, REAL *left)
{
	extern __shared__ REAL share[];

    REAL *shRight = (REAL *) share;
    REAL *shLeft = (REAL *) &share[32];

    REAL temper_reg1, temper_reg2;

	int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
	int tid = threadIdx.x; //Block Thread ID
    int tidp = tid + 2;

	//Assign the initial values to the first row in temper, each warp (in this
	//case each block) has it's own version of temper shared among its threads.
	temper_reg1 = IC[gid];

    temper_reg2 = execFunc(__shfl_down(temper_reg1,1), __shfl_up(temper_reg1,1), temper_reg1);

    if (tid < 1)
    {
        shLeft[tid] = temper_reg1;
        shLeft[tidp] = __shfl_up(temper_reg2,1);
    }
    if (tid > 29)
    {
        shRight[tid - 30] = temper_reg1;
        shRight[tid - 28] = __shfl_down(temper_reg2,1);
    }

	//The initial conditions are timeslice 0 so start k at 1.
	for (int k = 2; k<16; k+=2)
	{
        temper_reg1 = execFunc(__shfl_down(temper_reg2,1), __shfl_up(temper_reg2,1), temper_reg2);
		temper_reg2 = execFunc(__shfl_down(temper_reg1,1), __shfl_up(temper_reg1,1), temper_reg1);

        //Really tricky to get unique values with threads.
        //GET THE UNIQUE VALUES
        if (tid >= k && tid < (k+2))
        {
            shLeft[tid] = temper_reg1;
            shLeft[tidp] = __shfl_up(temper_reg2,1);
        }
        if (tid > (31-k) && tid < (29-k))
        {
            shRight[tid - 30] = temper_reg1;
            shRight[tid - 28] = __shfl_down(temper_reg2,1);
        }
	}

	//After the triangle has been computed, the right and left shared arrays
	//are stored in global memory by the global thread ID since (conveniently),
	//they're the same size as a warp!

	right[gid] = shRight[tid];
	left[gid] = shLeft[tid];

}

// Down triangle is only called at the end when data is passed left.  It's never split.
// It returns IC which is a full 1D result at a certain time.
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

    #pragma unroll
	for (int k = 2; k < blockDim.x; k+=2)
	{
		logic_position = (k/2 & 1);
		shft_wr = base * logic_position;

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

    if (gid == 0) temper[base] = temper[base+2];
    if (gid == (blockDim.x*gridDim.x-1)) temper[2*blockDim.x+3] = temper[2*blockDim.x+1];

    temper[tid] = execFunc(temper[tid+base], temper[tid2+base], temper[tid1+base]);

    IC[gid] = temper[tid];
}


/*__global__ void downSplitTriangle(REAL *IC, REAL *right, REAL *left)
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
*/

__global__ void wholeDiamond(REAL *right, REAL *left, bool full)
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
	int base = blockDim.x + 2;
	//int height = THREADBLK/2;

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
		temper[tid+2] = shRight[tid];
	}


    for (int k = 2; k < blockDim.x; k+=2)
	{
		logic_position = (k/2 & 1);
		shft_wr =base*logic_position;
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

    if (full)
    {
        if (gid == 0) temper[base] = temper[base+2];
        if (gid == (blockDim.x*gridDim.x-1)) temper[2*blockDim.x+3] = temper[2*blockDim.x+1];
    }

    temper[tid] = execFunc(temper[tid+base], temper[tid2+base], temper[tid1+base]);

    __syncthreads(); // Then make sure each block of threads are synced.

    //-------------------TOP PART------------------------------------------

    int itr = 0;
    const int right_pick[4] = {2,1,-1,0}; //Template!

	//The initial conditions are timeslice 0 so start k at 1.
	for (int k = (blockDim.x-2); k>1 ; k-=2)
	{
		//Bitwise even odd. On even iterations write to first row.
		logic_position = (k/2 & 1);
		shft_wr = base*logic_position;
		//On even iterations write to second row (starts at element 32)
		shft_rd = base*((logic_position+1) & 1);

		//Each iteration the triangle narrows.  When k = 1, 30 points are
		//computed, k = 2, 28 points.
		if (tid < k)
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

//Split one is always first.  Passing left like the downTriangle.  downTriangle
//should be rewritten so it isn't split.  Only write on a non split pass.
__global__ void splitDiamond(REAL *right, REAL *left)
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
        temper[tid] = shLeft[tid];
		temper[tid2] = shRight[tid];
	}
	//Wind it up!

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

    __syncthreads(); // Then make sure each block of threads are synced.

	int itr = 0;

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
        if (blockDim.x > 0)
        {
    		if (tid <= k)
    		{
    			temper[tid + shft_wr] = execFunc(temper[tid+shft_rd], temper[tid2+shft_rd], temper[tid1+shft_rd]);
    		}
        }
        else
        {
            if (tid == (k/2-1))
            {
                temper[tid + shft_wr] = execFunc(temper[tid + shft_rd], temper[tid+shft_rd], temper[tid1 + shft_rd]);
            }
            else if (tid == k/2)
            {
                temper[tid + shft_wr] = execFunc(temper[tid2 + shft_rd], temper[tid2+shft_rd], temper[tid1 + shft_rd]);
            }
            else
            {
                temper[tid + shft_wr] = execFunc(temper[tid + shft_rd], temper[tid2+shft_rd], temper[tid1 + shft_rd]);
            }
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


//Do the split diamond on the CPU?
__host__ void CPU_diamond(REAL *right, REAL *left, int tpb)
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
            temper[ht][n] = execFunc(temper[ht-1][n], temper[ht-1][n+2], temper[ht-1][n+1]);
        }
        //Double leading index.
        else if(n == ht)
        {
            temper[ht][n] = execFunc(temper[ht-1][n], temper[ht-1][n], temper[ht-1][n-1]);
        }
        else
        {
            temper[ht][n] = execFunc(temper[ht-1][n-2], temper[ht-1][n], temper[ht-1][n-1]);
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
double sweptWrapper(const int bks, int tpb, const int dv, REAL dt, const int t_end, const int cpu, REAL *IC, REAL *T_f)
{

	REAL *d_IC, *d_right, *d_left;
    REAL right, left;

	cudaMalloc((void **)&d_IC, sizeof(REAL)*dv);
	cudaMalloc((void **)&d_right, sizeof(REAL)*dv);
	cudaMalloc((void **)&d_left, sizeof(REAL)*dv);

	// Copy the initial conditions to the device array.
	cudaMemcpy(d_IC,IC,sizeof(REAL)*dv,cudaMemcpyHostToDevice);
	// Start the counter and start the clock.
	const double t_fullstep = dt*(double)tpb;

	const size_t smem1 = 4*tpb*sizeof(REAL);
	const size_t smem2 = (4*tpb+4)*sizeof(REAL);

	upTriangle <<< bks,tpb,smem1 >>>(d_IC,d_right,d_left);

	double t_eq = t_fullstep/2;

	// Call the kernels until you reach the iteration limit.
    if (cpu)
    {
    	while(t_eq < t_end)
    	{

            cudaMemcpy(&right,d_right,tpb*sizeof(REAL),cudaMemcpyDeviceToHost);
            cudaMemcpy(&left,d_left,tpb*sizeof(REAL),cudaMemcpyDeviceToHost);

            CPU_diamond(&right, &left, tpb);
            wholeDiamond <<< bks-1,tpb,smem2 >>>(d_right,d_left,1);

            cudaMemcpy(d_right, &right, tpb*sizeof(REAL), cudaMemcpyHostToDevice);
            cudaMemcpy(d_left, &left, tpb*sizeof(REAL), cudaMemcpyHostToDevice);

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

            splitDiamond <<< bks,tpb,smem2 >>>(d_right,d_left);

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

    return t_eq;

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

    int dv = atoi(argv[1]); //Setting it to an int helps with arrays
	const int tpb = atoi(argv[2]);
	const int tf = atoi(argv[4]);
	const int bks = dv/tpb; //The number of blocks since threads/block = 32.
	const int tst = atoi(argv[5]);
    const REAL ds = lx/((float)dv-1.f);
    REAL dt = atof(argv[3]);
    REAL fou = dt*th_diff/(ds*ds);

	//Conditions for main input.  Unit testing kinda.
	//dv and tpb must be powers of two.  dv must be larger than tpb and divisible by
	//tpb.

	//if ((dv & (tpb-1) !=0) || tpb&31 != 0)

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
		IC[k] = initFun(k, ds, lx);
	}

	// Call out the file before the loop and write out the initial condition.
	ofstream fwr, ftime;
	fwr.open("Results/Heat1D_Result.dat",ios::trunc);
	ftime.open("Results/Heat1D_Timing.txt",ios::app);
	// Write out x length and then delta x and then delta t.
	// First item of each line is timestamp.
	fwr << lx << " " << dv << " " << ds << " " << endl << 0 << " ";

	for (int k = 0; k<dv; k++)
	{
		fwr << IC[k] << " ";
	}

	fwr << endl;

	IC_p = IC;

    //Transfer data to GPU.
	// This puts the Fourier number in constant memory.
	cudaMemcpyToSymbol(fo,&fou,sizeof(REAL));

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

	tfm = sweptWrapper(bks,tpb,dv,dt,tf,tst,IC_p,Tfin_p);

	// Show the time and write out the final condition.
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime( &timed, start, stop);

	timed = timed * 1.e-3;

	cout << "That took: " << timed << " seconds" << endl;

	ftime << dv << " " << tpb << " " << timed << endl;

	ftime.close();

	fwr << tfm << " ";
	for (int k = 0; k<dv; k++)
	{
		fwr << Tfin_p[k] << " ";
	}

	fwr.close();

	// Free the memory and reset the device.

	cudaDeviceReset();
	cudaEventDestroy( start );
	cudaEventDestroy( start );

	return 0;

}
