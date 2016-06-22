

#include <cuda.h>

#include <cstdio>
#include <cstdilib>
#include <cmath>

#include "SwR_1DShared.h"

//-----------For testing --------------

__host__ void initFun(int xnode, REAL ds, REAL lx,REAL result)
{

    result = 500.f*expf((-ds*(REAL)xnode)/lx);

}

__device__ REAL execFunc(REAL tLeft, REAL tRight, REAL tCenter)
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

	if (blockIdx.x == (gridDim.x-1))
	{
		shRight[tid] = left[tid];
        __syncthreads();
        if (tid1 == blockDim.x) shRight[tid] = shRight[tid-1];
	}
	else
	{
		shRight[tid] = left[gid+blockDim.x];
        if (gid == 0) shLeft[tid-1] = shLeft[tid];
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

	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	int tidp = tid + 1;

	//___________-----------------________Need new paradigm.
	int base = THREADBLK + 2;
	int height = THREADBLK/2;

	int shft_rd;
	int shft_wr;

	shLeft[tid] = right[gid];

	// The right ridge is passed, each block 1-end gets the right of 0-end-1
	// Block 0 gets the right of the last block.
	if (blockIdx.x > 0)
	{
		shRight[tid] = left[gid-blockDim.x];
	}
	else
	{
		shRight[tid] = left[blockDim.x*(gridDim.x-1) + tid];
	}

	// Initialize temper. Kind of an unrolled for loop.  This is actually at
	// Timestep 0.

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
		shft_wr = ((k + 1) & 1);
		// Read and write are opposite rows.
		shft_rd = base*((shft_wr+1) & 1);
		//Block 0 is split so it needs a different algorithm.  This algorithm
		//is shLeftightly different than top triangle as described in the note above.

		if (tid <= ((THREADBLK+1)-k) && tid >= k)
		{
			temper[tidp + ((base)*shft_wr)] = fo * (temper[tid+shft_rd] + temper[tid+shft_rd+2]) + (1.f-2.f*fo) * temper[tidp+shft_rd];
		}

		//Add the next values in.
		if (tid < 2)
		{
			temper[tid+(k-2)+shft_wr*base] = shLeft[itr+tid];
			temper[tidp+k+shft_wr*base] = shRight[itr+tid];
			itr += 2;
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
			shLeft[k+itr+tid] = temper[(tid/2*(base-1))+(tid-1)+k];
			shRight[k+itr+tid] = temper[(((tid/2)+1)*(base-1))+(tid&1)-k];
			itr += 2;
		}

	}
	__syncthreads();

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
			temper[tidp] = 2.f * fo * (temper[tid+base]-temper[tid+base+1]) + temper[tidp+base;
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
    int iter;
    REAL temper[tpb+1][tpb+2];
    temper[0][0] = left[0];
    temper[0][1] = left[1];
    temper[0][2] = right[0];
    temper[0][3] = right[1];
    iter = 2;

    for (int k = 1;k < tpb+1; k++)
    {
        for(int n = 2)




    }


}

//The host routine.
void sweptWrapper(int bks, int tpb ,int t_end, REAL IC, REAL T_f)
{

	REAL *d_IC, *d_right, *d_left;
	cudaMalloc((void **)&d_IC, sizeof(REAL)*dv);
	cudaMalloc((void **)&d_right, sizeof(REAL)*dv);
	cudaMalloc((void **)&d_left, sizeof(REAL)*dv);

	// Copy the initial conditions to the device array.
	cudaMemcpy(d_IC,IC,sizeof(REAL)*dv,cudaMemcpyHostToDevice);
	// Start the counter and start the clock.
	REAL t_eq = 0.;
	REAL t_fullstep = TS*(THREADBLK+1);

	const size_t smem1 = 4*tpb*sizeof(REAL);
	const size_t smem2 = (4*tpb+4)*sizeof(REAL);

	//upTriangle <<< bks,tpb,smem1>>>(d_IC,d_right,d_left);

	// Call the kernels until you reach the iteration limit.
	while(t_eq < t_end)
	{

		upTriangle <<< bks,tpb,smem1>>>(d_IC,d_right,d_left);

		downTriangle <<< bks,tpb,smem2 >>>(d_IC,d_right,d_left);

		/*
		splitDiamond <<< bks,tpb,smem2 >>> (d_right,d_left);
		wholeDiamond <<< bks,tpb,smem2 >>> (d_right,d_left);
		*/

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

	//downTriangle <<< bks,tpb,smem2 >>>(d_IC,d_right,d_left);

	cudaMemcpy(T_f, d_IC, sizeof(REAL)*dv, cudaMemcpyDeviceToHost);

}
