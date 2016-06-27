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
#include <cstdilib>
#include <cmath>
#include <fstream>


//-----------For testing --------------

__host__ __device__ void initFun(int xnode, REAL ds, REAL lx, REAL result)
{
	REAL xs = xnode*ds;
	result.x = 2.f*cos(19.f*xs)/12.f);
	result.y = ;
}

__host__ __device__ REAL execFunc(REAL tLeft, REAL tRight, REAL tCenter)
{



}

//-----------For testing --------------

//Up is the same.
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
__global__
void
downTriangle(REAL *IC, REAL * __restrict__ right, REAL *__restrict__ left)
{

	extern __shared__ REAL share[];

	REAL *temper = (REAL *) share;
	REAL *shRight = (REAL *) &share[2*blockDim.x+4];
	REAL *shLeft = (REAL *) &share[3*blockDim.x+4];

	unsigned int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
	unsigned int tid = threadIdx.x; //Block Thread ID
    //Something like this.
    unsigned int gid_plus = (gid + blockDim.x) & (blockDim.x * gridDim.x - 1)

	unsigned int tid1 = tid + 1;
	unsigned int tid2 = tid + 2;
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


__global__ void wholeDiamond(REAL * __restrict__ right, REAL * __restrict__ left, int swap)
{
    //Swap is either -1 or 1.  Since it either passes back or forward.
    extern __shared__ REAL share[];

	REAL *temper = (REAL*) share;
	REAL *shRight = (REAL*) &share[2*blockDim.x+4];
	REAL *shLeft = (REAL*) &share[3*blockDim.x+4];

    unsigned int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
	unsigned int tid = threadIdx.x; //Block Thread ID
    //Something like this.
    unsigned int gid_plus = (gid + swap*blockDim.x) & (blockDim.x * gridDim.x - 1)

	unsigned int tid1 = tid + 1;
	unsigned int tid2 = tid + 2;
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



int main()
{
	// Choose the GPGPU.  This is device 0 in my machine which has 2 devices.
	cudaSetDevice(0);
	const int dv = int(DIVISIONS); //Setting it to an int helps with arrays
	const int bks = dv/THREADBLK; //The number of blocks since threads/block = 32.
	// Threads/block will be experimented on.
	const REAL ds = LENX/(DIVISIONS-1); //The x division length.
	REAL fou = TS*TH_DIFF/(ds*ds); //The Fourier number.

	// Initialize arrays.
	REAL IC[dv];
	REAL T_final[dv];
	double wall0, wall1, timed;

	// Some initial condition for the bar temperature, an exponential decay
	// function.
	for (int k = 0; k<dv; k++)
	{
		IC[k] = 500.f*expf((-ds*k)/LENX);
	}

	cout << fou << endl;
	// Call out the file before the loop and write out the initial condition.
	ofstream fwr;
	fwr.open("1DHeatEQResult.dat",ios::trunc);
	// Write out x length and then delta x and then delta t.
	// First item of each line is timestamp.
	fwr << LENX << " " << DIVISIONS << " " << TS << " " << endl << 0 << " ";

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

	// Start the counter and start the clock.
	REAL t_eq = 0.;
	REAL t_fullstep = TS*(THREADBLK+1);
	wall0 = clock();

	// Call the kernels until you reach the iteration limit.
	sweptWrapper(bks,THREADBLK,FINISH,IC,T_final);

	// Show the time and write out the final condition.
	wall1 = clock();
	timed = (wall1-wall0)/CLOCKS_PER_SEC;

	cout << "That took: " << timed << " seconds" << endl;

	fwr << t_eq << " ";
	for (int k = 0; k<dv; k++)
	{
		fwr << T_final[k] << " ";
	}

	fwr.close();

	// Free the memory and reset the device.
	cudaFree(d_IC);
	cudaFree(d_right);
	cudaFree(d_left);
	cudaDeviceReset();

	return 0;

}
