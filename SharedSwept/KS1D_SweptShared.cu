//I just really need to do these versions in one file before I really try to modularize
//the code.

//K-S involves no splitting.  And clearly a CPU diamond would be a waste.
//But it does need to pass it back and forth so it needs different passing versions.
//It would also be fruitful to test out streams on this one.
//The heat equation requires much more because of the boundary conditions.
//Also, we could test putting the whole thing on the GPU here.  So it never leaves shared memory.

//COMPILE LINE!
// nvcc -o ./bin/KSOut KS1D_SweptShared.cu -gencode arch=compute_35,code=sm_35 -lm -w -std=c++11 -Xcompiler -fopenmp

//RUN LINE!
// ./bin/KSOut 256 2048 .01 10 0

//DO:
//Use malloc or cudaHostAlloc for all initial and final conditions.
//Write the ability to pull out time snapshots.
//The ability to feed in initial conditions.
//Input paths to output as argc?  Input?
//Ask about just making up my own conventions.  Like .dat vs txt.  Use that as flag?

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_functions.h>

#include <iostream>
#include <ostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>


#ifndef REAL
#define REAL  float
#endif

//-----------For testing --------------

using namespace std;

const REAL dx = 0.5;

struct discConstants{

	REAL dx;
	REAL dx2;
	REAL dx4;
	REAL dt;
	REAL dt_half;
};

__constant__ discConstants disc;

__host__ __device__
REAL initFun(float xnode)
{
	return 2.f * cos(19.f*xnode*M_PI/128.f);
}

__host__ __device__
__forceinline__
REAL fourthDer(REAL tfarLeft, REAL tLeft, REAL tCenter, REAL tRight, REAL tfarRight)
{
	return (tfarLeft - 4.f*tLeft + 6.f*tCenter - 4.f*tRight + tfarRight)/(disc.dx4);
}

__host__ __device__
__forceinline__
REAL secondDer(REAL tLeft, REAL tRight, REAL tCenter)
{
	return (tLeft + tRight - 2.f*tCenter)/(disc.dx2);
}

__host__ __device__
__forceinline__
REAL convect(REAL tLeft, REAL tRight)
{
	return (tRight*tRight - tLeft*tLeft)/(4.f*disc.dx);
}

__host__ __device__
REAL stutterStep(REAL tfarLeft, REAL tLeft, REAL tCenter, REAL tRight, REAL tfarRight)
{
	return tCenter - disc.dt_half * (convect(tLeft, tRight) + secondDer(tLeft, tRight, tCenter) +
		fourthDer(tfarLeft, tLeft, tCenter, tRight, tfarRight));
}

__host__ __device__
REAL finalStep(REAL tfarLeft, REAL tLeft, REAL tCenter, REAL tCenter_orig, REAL tRight, REAL tfarRight)
{
	return tCenter_orig - disc.dt_half * (convect(tLeft, tRight) + secondDer(tLeft, tRight, tCenter) +
			fourthDer(tfarLeft, tLeft, tCenter, tRight, tfarRight));
}

//Up is the same.
__global__
void
upTriangle(REAL *IC, REAL *right, REAL *left)
{

	extern __shared__ REAL temper[];

	int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
	int tid = threadIdx.x; //Block Thread ID

    int tid_top[5], tid_bottom[5];
	#pragma unroll
	for (int k = -2; k<3; k++)
	{
		tid_top[k+2] = tid + k + blockDim.x;
		tid_bottom[k+2] = tid + k;
	}

	int leftidx = ((tid/4 & 1) * blockDim.x) + (tid/4)*2 + (tid & 3);
	int rightidx = (blockDim.x - 4) + ((tid/4 & 1) * blockDim.x) + (tid & 3) - (tid/4)*2;

	int step2;

    //Assign the initial values to the first row in temper, each block
    //has it's own version of temper shared among its threads.
	temper[tid] = IC[gid];

	if (tid > 1 && tid <(blockDim.x-2))
	{
		temper[tid_top[2]] = stutterStep(temper[tid_bottom[0]], temper[tid_bottom[1]], temper[tid_bottom[2]],
			temper[tid_bottom[3]], temper[tid_bottom[4]]);
	}

	__syncthreads();

	//The initial conditions are timslice 0 so start k at 1.
	for (int k = 4; k<(blockDim.x/2); k+=4)
	{
		if (tid < (blockDim.x-k) && tid >= k)
		{
			temper[tid] = finalStep(temper[tid_top[0]], temper[tid_top[1]], temper[tid_top[2]],
				temper[tid], temper[tid_top[3]], temper[tid_top[4]]);

		}
		step2 = k + 2;
		__syncthreads();

		if (tid < (blockDim.x-step2) && tid >= step2)
		{
			temper[tid_top[2]] = stutterStep(temper[tid_bottom[0]], temper[tid_bottom[1]], temper[tid_bottom[2]],
				temper[tid_bottom[3]], temper[tid_bottom[4]]);
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

	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	int tididx = tid + 2;
	int base = blockDim.x + 4;
	int height = base/2;
	int step2;

	int tid_top[5], tid_bottom[5];
	#pragma unroll
	for (int k = -2; k<3; k++)
	{
		tid_top[k+2] = tididx + k + base;
		tid_bottom[k+2] = tididx + k;
	}

	int leftidx = height + ((tid/4 & 1) * base) + (tid & 3) - (4 + (tid/4) * 2);
	int rightidx = height + ((tid/4 & 1) * base) + (tid/4)*2 + (tid & 3);
	int gidin = (gid + blockDim.x) & ((blockDim.x*gridDim.x)-1);


	temper[leftidx] = right[gid];
	temper[rightidx] = left[gidin];

	for (int k = (height-2); k>0; k-=4)
	{
		if (tididx < (base-k) && tididx >= k)
		{
			temper[tid_top[2]] = stutterStep(temper[tid_bottom[0]], temper[tid_bottom[1]], temper[tid_bottom[2]],
				temper[tid_bottom[3]], temper[tid_bottom[4]]);

		}

		step2 = k-2;

		if (tididx < (base-step2) && tididx >= step2)
		{
			temper[tididx] = finalStep(temper[tid_top[0]], temper[tid_top[1]], temper[tid_top[2]],
				temper[tididx], temper[tid_top[3]], temper[tid_top[4]]);
		}

		//Make sure the threads are synced
		__syncthreads();
	}

    IC[gid] = temper[tididx];
}

// Pass to split is false.  Pass to whole is true.
__global__
void
wholeDiamond(REAL *right, REAL *left, int pass)
{
	extern __shared__ REAL temper[];

	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	int tididx = tid + 2;
	int base = blockDim.x + 4;
	int height = base/2;
	int step2;

	int tid_top[5], tid_bottom[5];
	#pragma unroll
	for (int k = -2; k<3; k++)
	{
		tid_top[k+2] = tididx + k + base;
		tid_bottom[k+2] = tididx + k;
	}

	int leftidx = height + ((tid/4 & 1) * base) + (tid & 3) - (4 + (tid/4) * 2);
	int rightidx = height + ((tid/4 & 1) * base) + (tid/4)*2 + (tid & 3);
	int gidin = (gid + pass*blockDim.x) & ((blockDim.x*gridDim.x)-1);

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


	for (int k = (height-2); k>0; k-=4)
	{
		if (tididx < (base-k) && tididx >= k)
		{
			temper[tid_top[2]] = stutterStep(temper[tid_bottom[0]], temper[tid_bottom[1]], temper[tid_bottom[2]],
				temper[tid_bottom[3]], temper[tid_bottom[4]]);

		}

		step2 = k-2;
		__syncthreads();

		if (tididx < (base-step2) && tididx >= step2)
		{
			temper[tididx] = finalStep(temper[tid_top[0]], temper[tid_top[1]], temper[tid_top[2]],
				temper[tididx], temper[tid_top[3]], temper[tid_top[4]]);
		}

		//Make sure the threads are synced
		__syncthreads();
	}

	//Shift the last row to justify it at 0.
	temper[tid] = temper[tididx];
    //-------------------TOP PART------------------------------------------

	leftidx = ((tid/4 & 1) * blockDim.x) + (tid/4)*2 + (tid & 3);
	rightidx = (blockDim.x - 4) + ((tid/4 & 1) * blockDim.x) + (tid & 3) - (tid/4)*2;

	#pragma unroll
	for (int k = -2; k<3; k++)
	{
		tid_top[k+2] = tid + k + blockDim.x;
		tid_bottom[k+2] = tid + k;
	}

	if (tid > 1 && tid <(blockDim.x-2))
	{
		temper[tid_top[2]] = stutterStep(temper[tid_bottom[0]], temper[tid_bottom[1]], temper[tid_bottom[2]],
			temper[tid_bottom[3]], temper[tid_bottom[4]]);
	}

	__syncthreads();

	//The initial conditions are timslice 0 so start k at 1.
	for (int k = 4; k<(blockDim.x/2); k+=4)
	{
		if (tid < (blockDim.x-k) && tid >= k)
		{
			temper[tid] = finalStep(temper[tid_top[0]], temper[tid_top[1]], temper[tid_top[2]],
				temper[tid], temper[tid_top[3]], temper[tid_top[4]]);
		}

		step2 = k+2;
		__syncthreads();

		if (tid < (blockDim.x-step2) && tid >= step2)
		{
			temper[tid_top[2]] = stutterStep(temper[tid_bottom[0]], temper[tid_bottom[1]], temper[tid_bottom[2]],
				temper[tid_bottom[3]], temper[tid_bottom[4]]);
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

//The host routine.
double
sweptWrapper(const int bks, int tpb, const int dv, REAL dt, const int t_end,
	REAL *IC, REAL *T_f, const float freq, ofstream &fwr)
{

	REAL *d_IC, *d_right, *d_left;
	cudaMalloc((void **)&d_IC, sizeof(REAL)*dv);
	cudaMalloc((void **)&d_right, sizeof(REAL)*dv);
	cudaMalloc((void **)&d_left, sizeof(REAL)*dv);

	// Copy the initial conditions to the device array.
	cudaMemcpy(d_IC,IC,sizeof(REAL)*dv,cudaMemcpyHostToDevice);
	//Start the counter and start the clock.
	//
	//Every other step is a full timestep and each cycle is half tpb steps.
	const double t_fullstep = 0.25 * dt * (double)tpb;
	double twrite = freq;

	const size_t smem1 = 2*tpb*sizeof(REAL);
	const size_t smem2 = (2*tpb+8)*sizeof(REAL);

	upTriangle <<< bks,tpb,smem1 >>> (d_IC,d_right,d_left);
	wholeDiamond <<< bks,tpb,smem2 >>> (d_right,d_left,-1);
	double t_eq = t_fullstep;

	// Call the kernels until you reach the iteration limit.
	while(t_eq < t_end)
	{

		wholeDiamond <<< bks,tpb,smem2 >>> (d_right,d_left,1);
		//So it always ends on a left pass since the down triangle is a right pass.
		wholeDiamond <<< bks,tpb,smem2 >>> (d_right,d_left,-1);

		t_eq += t_fullstep;


	 	if (t_eq > twrite)
		{
			downTriangle <<< bks,tpb,smem2 >>>(d_IC,d_right,d_left);

			cudaMemcpy(T_f, d_IC, sizeof(REAL)*dv, cudaMemcpyDeviceToHost);
			fwr << t_eq << " ";

			for (int k = 0; k<dv; k++)
			{
				fwr << T_f[k] << " ";
			}
			fwr << endl;

			upTriangle <<< bks,tpb,smem1 >>>(d_IC,d_right,d_left);
			wholeDiamond <<< bks,tpb,smem2 >>>(d_right,d_left,-1);

			twrite += freq;
		}

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

	if (argc != 7)
	{
		cout << "The Program takes six inputs, #Divisions, #Threads/block, dt, finish time, test or not, and output frequency" << endl;
		exit(-1);
	}

	// Choose the GPGPU.  This is device 0 in my machine which has 2 devices.
	cudaSetDevice(0);

	int dv = atoi(argv[1]); //Setting it to an int helps with arrays
	const int tpb = atoi(argv[2]);
	const int tf = atoi(argv[4]);
	const int bks = dv/tpb; //The number of blocks since threads/block = 32.
	const int tst = atoi(argv[5]);
	const float freq = atof(argv[6]);

	const float lx = dv*dx;

	//Conditions for main input.  Unit testing kinda.
	//dv and tpb must be powers of two.  dv must be larger than tpb and divisible by
	//tpb.

	if ((dv & (tpb-1) !=0) || (tpb&31) != 0)
    {
        cout << "INVALID NUMERIC INPUT!! "<< endl;
        cout << "2nd ARGUMENT MUST BE A POWER OF TWO >= 32 AND FIRST ARGUMENT MUST BE DIVISIBLE BY SECOND" << endl;
        exit(-1);
    }

	discConstants dsc = {
		dx, //dx
		dx*dx, //dx^2
		dx*dx*dx*dx, //dx^4
		atof(argv[3]), //dt
		atof(argv[3])*0.5, //dt half
	};

	// Initialize arrays.
    REAL *IC = (REAL*)malloc(dv*sizeof(REAL));
	REAL *T_final = (REAL*)malloc(dv*sizeof(REAL));

	// Some initial condition for the bar temperature, an exponential decay
	// function.
	for (int k = 0; k<dv; k++)
	{
		IC[k] = initFun((float)k*dsc.dx);
	}

	// Call out the file before the loop and write out the initial condition.
	ofstream fwr, ftime;
	fwr.open("Results/KS1D_Result.dat",ios::trunc);
	if (tst) ftime.open("Results/KS1D_Timing.txt",ios::app);

	// Write out x length and then delta x and then delta t.
	// First item of each line is timestamp.
	fwr << lx << " " << dv << " " << dsc.dx << " " << endl << 0 << " ";

	for (int k = 0; k<dv; k++)
	{
		fwr << IC[k] << " ";
	}

	fwr << endl;


	// Transfer data to GPU.

	// This puts the Fourier number in constant memory.
	cudaMemcpyToSymbol(disc,&dsc,sizeof(dsc));

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

	tfm = sweptWrapper(bks, tpb, dv, dsc.dt, tf, IC, T_final, freq, fwr);

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
		fwr << T_final[k] << " ";
	}

	fwr.close();

	// Free the memory and reset the device.
	cudaDeviceReset();
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
	free(IC);
    free(T_final);

	return 0;

}
