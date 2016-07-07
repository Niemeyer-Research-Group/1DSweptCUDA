//HEAT Yo.
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

__global__ void upTriangle(REAL *IC, REAL *right, REAL *left)
{

	extern __shared__ REAL share[];

	REAL *temper = (REAL*) share;
	REAL *shRight = (REAL*) &share[2*blockDim.x];
	REAL *shLeft = (REAL*) &share[3*blockDim.x];

	int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
	int tid = threadIdx.x; //Block Thread ID
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

// Down triangle is only called at the end when data is passed left.  It's never split.
// It returns IC which is a full 1D result at a certain time.
__global__ void downTriangle(REAL *IC, REAL *right, REAL *left)
{

    //Now temper needs to accommodate a longer row by 2, one on each side.
	//since it has two rows that's 4 extra floats.  The last row will still be
	//32 numbers long.

	extern __shared__ REAL temper[];

	//Same as upTriangle
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
    int lastidx = ((blockDim.x*gridDim.x)-1);
	int tid1 = tid + 1;
	int tid2 = tid + 2;
	int base = blockDim.x + 2;
	int height = base/2;
	int shft_rd;
	int shft_wr;
	int leftidx = base/2 - tid/2 + ((tid/2 & 1) * base) + (tid & 1) - 2;
	int rightidx = base/2 + tid/2 + ((tid/2 & 1) * base) + (tid & 1);
	int gidin = (gid - blockDim.x) & lastidx;

	// Initialize temper. Kind of an unrolled for loop.  This is actually at
	// Timestep 0.

	temper[leftidx] = right[gid];
	temper[rightidx] = left[gidin];

    //k needs to insert the relevant left right values around the computed values
	//every timestep.  Since it grows larger the loop is reversed.
	for (int k = height-1; k>1; k--)
	{
		// This tells you if the current row is the first or second.
		shft_wr = base * ((k+1) & 1);
		// Read and write are opposite rows.
		shft_rd = base * (k & 1);

		if (tid1 < (base-k) && tid1 >= k)
		{
			temper[tid1 + shft_wr] = execFunc(temper[tid+shft_rd], temper[tid2+shft_rd], temper[tid1+shft_rd]);
		}
        __syncthreads();
	}

    if (gid == 0) temper[base] = temper[base+2];
    if (gid == lastidx) temper[2*blockDim.x+3] = temper[2*blockDim.x+1];

    temper[tid1] = execFunc(temper[tid+base], temper[tid2+base], temper[tid1+base]);

    IC[gid] = temper[tid1];
}

//Full refers to whether or not there is a node run on the CPU.
__global__ void wholeDiamond(REAL *right, REAL *left, bool full)
{

    extern __shared__ REAL temper[];

	//Same as upTriangle
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
    int lastidx = ((blockDim.x*gridDim.x)-1);
	int tid1 = tid + 1;
	int tid2 = tid + 2;
	int base = blockDim.x + 2;
	int height = base/2;
	int shft_rd;
	int shft_wr;
	int leftidx = base/2 - tid/2 + ((tid/2 & 1) * base) + (tid & 1) - 2;
	int rightidx = base/2 + tid/2 + ((tid/2 & 1) * base) + (tid & 1);
	// Initialize temper. Kind of an unrolled for loop.  This is actually at
	// Timestep 0.

    if (full)
    {
        int gidin = (gid - blockDim.x) & lastidx;
        temper[leftidx] = right[gid];
        temper[rightidx] = left[gidin];
    }
    else
    {
        int gidin = (gid + blockDim.x) & lastidx;
        temper[leftidx] = right[gidin];
        temper[rightidx] = left[gid];
    }

	for (int k = height-1; k>1; k--)
	{
        // This tells you if the current row is the first or second.
		shft_wr = base * ((k+1) & 1);
		// Read and write are opposite rows.
		shft_rd = base * (k & 1);

        if (tid1 < (base-k) && tid1 >= k)
		{
			temper[tid1 + shft_wr] = execFunc(temper[tid+shft_rd], temper[tid2+shft_rd], temper[tid1+shft_rd]);
		}
        __syncthreads();
	}

    //Boundary Conditions!
    if (full)
    {
        if (gid == 0) temper[base] = temper[base+2];
        if (gid == (blockDim.x*gridDim.x-1)) temper[2*blockDim.x+3] = temper[2*blockDim.x+1];
    }

    temper[tid] = execFunc(temper[tid+base], temper[tid2+base], temper[tid1+base]);

    __syncthreads(); // Then make sure each block of threads are synced.

    //-------------------TOP PART------------------------------------------

    leftidx = tid/2 + ((tid/2 & 1) * base) + (tid & 1);
    rightidx = (base - 1) + ((tid/2 & 1) * base) + (tid & 1) -  tid/2 - 1;

    int tidm = tid - 1;

	//The initial conditions are timeslice 0 so start k at 1.
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
			temper[tid + shft_wr] = execFunc(temper[tidm+shft_rd], temper[tid1+shft_rd], temper[tid+shft_rd]);
		}

		//Make sure the threads are synced
		__syncthreads();

	}

    right[gid] = temper[rightidx];
	left[gid] = temper[leftidx];

}

//Split one is always first.  Passing left like the downTriangle.  downTriangle
//should be rewritten so it isn't split.  Only write on a non split pass.
__global__ void splitDiamond(REAL *right, REAL *left)
{

    extern __shared__ REAL temper[];

	//Same as upTriangle
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
    int lastidx = ((blockDim.x*gridDim.x)-1);
	int tid1 = tid + 1;
	int tid2 = tid + 2;
	int base = blockDim.x + 2;
	int height = base/2;
	int shft_rd;
	int shft_wr;
	int leftidx = base/2 - tid/2 + ((tid/2 & 1) * base) + (tid & 1) - 2;
	int rightidx = base/2 + tid/2 + ((tid/2 & 1) * base) + (tid & 1);
    int gidin = (gid - blockDim.x) & lastidx;
	// Initialize temper. Kind of an unrolled for loop.  This is actually at
	// Timestep 0.

    temper[leftidx] = right[gidin];
	temper[rightidx] = left[gid];

    //Wind it up!
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
                    temper[tid + shft_wr] =execFunc(temper[tid+shft_rd], temper[tid+shft_rd], temper[tid1+shft_rd]);
                }
                else if (tid1 == height)
                {
                    temper[tid + shft_wr] = execFunc(temper[tid2+shft_rd], temper[tid2+shft_rd], temper[tid1+shft_rd]);
                }
                else
                {
                    temper[tid + shft_wr] = execFunc(temper[tid+shft_rd], temper[tid2+shft_rd], temper[tid1+shft_rd]);
                }
            }

        }

        __syncthreads();
    }

    //-------------------TOP PART------------------------------------------
    leftidx = tid/2 + ((tid/2 & 1) * base) + (tid & 1);
    rightidx = (base - 1) + ((tid/2 & 1) * base) + (tid & 1) -  tid/2 - 1;
    int tidm = tid - 1;

    //The initial conditions are timslice 0 so start k at 1.
	for (int k = 1; k<(height-1); k++)
	{
		//Bitwise even odd. On even iterations write to first row.
		shft_wr = base * (k & 1);
		//On even iterations write to second row (starts at element 32)
		shft_rd = base * ((k + 1) & 1);

		//Each iteration the triangle narrows.  When k = 1, 30 points are
		//computed, k = 2, 28 points.
        if (blockDim.x > 0)
        {
            if (tid < (blockDim.x-k) && tid >= k)
    		{
    			temper[tid + shft_wr] = execFunc(temper[tidm + shft_rd], temper[tid1 + shft_rd], temper[tid + shft_rd]);
    		}
        }
        else
        {
            if (tid < (blockDim.x-k) && tid >= k)
            {
                if (tid == (height - 2))
                {
                    temper[tid + shft_wr] = execFunc(temper[tidm + shft_rd], temper[tidm + shft_rd], temper[tid + shft_rd]);
                }
                else if (tid == (height - 1))
                {
                    temper[tid + shft_wr] = execFunc(temper[tid1 + shft_rd], temper[tid1 + shft_rd], temper[tid + shft_rd]);
                }
                else
                {
                    temper[tid + shft_wr] = execFunc(temper[tidm + shft_rd], temper[tid1 + shft_rd], temper[tid + shft_rd]);
                }
            }
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

//Do the split diamond on the CPU?
__host__ void CPU_diamond(REAL *right, REAL *left, int tpb)
{
    int idx;
    int ht = tpb/2;
    int *temper;
    int *leftidx, *rightidx;

    //malloc these!

    for (int k = 0; k<tpb; k++)
    {
        leftidx[k] = k/2 + ((k/2 & 1) * tpb) + (k & 1);
        rightidx[k] = (tpb - 1) + ((k/2 & 1) * tpb) + (k & 1) -  k/2 - 1;
    }

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
double
sweptWrapper(const int bks, int tpb, const int dv, REAL dt, const int t_end,
    const int cpu, REAL *IC, REAL *T_f)
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

	const size_t smem1 = 2*tpb*sizeof(REAL);
	const size_t smem2 = (2*tpb+4)*sizeof(REAL);

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
            wholeDiamond <<< bks-1,tpb,smem2 >>>(d_right,d_left,TRUE);

            cudaMemcpy(d_right, &right, tpb*sizeof(REAL), cudaMemcpyHostToDevice);
            cudaMemcpy(d_left, &left, tpb*sizeof(REAL), cudaMemcpyHostToDevice);

            wholeDiamond <<< bks,tpb,smem2 >>>(d_right,d_left,FALSE);

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

            /*
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

	if ((dv & (tpb-1) !=0) || (tpb&31) != 0)
    {
        cout << "INVALID NUMERIC INPUT!! "<< endl;
        cout << "2nd ARGUMENT MUST BE A POWER OF TWO >= 32 AND FIRST ARGUMENT MUST BE DIVISIBLE BY SECOND" << endl;
        exit(-1);
    }

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
