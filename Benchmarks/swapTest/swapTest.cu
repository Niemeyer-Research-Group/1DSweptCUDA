

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_functions.h>

#include <ostream>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>

using namespace std;

//Real is just going to have to be all four types (FLOAT,DOUBLE),(ONE,THREE).
#ifndef REAL
    #define REAL        float
    #define RSINGLE     float
#endif

int idxendh;

__constant__ int idxend;

__global__
void
swapKernel0(const REAL *passing_side, REAL *bin, int direction)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
    int gidout = (gid + direction*blockDim.x) & idxend;

    bin[gidout] = passing_side[gid];

}

__global__
void
swapKernel1(const REAL *passing_side, REAL *bin, int direction)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
    int gidout = (gid + direction*blockDim.x) & idxend;

    REAL a = passing_side[gid];

    bin[gidout] = a;

}

__global__
void
swapKernel2(const REAL *passing_side, REAL *bin, int direction)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
    int gdim = blockDim.x * gridDim.x;
    int gidout1 = (gid + direction*blockDim.x) & idxend;
    int gidout2 = (gid + gdim + direction*blockDim.x) & idxend;

    REAL a1 = passing_side[gid];
    REAL a2 = passing_side[gid+gdim];

    bin[gidout1] = a1;
    bin[gidout2] = a2;

}

__global__
void
swapKernel4(const REAL *passing_side, REAL *bin, int direction)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
    int gdim = blockDim.x * gridDim.x;
    int gidout1 = (gid + direction*blockDim.x) & idxend;
    int gidout2 = (gid + gdim + direction*blockDim.x) & idxend;
    int gidout3 = (gid + 2*gdim + direction*blockDim.x) & idxend;
    int gidout4 = (gid + 3*gdim + direction*blockDim.x) & idxend;

    REAL a1 = passing_side[gid];
    REAL a2 = passing_side[gid+gdim];
    REAL a3 = passing_side[gid+2*gdim];
    REAL a4 = passing_side[gid+3*gdim];

    bin[gidout1] = a1;
    bin[gidout2] = a2;
    bin[gidout3] = a3;
    bin[gidout4] = a4;
}

__global__
void
swapKernel8(const REAL *passing_side, REAL *bin, int direction)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
    int gdim = blockDim.x * gridDim.x;

    int gidout1 = (gid + direction*blockDim.x) & idxend;
    int gidout2 = (gid + gdim + direction*blockDim.x) & idxend;
    int gidout3 = (gid + 2*gdim + direction*blockDim.x) & idxend;
    int gidout4 = (gid + 3*gdim + direction*blockDim.x) & idxend;
    int gidout5 = (gid + 4*gdim + direction*blockDim.x) & idxend;
    int gidout6 = (gid + 5*gdim + direction*blockDim.x) & idxend;
    int gidout7 = (gid + 6*gdim + direction*blockDim.x) & idxend;
    int gidout8 = (gid + 7*gdim + direction*blockDim.x) & idxend;

    REAL a1 = passing_side[gid];
    REAL a2 = passing_side[gid+gdim];
    REAL a3 = passing_side[gid+2*gdim];
    REAL a4 = passing_side[gid+3*gdim];
    REAL a5 = passing_side[gid+4*gdim];
    REAL a6 = passing_side[gid+5*gdim];
    REAL a7 = passing_side[gid+6*gdim];
    REAL a8 = passing_side[gid+7*gdim];

    bin[gidout1] = a1;
    bin[gidout2] = a2;
    bin[gidout3] = a3;
    bin[gidout4] = a4;
    bin[gidout5] = a5;
    bin[gidout6] = a6;
    bin[gidout7] = a7;
    bin[gidout8] = a8;

}


int main( int argc, char *argv[] )
{

    const int dv = atoi(argv[1]); //Number of spatial points
    const int tpb = atoi(argv[2]); //Threads per Block
    const int bks = dv/tpb; //The number of blocks

    ofstream fwr;
    fwr.open(argv[3],ios::app);
    idxendh = dv-1;

    cudaMemcpyToSymbol(idxend,&idxendh,sizeof(int));

    REAL *initArray;
    cudaHostAlloc((void **) &initArray, dv*sizeof(REAL), cudaHostAllocDefault);

    for (int k = 0; k<dv; k++)
    {
        if (sizeof(REAL)<9)
        {
            initArray[k] = (RSINGLE)k;
        }
        else
        {
            initArray[k] = (RSINGLE)k;
        }
    }

    REAL *d_in, *d_out;
    cudaMalloc((void **)&d_in, sizeof(REAL)*dv);
    cudaMalloc((void **)&d_out, sizeof(REAL)*dv);
    cudaMemcpy(d_in,initArray,sizeof(REAL)*dv,cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
	float timed;
	cudaEventCreate( &start );
	cudaEventCreate( &stop );
	cudaEventRecord( start, 0);

    for (int k = 0; k<1.e5; k++)
    {
        swapKernel0 <<< bks,tpb >>> (d_in, d_out, ((k&1)*2-1));
        swapKernel0 <<< bks,tpb >>> (d_out, d_in, 0);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime( &timed, start, stop);

    fwr << 0 << "\t" << dv << "\t" << tpb << "\t" << timed << endl;
    cout << 0 << "\t" << dv << "\t" << tpb << "\t" << timed << endl;

    //HERE do it again, but don't reinitialize the host array, read it back to the device array.
	cudaEventRecord( start, 0);

    for (int k = 0; k<1.e5; k++)
    {
        swapKernel1 <<< bks,tpb >>> (d_in, d_out, ((k&1)*2-1));
        swapKernel1 <<< bks,tpb >>> (d_out, d_in, 0);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime( &timed, start, stop);

    cout << 1 << "\t" << dv << "\t" << tpb << "\t" << timed << endl;

    fwr << 1 << "\t" << dv << "\t" << tpb << "\t" << timed << endl;

    cudaEventRecord( start, 0);

    for (int k = 0; k<1.e5; k++)
    {
        swapKernel2 <<< bks/2,tpb >>> (d_in, d_out, ((k&1)*2-1));
        swapKernel2 <<< bks/2,tpb >>> (d_out, d_in, 0);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime( &timed, start, stop);

    cout << 2 << "\t" << dv << "\t" << tpb << "\t" << timed << endl;

    fwr << 2 << "\t" << dv << "\t" << tpb << "\t" << timed << endl;

    cudaEventRecord( start, 0);

    for (int k = 0; k<1.e5; k++)
    {
        swapKernel4 <<< bks/4,tpb >>> (d_in, d_out, ((k&1)*2-1));
        swapKernel4 <<< bks/4,tpb >>> (d_out, d_in, 0);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime( &timed, start, stop);

    cout << 4 << "\t" << dv << "\t" << tpb << "\t" << timed << endl;
    fwr << 4 << "\t" << dv << "\t" << tpb << "\t" << timed << endl;

    cudaEventRecord( start, 0);

    for (int k = 0; k<1.e5; k++)
    {
        swapKernel8 <<< bks/8,tpb >>> (d_in, d_out, ((k&1)*2-1));
        swapKernel8 <<< bks/8,tpb >>> (d_out, d_in, 0);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime( &timed, start, stop);

    cout << 8 << "\t" << dv << "\t" << tpb << "\t" << timed << endl;

    fwr << 8 << "\t" << dv << "\t" << tpb << "\t" << timed << endl;


    cudaDeviceSynchronize();

	cudaEventDestroy( start );
	cudaEventDestroy( stop );
    cudaDeviceReset();

    cudaFreeHost(initArray);
    cudaFree(d_in);
    cudaFree(d_out);
    // free(IC);
    // free(T_final);

	return 0;

}
