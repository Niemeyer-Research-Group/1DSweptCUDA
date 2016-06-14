

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

#define REAL       float

#ifndef DT
#define DT         .01
#endif

#define TH_DIFF     8.418e-5

#ifndef FINISH
#define FINISH		1e4
#endif

#ifndef THREADBLK
#define THREADBLK   32
#endif

#ifndef INTERVAL
#define INTERVAL    FINISH
#endif

#ifndef DIVISIONS
#define DIVISIONS   1024
#endif


// Declare constant Fourier number that will go in Device constant memory.
__constant__ REAL fo;


__global__ void NewRadicals(float *give, float *get)
{

    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid == 0)
    {
        get[gid] = fo * (2.f * give[gid+1]) + (1.f-2.f*fo) * give[gid];
    }
    else if (gid == gridDim.x*blockDim.x-1)
    {
        get[gid] = fo * (2 * give[gid-1]) + (1.f-2.f*fo) * give[gid];
    }
    else
    {
        get[gid] = fo * (give[gid-1] + give[gid+1]) + (1.f-2.f*fo) * give[gid];
    }

    give[gid] = get[gid];

}

int main()
{

    //Choose the GPGPU.  This is device 0 in my machine which has 2 devices.
	cudaSetDevice(0);
	const int dv = DIVISIONS; //Setting it to an int helps with arrays
	const int bks = DIVISIONS/THREADBLK; //The number of blocks.
	// Threads/block will be experimented on.
	const REAL lx = 5.f*DIVISIONS/1024;
	const REAL ds = lx/((REAL)DIVISIONS-1.f); //The x division length.
	REAL fou = DT*TH_DIFF/(ds*ds); //The Fourier number.
    REAL T_final[dv];

    ofstream fwr;
    ofstream ftime;
    ftime.open("GPUBenchTiming.txt",ios::app);
    fwr.open("1DHeatEQResult.dat",ios::trunc);
    // Write out x length and then delta x and then delta t.
    // First item of each line is timestamp.
    fwr << lx << " " << DIVISIONS << " " << DT << " " << endl << 0 << " ";

    #ifdef UNIFIED
    REAL *d_gv, *d_get;

    cudaMallocManaged(&d_gv,sizeof(REAL)*dv);
    cudaMallocManaged(&d_get,sizeof(REAL)*dv);

    for (int k = 0; k<dv; k++)
    {
        d_gv[k] = 500.f*expf((-ds*k)/lx); //*sin((float)k/10.f)
    }

    for (int k = 0; k<dv; k++)
    {
        fwr << d_gv[k] << " ";
    }

    fwr << endl;

    #else
	REAL h_gv[dv];

	REAL *d_gv, *d_get;

	// Some initial condition for the bar temperature, an exponential decay
	// function.

	for (int k = 0; k<dv; k++)
	{
		h_gv[k] = 500.f*expf((-ds*k)/lx); //*sin((float)k/10.f)
	}

	for (int k = 0; k<dv; k++)
	{
		fwr << h_gv[k] << " ";
	}

	fwr << endl;

    // Transfer data to GPU.

    // This puts the Fourier number in constant memory.
    cudaMemcpyToSymbol(fo,&fou,sizeof(REAL));

    // This initializes the device arrays on the device in global memory.
    // They're all the same size.  Conveniently.
    cudaMalloc((void **)&d_gv, sizeof(REAL)*dv);
    cudaMalloc((void **)&d_get, sizeof(REAL)*dv);

    //Copy the initial conditions to the device array.
    cudaMemcpy(d_gv,h_gv,sizeof(REAL)*dv,cudaMemcpyHostToDevice);

    #endif

    // Start the counter and start the clock.
    REAL t_eq = 0.f;
    REAL t_fin = (REAL)INTERVAL;
    double wall0 = clock();

    // Call the kernels until you reach the iteration limit.
    while(t_eq < FINISH)
    {

        NewRadicals <<< bks,THREADBLK >>>(d_gv,d_get);
        t_eq += DT;

        // if (t_eq>=t_fin)
        // {
        //     cudaMemcpy(T_final, d_get, sizeof(REAL)*dv, cudaMemcpyDeviceToHost);
        //     fwr << t_eq << " ";
        //     for (int k = 0; k<dv; k++)
        //     {
        //         fwr << T_final[k] << " ";
        //     }
        //     fwr << endl;
        //     t_fin += INTERVAL;
        // }

    }

    fwr.close();
    //Show the time and write out the final condition.
    double wall1 = clock();
    double timed = (wall1-wall0)/CLOCKS_PER_SEC;

    ftime << timed << endl;
    ftime.close();
    cout << "That took: " << timed << " seconds" << endl;

    //Free the memory and reset the device.
    cudaFree(d_gv);
    cudaDeviceReset();

    return 0;

}
