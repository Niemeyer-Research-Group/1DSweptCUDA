

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
#define DT         .01
#define REAL       float

#define TH_DIFF    8.418e-5
#ifndef FINISH
#define FINISH		1e4
#endif

// Declare constant Fourier number that will go in Device constant memory.
__constant__ REAL fo;


__global__ void NewRadicals(float *give, float ti, float ts)
{

    REAL get = new REAL [DIVISIONS];
    gid = 1 + threadIdx.x + blockIdx.x * blockDim.x;
    get[gid] = fo * (give[gid-1] + give[gid+1]) + (1.f-2.f*fo) * give[gid];
    ti += DT;
    for (k = ti; k<ts; k += DT)
    {
        get[gid] = fo * (get[gid+1] + get[gid-1]) + (1.f-2.f*fo) * get[gid];
    }

    give[gid] = get[gid];
    delete[] get;

}

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
