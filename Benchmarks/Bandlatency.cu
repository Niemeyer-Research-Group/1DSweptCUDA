#include <cuda.h>

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <ostream>
#include <fstream>

using namespace std;

#define END 1e7

int main()
{
	// Choose the GPGPU.  This is device 0 in my machine which has 2 devices.
	cudaSetDevice(0);
    ofstream fwr;
    fwr.open("BandwithvsLatency.txt",ios::trunc);

    float *d_IC;
    float *IC;
    double wall0, wall1, timed;
    cudaMalloc((void **)&d_IC, sizeof(float)*END);

    for(int k = 1; k<END; k*=2)
    {
        IC = (float *) malloc(k*sizeof(float));
        for(int n = 0; n<k; n++)
        {
            IC[n] = k;
        }

        wall0 = clock();

        cudaMemcpy(d_IC,IC,sizeof(float)*k,cudaMemcpyHostToDevice);

        wall1 = clock();
        timed = (wall1-wall0)/CLOCKS_PER_SEC*1000000.f;
        fwr << k << " " << timed << endl;


    }

    free(IC);
    cudaFree(d_IC);
    cudaDeviceReset();

    return 0;
}
