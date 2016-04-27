
#include <cuda.h>
#include "cuda_runtime_api.h"
#include "device_functions.h"

#include <iostream>
#include <cmath>
#include <ostream>
#include <fstream>
#include <math.h>

using namespace std;

// Define Given Parameters
#define DIVISIONS  1024.
#define LENX       50.
#define TS         5.
#define TIMELIMIT  5000.
#define REAL       float
#define TH_DIFF    8.418e-5

__constant__ REAL F;


__global__ void upTriangle(REAL *IC)
{



}


int main()
{
	const int dv = int(DIVISIONS);
	REAL ds = LENX/(DIVISIONS-1);
	REAL FO = TS*TH_DIFF/(ds*ds);

	REAL x[dv], 
	REAL IC[dv];
	for (int k = 0; k<dv; k++)
	{
		x[k] = ds*k;
		IC[k] = 500.f*expf((-x[k])/LENX);
	}

	cout << FO << endl;
	return 1;

}








