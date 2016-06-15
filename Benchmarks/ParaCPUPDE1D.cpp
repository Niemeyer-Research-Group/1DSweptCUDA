#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <ostream>
#include <cstring>
#include <fstream>
#include <omp.h>

using namespace std;

#define REAL       float

#ifndef DT
#define DT         .01
#endif

#define TH_DIFF     8.418e-5

#ifndef FINISH
#define FINISH		1e4
#endif

#ifndef NUMT
#define NUMT        8
#endif

#ifndef INTERVAL
#define INTERVAL    FINISH
#endif

#ifndef DIVISIONS
#define DIVISIONS   1024
#endif

const int dv = DIVISIONS-1; //Setting it to an int helps with arrays
const int dv2 = dv-1 ; //The number of blocks since threads/block = 32.
//Threads/block will be experimented on.
const REAL lx = 5.0*DIVISIONS/1024;
const REAL ds = lx/dv; //The x division length.
const REAL fo = TS*TH_DIFF/(ds*ds); //The Fourier number.

int NewRadicals(REAL left, REAL center, REAL right)
{
    return fo * (left + right) + (1.f-2.f*fo) * center;
}

void intialCond(REAL *give, int k)
{
    give[k] = 500.f*expf((-ds*k)/lx);
}

int main()
{
    ofstream ftime;
    ftime.open("GPUBenchTiming.txt",ios::app);

    omp_set_num_threads( NUMT );
    REAL *give = new REAL[DIVISIONS],
    REAL *get = new REAL[DIVISIONS];
    REAL t_eq = 0.f;

    #pragma omp parallel for
    for (int k = 0, k<DIVISIONS, k++)
    {
        initialCond(give[k],k);
    }

    double time0 = omp_get_wtime();
    while(t_eq < FINISH)
    {

        get[0] = 2.f * fo * give[1] + (1.f-2.f*fo) * give[0];
        get[dv] = 2.f * fo * give[dv2] + (1.f-2.f*fo) * give[dv];

        #pragma omp parallel for
        for (unsigned int k = 1, k<(DIVISIONS-1), k++)
        {
            get[k] = NewRadicals(give[k-1], give[k], give[k+1]);
        }
        t_eq += DT;
        memcpy(give, get, DIVISIONS*sizeof(REAL));
    }

    double time1 = omp_get_wtime(); // Seconds it takes.
    double tf = (time1-time0); // Seconds to complete task

    ftime << tf << endl;
    ftime.close();
    cout << "That took: " << tf << " seconds" << endl;

    return 0;

}
