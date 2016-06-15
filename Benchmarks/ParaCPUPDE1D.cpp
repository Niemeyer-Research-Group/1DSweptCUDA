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
const REAL lx = 5.f*(REAL)DIVISIONS/1024;
const REAL ds = lx/(REAL)dv; //The x division length.
const REAL fo = DT*TH_DIFF/(ds*ds); //The Fourier number.


REAL NewRadicals(REAL left, REAL center, REAL right)
{
    return fo * (left + right) + (1.f-2.f*fo) * center;
}


int main()
{
    ofstream ftime;
    ftime.open("CPUBenchTiming.txt",ios::app);
    ofstream fwr;
    fwr.open("1DHeatEQResult.dat",ios::trunc);
    fwr << lx << " " << DIVISIONS << " " << DT << " " << endl << 0 << " ";
    omp_set_num_threads( NUMT );

    REAL *give = (REAL *) malloc(DIVISIONS*sizeof(REAL));
    REAL *get = (REAL *) malloc(DIVISIONS*sizeof(REAL));
    REAL t_eq = 0.f;

    cout << NUMT << " " << DIVISIONS << endl;

    #pragma omp parallel for
    for (int k = 0; k<DIVISIONS; k++)
    {
        give[k] = 500.f*expf((-ds*k)/lx);
    }

    for (int k = 0; k<DIVISIONS; k++)
    {
        fwr << give[k] << " ";
    }

    fwr << endl;

    double time0 = omp_get_wtime();
    while(t_eq < FINISH)
    {

        get[0] = 2.f * fo * give[1] + (1.f-2.f*fo) * give[0];
        get[dv] = 2.f * fo * give[dv2] + (1.f-2.f*fo) * give[dv];

        #pragma omp parallel for default(none),shared(get,give),private(k),schedule(static)
        for (int k = 1; k<dv; k++)
        {
            get[k] = NewRadicals(give[k-1], give[k], give[k+1]);
        }
        t_eq += DT;
        memcpy(give, get, DIVISIONS*sizeof(REAL));

    }

    double time1 = omp_get_wtime(); // Seconds it takes.
    double tf = (time1-time0); // Seconds to complete task

    fwr << t_eq << " ";

    for (int k = 0; k<DIVISIONS; k++)
    {
        fwr << give[k] << " ";
    }

    fwr << endl;

    fwr.close();

    ftime << tf << endl;
    ftime.close();
    cout << "That took: " << tf << " seconds" << endl;

    free(give);
    free(get);
    return 0;

}
