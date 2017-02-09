/*
Make some OMP versions of the code.
*/

//COMPILE LINE:
// nvcc -o ./bin/HeatOut Heat1D_SweptShared.cu -lm -O3 -fopenmp

#include <ostream>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <omp.h>

#define REAL        double
#define ONE         1.0
#define TWO         2.0

using namespace std;

REAL fou;

const REAL th_diff = 8.418e-5;

const REAL ds = 0.001;

int dv;
REAL dt, tf, freq;

REAL initFun(int xnode)
{
    REAL a = ((REAL)xnode*ds);
    return 100.f*a*(ONE-a/lx);
}

REAL stepForward(REAL tLeft, REAL tRight, REAL tCenter)
{
    return fou*(tLeft+tRight) + (ONE - TWO*fou) * tCenter;
}

//Classic Discretization wrapper.
double
classicWrapper(REAL *IC, ofstream &fwr)
{
    REAL *Texch;
    T_exch = (REAL *) malloc(dv*sizeof(REAL));
    double t_eq = 0.0;
    double twrite = freq;

    while (t_eq <= t_end)
    {
        T_exch[k] = stepForward(IC[k+1], IC[k+1], IC[k]);

        #omp parallel for
        for (int k=1; k<dv-1; k++) T_exch[k] = stepForward(IC[k-1], IC[k+1], IC[k]);

        T_exch[k] = stepForward(IC[k-1], IC[k-1], IC[k]);

        IC[k] = stepForward(T_exch[k+1], T_exch[k+1], T_exch[k]);

        #omp parallel for
        for (int k=1; k<dv-1; k++) IC[k] = stepForward(T_exch[k-1], T_exch[k+1], T_exch[k]);

        IC[k] = stepForward(T_exch[k-1], T_exch[k-1], T_exch[k]);

        t_eq += TWO*dt;

        if (t_eq > twrite)
        {
            fwr << " Temperature " << t_eq << " ";

            for (int k = 0; k<dv; k++)   fwr << IC[k] << " ";

            fwr << endl;

            twrite += freq;
        }
    }

    return t_eq;

}

int main(int argc, char *argv[])
{

    dv = atoi(argv[1]); //Number of spatial points
    dt =  atof(argv[2]);
	tf = atof(argv[3]); //Finish time
    freq = atof(argv[4]);
    const int scheme = atoi(argv[5]); //1 for Swept 0 for classic

    lx = ds * ((REAL)dv - 1.f);
    fou = th_diff*dt/(ds*ds);  //Fourier number

    cout << "Heat --- Fo: " << fou << endl;

	// Initialize arrays.
    REAL *IC;
    IC = (REAL *) malloc(dv*sizeof(REAL));
    
	for (int k = 0; k<dv; k++) IC[k] = initFun(k);

	// Call out the file before the loop and write out the initial condition.
	ofstream fwr;
	fwr.open(argv[6],ios::trunc);

	// Write out x length and then delta x and then delta t.
	// First item of each line is timestamp.
	fwr << lx << " " << dv << " " << ds << " " << endl << "Temperature " << 0.0 << " ";
	for (int k = 0; k<dv; k++) fwr << IC[k] << " ";
	fwr << endl;

	// Start the counter and start the clock.
	float timed;

	double tfm;

    cout << "Classic" << endl;
    tfm = classicWrapper(IC, fwr);

	// Show the time and write out the final condition.

	timed *= 1.e3;

    double n_timesteps = tfm/dt;

    double per_ts = timed/n_timesteps;

    cout << n_timesteps << " timesteps" << endl;
	cout << "Averaged " << per_ts << " microseconds (us) per timestep" << endl;

    if (argc>6)
    {
        ofstream ftime;
        ftime.open(argv[7],ios::app);
    	ftime << dv << "\t" << tpb << "\t" << per_ts << endl;
    	ftime.close();
    }

	fwr << "Temperature " << tfm << " ";
	for (int k = 0; k<dv; k++)	fwr << IC[k] << " ";

	fwr.close();

	// Free the memory and reset the device.

    Free(IC);
    Free(T_exchinal);

	return 0;
}
