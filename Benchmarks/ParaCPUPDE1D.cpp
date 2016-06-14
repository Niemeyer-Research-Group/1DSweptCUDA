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

const int dv = int(DIVISIONS); //Setting it to an int helps with arrays
const int bks = dv/THREADBLK; //The number of blocks since threads/block = 32.
//Threads/block will be experimented on.
const REAL lx = 5.0*DIVISIONS/1024;
const REAL ds = lx/((double)DIVISIONS-1.0); //The x division length.
const REAL fo = TS*TH_DIFF/(ds*ds); //The Fourier number.
