

#include <cuda.h>

#include <cmath>
#include <cstdlib>

//Perhaps nevermind the type difference.  It may be that we're not bringing the
//Derivative along.

__host__ void initFun(REAL xnode, REAL result)
{

    result = 2.f*cos(19.f*xnode/128);

}

__device void execFunc(REAL uLeft, REAL uRight, REAL uCenter)
{



}
__device__ REAL ks(REAL uLeft, REAL uRight, REAL uCenter)
{

    REAL cv = (uLeft*uLeft - uRight*uRight)/(4.f*dx);
    REAL df = ((uLeft+uLeft)+(uRight+uRight)-2.f*(uCenter+uCenter))/(dx*dx);
    return -(cv+df);

}
