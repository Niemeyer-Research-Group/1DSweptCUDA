//

#include <cmath>
#include <cstdlib>
#include <cuda.h>


// Maybe use float2 to store 2nd derivative with velocity.
__device__ REAL Kuramoto_Sivashinsky(REAL uLeft, REAL uRight, REAL uCenter, REAL uxxLeft, REAL uxxRight, REAL uxxCenter)
{

    REAL cv = (uLeft*uLeft - uRight*uRight)/(4.f*dx);
    REAL df = ((uLeft+uxxLeft)+(uRight+uxxRight)+(uCenter+uxxCenter))/(dx*dx);
    return -(cv+df);

}

__device__ REAL Heat_Diffusion(REAL tLeft, REAL tRight, REAL tCenter)
{

    return fo*(tLeft+tRight) + (1.f-2.f*fo)*tCenter;

}
