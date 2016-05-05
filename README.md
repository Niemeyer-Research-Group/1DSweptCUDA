# 1DSweptCUDA
Repository for 1D Swept rule scheme written in CUDA C.

General Instructions:
First, install the cuda toolkit v7.5.  Instructions for all OS are on the
sidebar here: http://docs.nvidia.com/cuda/index.html

Next, find your GPU compute capability with the Devicequery.cu program in the CUDA
samples included in the toolkit installation.  It should be in the Utilities folder.
You can run this program on a Unix machine from the command line with:
nvcc -o [Executablefilename] Devicequery.cu.  On a windows machine, open visual studio
version must be pre 2015, install the cuda runtime option for new solutions, make a new
solution and copy the code over into the .cu file.

To run the Swept program on a unix machine you can either compile from the command line
as shown above, or run the CMakeLists.txt with cmake to generate a makefile.  Before doing this,
change the compiler flags -arch = compute_##  and sm_## to the compute capability of your GPU
in the Cmake file.  Don't include the decimal.  Then run the makefile and the executable.

-5/5/2016
Right now, only the ...CUDA_main.cu works.  Everything else is a work in progress.
So the compilation is very simple, since all the functions and the main routine
are in that file.  The ...Register.cu file is a routine to eliminate all the shared memory
and use one thread per triangle/diamond.  The ...Triangle.cu routine is simply a copy
of the functions in the main file.  There are currently no diamonds only triangles.

NOTE:  Do not compile with an optimizer option, it causes an error in memcpy.  I haven't
figured out why yet.
