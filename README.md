#1DSweptCUDA

Repository for 1D Swept rule scheme written in CUDA C.
*This project was developed on the Ubuntu 16.04 OS.  All programs are designed to be run on the command line.  A UNIX OS is strongly recommended*

##CUDA instructions
1. install the cuda toolkit v7.5.  Instructions for all OS are on the sidebar here: http://docs.nvidia.com/cuda/index.html

2. Find your GPU compute capability with the deviceQuery.cu program in the utilities folder in the samples folder included in the CUDA toolkit installation.
Linux Path: usr/local/cuda-7.5/samples/1_Utilities
Navigate to the folder and call make or call make in the samples folder to make all the samples and run the deviceQuery program.

3. Open the Makefile in the SharedSwept folder and change the compute_ and sm_ numbers in CUDAFLAGS to your compute capability.

4. Copy the files in the samples/common/inc folder to your default include path or add this folder to your path.  Euler requires these headers.

##Swept program instructions
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
