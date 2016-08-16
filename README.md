#1DSweptCUDA

Repository for 1D Swept rule scheme written in CUDA C.

*This project was developed on the Ubuntu 16.04 OS.  All programs are designed to be run on the command line.  A UNIX OS is strongly recommended*

##CUDA instructions

1. install the cuda toolkit v7.5.  Instructions for all OS are on the sidebar here: http://docs.nvidia.com/cuda/index.html

2. Find your GPU compute capability with the deviceQuery.cu program in the utilities folder in the samples folder included in the CUDA toolkit installation.
Linux Path: /usr/local/cuda-7.5/samples/1_Utilities
Navigate to the folder and call make or call make in the samples folder to make all the samples and run the deviceQuery program.

3. Open the Makefile in the SharedSwept folder and change the compute_ and sm_ numbers in CUDAFLAGS to your compute capability.

4. Copy the files in the samples/common/inc folder to your default include path or add this folder to your path.  Euler requires these headers.

5. Make sure the nvcc compiler is on your shell path.  On Linux it should be in: /usr/local/cuda-7.5/bin

##Swept program instructions

As of 8/15/2016 only the source files in /SharedSwept are complete.
It's recommended that the user begin by running the performance plot python script from the command line in the source code directory i.e. /SharedSwept.
This will create the bin subdirectory for the executables, compile all the programs and plot the result of the program.

In addition, the programs may be compiled using make from the command line or individually.
Line 20 of all source files is a compile command for that file.  

All programs can be run from the command line with the same arguments:
./bin/NameOut #SpatialPoints  #ThreadsPerBlock  timestep(s)  finishtime(s)  outputfrequency(s)  swept/classic  CPUshare/allGPU  SolutionfilePath  TimingFilePath (Optional)

Output frequency tells program to output the solution every set number of seconds in simulation time.
For example, if the finish time is 1000s and the output frequency is 400s, the program will write out the initial condition and the solution at two intermediate times (~400 and 800s) and when it finishes.
It will not write out the solution at the exact times requested because it can only write out the solution at a certain point in the cycle.

Output files are placed in the Results subdirectory.  Solution files are .dat, timing files are .txt.

The KS equation and all classic discretizations do not have CPUshare options, so if a 1 is given it will simply ignore it.

##Test Problem Discretizations
1_D_swept_equations.pdf explains the numerical methods used in these programs
