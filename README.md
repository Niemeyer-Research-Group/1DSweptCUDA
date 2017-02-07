# 1DSweptCUDA

Repository for 1D Swept rule written in CUDA.

*This project was developed on the Ubuntu 16.04 OS.  All programs are designed to be executed via the terminal.  A UNIX OS is strongly recommended*

## CUDA instructions

1. Install the cuda toolkit v8.0 or 7.5. 

    [OS installation guides](http://docs.nvidia.com/cuda/index.html#installation-guides) 
   
    [Toolkit Download](https://developer.nvidia.com/cuda-downloads)
    
2. If you don't know your GPU's compute capability, find it with the deviceQuery.cu program in the utilities folder in the samples folder included in the CUDA toolkit installation.
Linux Path: /usr/local/cuda/samples/1_Utilities
Navigate to the folder, run make and run the deviceQuery program.

3. Open the Makefile in the SweptSource folder and change the compute_ and sm_ numbers in CUDAFLAGS to your compute capability.

4. Make sure the nvcc compiler is on your shell path.  On Linux it should be in: /usr/local/cuda/bin
The path settings I use can be found in examplerc.txt.

## Swept program instructions

It's recommended that the user begin by running the plotSolutions.py python script from the command line in the source code directory.
This will create the bin subdirectory for the executables, compile all the programs and plot the result of the program.

In addition, the programs may be compiled using make from the command line or individually.
All source files have a compile command in the header comments.

All programs can be run from the terminal with the same command:

| Argument  |  Meaning |
| --------- | -------- |
|./bin/NamePrecisionOut  |  The path to the executable.  Executables are identified by the problem name and precision level and reside in bin folder|
| #SpatialPoints | Number of points in the spatial domain (must be a power of 2 greater than 32)
| #ThreadsPerBlock | Number of points per node = threads per block (must be power of 2 greater than 32 less than #Spatial points)
| timestep(s) | Will give error if relevant ratio (dt/dx) is unstable.
| finishtime(s) | Ending time for the simulation.  Approximate for swept rule.
| outputfrequency(s) | Checkpoint at which program writes out simulation results (i.e. every 400s).  Example: if the finish time is 1000s and the output frequency is 400s, the program will write out the initial condition and the solution at two intermediate times (~400 and 800s)
| algorithm | 0 for classic, 1 for swept
| variant | 0 for standard shared memory swept, 1 for variant
| SolutionfilePath | Path should point to .dat folder in Results file. Name should include problem and precision. Format: Line 1 is length of full spatial domain, number of spatial points and grid step, Other rows are results with format: variable, time, value at each spatial point.
| TimingFilePath (Optional) | Path should point to .txt folder in Results file. Title should include problem, precision and algorithm.  Format is #SpatialPoints, #ThreadsPerBlock, us per timestep.

## Directory Structure
* SweptSource 
    * Contains the relevant source code for this project.  
        * plotSolutions.py: Plot simulation results.
        * sweptPerformanceTest.py: Conduct a performance test on a single problem, precision and algorithm over a range of problem sizes and launch configurations.
        * quickperform.py: Run all performance tests.
    * All output from CUDA programs is kept in the Results folder.  
        * perfAnalysis: Parse timing output for all problems and produce performance plots.  Plots and tables saved in Result plots folder in top level.
    * Testing folder for accuracy tests of all problems.    
* Result plots: All plots for accuracy, simulation results, swept rule visualization, and performance results. 
* Other top level folders
    * Scratch pad for development of algorithms, unit tests, and intermediate versions to show effect of development process on performance.

## Test Problem Discretizations
[This Document](1_D_swept_equations.pdf) explains the numerical methods used in these programs

## Swept Scheme
[This Document](Swept_1_D_Scheme_Description.pdf) explains the swept algorithm and it's motivation.

## Dependencies
### Hardware
* An nVidia GPU

### Python
* Python version 2.7
* Anaconda for python 2.7
* [exactpack](https://github.com/losalamos/ExactPack)
* [palettable](https://jiffyclub.github.io/palettable/#palette-interface)

### CUDA
* Cuda 7.5 or 8.0
* gcc and g++
* OpenMP

### Command line
* ffmpeg (for gif creation, not essential)

## Additional Notes
* CUDA 7.5 requires older versions of gcc and g++ (pre 4.4).  If using CUDA 7.5 install old versions and use update-alternatives.
* Use the nvidia-smi command to find the ID of your GPU device if there is more than one in the workstation.  If the desired computation GPU is not ID 0 change the number in the cudaSetDevice call at the start of the main function in each source to the desired GPU ID.
