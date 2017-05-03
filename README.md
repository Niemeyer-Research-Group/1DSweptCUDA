[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.570984.svg)](https://doi.org/10.5281/zenodo.570984)

# 1DSweptCUDA

Repository for 1D Swept rule written in CUDA.

*This project was developed on the Ubuntu 16.04 OS.  All programs are designed to be executed via the terminal.  A UNIX OS is strongly recommended*

## CUDA instructions

1. Install the cuda toolkit v8.0 or 7.5. 

    [OS installation guides](http://docs.nvidia.com/cuda/index.html#installation-guides) 
   
    [Toolkit Download](https://developer.nvidia.com/cuda-downloads)
    
2. If you don't know your GPU's compute capability, find it with the deviceQuery.cu program in the utilities folder in the samples folder included in the CUDA toolkit installation.
Linux Path: /usr/local/cuda/samples/1_Utilities
Navigate to the folder, 'make' the executable and run the deviceQuery program.

    __OR__ if MATLAB is available, you can open matlab and type gpuDevice in the command line.  You can also see the number of GPUs in your environment with gpuDeviceCount and feed the index to gpuDevice to query individual GPUs.

    If using a GPU numbered other than 0: change the GPUNUM preprocessor variable in each source code file.

3. Open the Makefile in the SweptSource folder and change the compute_ and sm_ numbers in CUDAFLAGS to your compute capability.

4. Make sure the nvcc compiler is on your shell path.  On Linux it should be in: /usr/local/cuda/bin
The path settings I use can be found in examplerc.txt.

## Swept program instructions

It's recommended that the user begin by running the plotSolutions.py python script from the command line in the SweptSource directory.
This will create the SweptSource/bin subdirectory for the executables, compile all the programs and plot the result of the program with the conditions you select in the GUI.

In addition, the programs may be compiled using make from the command line or individually.
All source files have an example of a compile command in the docstring.

All programs can be run from the terminal with the same command:

| Argument  |  Meaning |
| --------- | -------- |
|./bin/NamePrecisionOut  |  The path to the executable.  Executables are identified by the problem name and precision level and reside in bin folder|
| #SpatialPoints | Number of points in the spatial domain (must be a power of 2 greater than 32)
| #ThreadsPerBlock | Number of points per node = threads per block (must be power of 2 greater than 32 less than #Spatial points)
| timestep(s) | Will give error if relevant ratio (dt/dx) is unstable.
| finishtime(s) | Ending time for the simulation.  Approximate for swept rule.
| outputfrequency(s) | Checkpoint at which program writes out simulation results (i.e. every 400s).  Example: if the finish time is 1000s and the output frequency is 400s, the program will write out the initial condition and the solution at two intermediate times (~400 and 800s)
| algorithm | 0 for classic, 1 for shared swept, 2 for variant (hybrid or register)
| SolutionfilePath | Path should point to .dat folder in Results file. Name should include problem and precision. Format: Line 1 is length of full spatial domain, number of spatial points and grid step, Other rows are results with format: variable, time, value at each spatial point.
| TimingFilePath (Optional) | Path should point to .txt folder in Results file. Title should include problem, precision and algorithm.  Format is #SpatialPoints, #ThreadsPerBlock, us per timestep.

## Directory Structure
* __SweptSource__
    * Contains the relevant source code for this project (.cu) files.  
    * _myVectorTypes.h_: Modifies the CUDA VectorTypes header to allow vector operations on double3 and 4 data types.
    * _plotSolutions.py_: Plot simulation results with GUI.
    * _sweptPerformanceTest.py_: Conduct a performance test on a single problem, precision and algorithm over a range of problem sizes and launch configurations.
    * _quickperform.py_: Runs a full performance test suite.  Takes a command line argument for # of test suites to run, defaults to 1.
    * __Results__ 
        * Contains results of performance test (as .txt and .h5) and results of simulation (as .dat) files. 
        * _perfAnalysis.py_: Parses results of performance tests and makes informative plots in __ResultPlots__ folder.
    * __Testing__ 
        * _KSDouble_Official.txt_: The official version of the KS result.  Run with very small (10^-8) timestep.  KS has no analytical solution so this is the basis for accuracy judgements.
        * _testProcedure.py_: Python script that determines the consistency and accuracy of the programs.  Consistency is measured by comparing the results of the versions (classic, swept, variant) to each other and cataloguing the differences in ResultPlots/ExactTesting/cinsistency.out.  Accuracy is measured against analytic solutions (Heat and Euler Equations) or small timestep solution (KS equation.)  Error plots available in ResultPlots/ExactTesting/.
* __ResultPlots__: 
    * All plots for accuracy, simulation results, swept rule visualization, and performance results.  See [the plot readme](ResultPlots/PlotREADME.md) for further details.
* __pyAnalysisTools__
    * _Analysis_help.py_: Classes and functions for post-processing and analysis of performance.
    * _main_help.py_: Classes and functions for collecting and parsing raw performance data.
    * _result_help.py_: Classes and functions for parsing and plotting simulation data.
* __docs__
    * Explanation of decomposition, discretization, and environment.

## Test Problem Discretizations
[This Document](docs/1_D_swept_equations.pdf) explains the numerical methods used in these programs

## Swept Scheme
[This Document](docs/Swept_1_D_Scheme_Description.pdf) explains the swept algorithm and it's motivation.

## Test Hardware
GPU Performance Tests were run on an NVIDIA Tesla K40c GPGPU
CPU Performance Tests were run on a single Intel Xeon E5-2630 @ 2.4 GHz with 8 cores and max 16 threads.

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
