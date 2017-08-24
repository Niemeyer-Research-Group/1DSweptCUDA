# Simple makefile for all test problems for 1D Swept Rule
<<<<<<< HEAD
# -maxrregcount=73 -DGPUNUM=1

CUDAFLAGS = -gencode arch=compute_35,code=sm_35 -lm -restrict -O3 -Xcompiler -fopenmp --ptxas-options=-v 
=======
# -maxrregcount=73 -ccbin $(CLPATH2)

CUDAFLAGS = -ccbin "/cygdrive/c/Program Files (x86)/Microsoft Visual Studio 14.0/VC/bin" -gencode arch=compute_35,code=sm_35 -lm -restrict -O3 -Xcompiler -fopenmp --ptxas-options=-v
>>>>>>> 77579b6024b76ca374a63d1b6c1b7c862c51ec1b
CUDAFLAGSDOUBLE := $(CUDAFLAGS) -DREAL=double -DREALtwo=double2 -DREALthree=double3 -DREALfour=double4

# It will put the compiled programs in the bin subdirectory within the directory
# containing the makefile
PROJECT_DIR = $(shell pwd)
OUT_DIR := ./bin
#$(PROJECT_DIR)


default:	$(OUT_DIR)/HeatSingleOut $(OUT_DIR)/KSSingleOut $(OUT_DIR)/EulerSingleOut $(OUT_DIR)/HeatDoubleOut $(OUT_DIR)/KSDoubleOut $(OUT_DIR)/EulerDoubleOut

$(OUT_DIR)/HeatSingleOut:	Heat1D_SweptShared.cu
	nvcc -o $@ $< $(CUDAFLAGS)

$(OUT_DIR)/KSSingleOut:	KS1D_SweptShared.cu
	nvcc -o $@ $< $(CUDAFLAGS)

$(OUT_DIR)/EulerSingleOut:	Euler1D_SweptShared.cu
	nvcc -o $@ $< $(CUDAFLAGS)

$(OUT_DIR)/HeatDoubleOut:	Heat1D_SweptShared.cu
	nvcc -o $@ $< $(CUDAFLAGSDOUBLE)

$(OUT_DIR)/KSDoubleOut:	KS1D_SweptShared.cu
	nvcc -o $@ $< $(CUDAFLAGSDOUBLE)

$(OUT_DIR)/EulerDoubleOut:	Euler1D_SweptShared.cu
	nvcc -o $@ $< $(CUDAFLAGSDOUBLE)
# DO NOT DELETE
