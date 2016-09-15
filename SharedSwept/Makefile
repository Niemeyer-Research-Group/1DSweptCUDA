# Simple makefile for all test problems for 1D Swept Rule

CUDAFLAGS = -gencode arch=compute_35,code=sm_35 -lm -restrict -Xcompiler -fopenmp -Xptxas -v
CUDAFLAGSDOUBLE := $(CUDAFLAGS) -DREAL=double -DREALthree=double3 -DREALfour=double4

# It will put the compiled programs in the bin subdirectory within the directory
# containing the makefile
PROJECT_DIR = $(shell pwd)
OUT_DIR := $(PROJECT_DIR)/bin

default:	$(OUT_DIR)/HeatSingleOut $(OUT_DIR)/KSSingleOut $(OUT_DIR)/EulerSingleOut \
	 $(OUT_DIR)/HeatDoubleOut $(OUT_DIR)/KSDoubleOut $(OUT_DIR)/EulerDoubleOut

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
