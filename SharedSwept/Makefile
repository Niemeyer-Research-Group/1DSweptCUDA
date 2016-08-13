# Simple makefile for all test problems for 1D Swept Rule

CUDAFLAGS = -gencode arch=compute_35,code=sm_35 -lm -restrict -Xcompiler -fopenmp

# It will put the compiled programs in the bin subdirectory within the directory
# containing the makefile
PROJECT_DIR = $(shell pwd)
OUT_DIR := $(PROJECT_DIR)/bin

print-%  : ; @echo $* = $($*)

default:	$(OUT_DIR)/HeatOut $(OUT_DIR)/KSOut $(OUT_DIR)/EulerOut

$(OUT_DIR)/HeatOut:	Heat1D_SweptShared.cu
	nvcc -o $@ $< $(CUDAFLAGS)

$(OUT_DIR)/KSOut:	KS1D_SweptShared.cu
	nvcc -o $@ $< $(CUDAFLAGS)

$(OUT_DIR)/EulerOut:	Euler1D_SweptShared.cu
	nvcc -o $@ $< $(CUDAFLAGS)
