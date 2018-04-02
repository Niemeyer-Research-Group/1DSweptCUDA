

CUDAFLAGS :=  -gencode arch=compute_35,code=sm_35 -restrict -O3 -Xcompiler -fopenmp --ptxas-options=-v
CUDAFLAGSDOUBLE := $(CUDAFLAGS) -DREAL=double -DREALtwo=double2 -DREALthree=double3 -DREALfour=double4

# It will put the compiled programs in the bin subdirectory 
PROJECT_DIR = $(shell pwd)
OUT_DIR := ./bin
#$(PROJECT_DIR)

EUL := Euler1D_SweptShared.cu
HT := Heat1D_SweptShared.cu
KS := KS1D_SweptShared.cu

default:	$(OUT_DIR)/HeatSingleOut $(OUT_DIR)/KSSingleOut $(OUT_DIR)/EulerSingleOut $(OUT_DIR)/HeatDoubleOut $(OUT_DIR)/KSDoubleOut $(OUT_DIR)/EulerDoubleOut

$(OUT_DIR)/HeatSingleOut:	$(HT)
	nvcc -o $@ $< $(CUDAFLAGS)

$(OUT_DIR)/KSSingleOut:	$(KS)
	nvcc -o $@ $< $(CUDAFLAGS)

$(OUT_DIR)/EulerSingleOut:	$(EUL)
	nvcc -o $@ $< $(CUDAFLAGS)

$(OUT_DIR)/HeatDoubleOut:	$(HT)
	nvcc -o $@ $< $(CUDAFLAGSDOUBLE)

$(OUT_DIR)/KSDoubleOut:	$(KS)
	nvcc -o $@ $< $(CUDAFLAGSDOUBLE)

$(OUT_DIR)/EulerDoubleOut:	$(EUL)
	nvcc -o $@ $< $(CUDAFLAGSDOUBLE)

