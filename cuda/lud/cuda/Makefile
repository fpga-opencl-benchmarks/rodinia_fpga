include ../../../common/make.config

DEFS += \
		-DGPU_TIMER \
		$(SPACE)

CUDA_FLAGS += -I../common \
			 -use_fast_math \
			 -lm \
			 $(SPACE)

CFLAGS += -I../common \
					-I/usr/include/cuda \
		  -O3 \
		  -Wall \
		  $(SPACE)

# Add source files here
EXECUTABLE  := lud_cuda
# Cuda source files (compiled with cudacc)
CUFILES     := lud_kernel.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES     := lud.c lud_cuda.c ../common/common.c

OBJS = ../common/common.o lud.o lud_kernel.o

.PHONY: all clean 
all : $(EXECUTABLE)

.c.o : 
	$(CUDA_CC) $(CUDA_FLAGS) $(KERNEL_DIM) $(DEFS) -o $@ -c $< $(NVML_INC) $(NVML_LIB)

%.o:	%.cu 
	$(CUDA_CC) $(CUDA_FLAGS) $(KERNEL_DIM) $(DEFS) -o $@ -c $< $(NVML_INC) $(NVML_LIB)


$(EXECUTABLE) : $(OBJS)
	$(CUDA_CC) $(CUDA_FLAGS) -o $@  $? $(NVML_INC) $(NVML_LIB)

clean:
	rm -f $(EXECUTABLE) $(OBJS) *.linkinfo
