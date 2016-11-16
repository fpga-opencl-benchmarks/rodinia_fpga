include ../../common/make.config

all: euler3d euler3d_double pre_euler3d pre_euler3d_double

euler3d: euler3d.cu
	$(CUDA_CC) $(CUDA_FLAGS) $(KERNEL_DIM) -Xptxas -v euler3d.cu -o euler3d $(NVML_INC) $(NVML_LIB)

euler3d_double: euler3d_double.cu
	$(CUDA_CC) $(CUDA_FLAGS) $(KERNEL_DIM) -Xptxas -v euler3d_double.cu -o euler3d_double $(NVML_INC) $(NVML_LIB)

pre_euler3d: pre_euler3d.cu
	$(CUDA_CC) $(CUDA_FLAGS) $(KERNEL_DIM) -Xptxas -v pre_euler3d.cu -o pre_euler3d $(NVML_INC) $(NVML_LIB)

pre_euler3d_double: pre_euler3d_double.cu
	$(CUDA_CC) $(CUDA_FLAGS) $(KERNEL_DIM) -Xptxas -v pre_euler3d_double.cu -o pre_euler3d_double $(NVML_INC) $(NVML_LIB)

clean:
	rm -f euler3d euler3d_double pre_euler3d pre_euler3d_double *.linkinfo
