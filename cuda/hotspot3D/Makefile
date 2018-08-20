include ../../common/make.config

SRC = hotspot3D.cu

EXE = hotspot3D 

OUTPUT = output.txt

FLAGS = -g -G #-arch sm_20 --ptxas-options=-v
release: $(SRC)
	$(CUDA_CC) $(KERNEL_DIM) $(FLAGS) $(SRC) -o $(EXE) $(CUDA_FLAGS) $(NVML_INC) $(NVML_LIB)

verify:
	$(CUDA_CC) $(KERNEL_DIM) $(FLAGS) $(SRC) -o $(EXE) $(CUDA_FLAGS) $(NVML_INC) $(NVML_LIB) -DVERIFY

enum: $(SRC)
	$(CUDA_CC) $(KERNEL_DIM) $(FLAGS) -deviceemu $(SRC) -o $(EXE) $(CUDA_FLAGS) $(NVML_INC) $(NVML_LIB)

debug: $(SRC)
	$(CUDA_CC) $(KERNEL_DIM) $(FLAGS) -g $(SRC) -o $(EXE) $(CUDA_FLAGS) $(NVML_INC) $(NVML_LIB)

debugenum: $(SRC)
	$(CUDA_CC) $(KERNEL_DIM) $(FLAGS) -g -deviceemu $(SRC) -o $(EXE) $(CUDA_FLAGS) $(NVML_INC) $(NVML_LIB)

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo $(OUTPUT) 
