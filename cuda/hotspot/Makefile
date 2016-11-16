include ../../common/make.config

SRC = hotspot.cu

EXE = hotspot

release: $(SRC)
	$(CUDA_CC) $(KERNEL_DIM) $(SRC) -o $(EXE)  $(CUDA_FLAGS) $(NVML_INC) $(NVML_LIB)

enum: $(SRC)
	$(CUDA_CC) $(KERNEL_DIM) -deviceemu $(SRC) -o $(EXE)  $(CUDA_FLAGS) $(NVML_INC) $(NVML_LIB) 

debug: $(SRC)
	$(CUDA_CC) $(KERNEL_DIM) -g $(SRC) -o $(EXE)  $(CUDA_FLAGS) $(NVML_INC) $(NVML_LIB) 

debugenum: $(SRC)
	$(CUDA_CC) $(KERNEL_DIM) -g -deviceemu $(SRC) -o $(EXE)  $(CUDA_FLAGS) $(NVML_INC) $(NVML_LIB) 

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt
