include ../../common/make.config

ifdef FOR
	CUDA_FLAGS += -DFOR
endif

SRC = pathfinder.cu
EXE = pathfinder

release:
	$(CUDA_CC) $(SRC) -o $(EXE) $(CUDA_FLAGS) $(NVML_INC) $(NVML_LIB)

enum:
	$(CUDA_CC) -deviceemu $(SRC) -o $(EXE) $(CUDA_FLAGS) $(NVML_INC) $(NVML_LIB)

debug:
	$(CUDA_CC) -g $(SRC) -o $(EXE) $(CUDA_FLAGS) $(NVML_INC) $(NVML_LIB)

debugenum:
	$(CUDA_CC) -g -deviceemu $(SRC) -o $(EXE) $(CUDA_FLAGS) $(NVML_INC) $(NVML_LIB)

clean:
	rm -f pathfinder
