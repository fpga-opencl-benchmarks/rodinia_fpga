include ../../../common/make.config


SRC = srad.cu

EXE = srad

release: $(SRC)
	$(CUDA_CC) $(KERNEL_DIM) $(SRC) -o $(EXE) $(CUDA_FLAGS)

enum: $(SRC)
	$(CUDA_CC) $(KERNEL_DIM) -deviceemu $(SRC) -o $(EXE) $(CUDA_FLAGS)

debug: $(SRC)
	$(CUDA_CC) $(KERNEL_DIM) -g $(SRC) -o $(EXE) $(CUDA_FLAGS)

debugenum: $(SRC)
	$(CUDA_CC) $(KERNEL_DIM) -g -deviceemu $(SRC) -o $(EXE) $(CUDA_FLAGS)

clean: $(SRC)
	$(RM) $(EXE) $(EXE).linkinfo result.txt
