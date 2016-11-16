include ../../common/make.config

SRC = bfs.cu

EXE = bfs

release: $(SRC)
	$(CUDA_CC) $(SRC) -o $(EXE) $(CUDA_FLAGS)

enum: $(SRC)
	$(CUDA_CC) -deviceemu $(SRC) -o $(EXE) $(CUDA_FLAGS)

debug: $(SRC)
	$(CUDA_CC) -g $(SRC) -o $(EXE) $(CUDA_FLAGS)

debugenum: $(SRC)
	$(CUDA_CC) -g -deviceemu $(SRC) -o $(EXE) $(CUDA_FLAGS)

clean: $(SRC)
	$(RM) $(EXE) $(EXE).linkinfo result.txt
