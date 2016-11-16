include ../../common/make.config

all : nn hurricane_gen

clean :
	$(RM) *.o nn hurricane_gen

nn : nn_cuda.o
	$(CUDA_CC) $(CUDA_FLAGS) $^ -o $@ $(NVML_LIB)

hurricane_gen : hurricane_gen.o

#data :
#	mkdir data
#	./gen_dataset.sh
