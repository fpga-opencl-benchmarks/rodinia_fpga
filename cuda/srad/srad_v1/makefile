include ../../../common/make.config

# Example
# target: dependencies
	# command 1
	# command 2
          # .
          # .
          # .
	# command n

# link objects(binaries) together
srad:		main.o
	$(CUDA_CC) $(CUDA_FLAGS) main.o -lm -o $@  $(NVML_INC) $(NVML_LIB)

# compile main function file into object (binary)
main.o: 	main.cu \
				define.c \
				graphics.c \
				extract_kernel.cu \
				prepare_kernel.cu \
				reduce_kernel.cu \
				srad_kernel.cu \
				srad2_kernel.cu \
				compress_kernel.cu

# delete all object files
clean:
	$(RM) *.o srad
