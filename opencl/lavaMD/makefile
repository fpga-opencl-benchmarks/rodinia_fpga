include ../../common/make.config

NAME = lavaMD
ifeq ($(OS),Windows_NT)
	EXE = $(NAME).exe
else
	EXE = $(NAME)
endif
SRC = main.c ./kernel/kernel_gpu_opencl_wrapper.c ./util/timer/timer.c ./util/opencl/opencl.c ./util/num/num.c

ifdef OUTPUT
	EXTRA_FLAGS += -DOUTPUT
endif

VERSIONS = 0 1 3
KERNEL_NAMES = $(foreach var,$(VERSIONS),$(NAME)_kernel_v$(var))
KERNEL_ALTERA_NAMES = $(addsuffix .aocx, $(KERNEL_NAMES))

ifdef ALTERA
	ifdef HOST_ONLY
		AOT_PROGRAMS = 
	else
		AOT_PROGRAMS = $(KERNEL_ALTERA_NAMES)
	endif
else
	AOT_PROGRAMS =
endif

all: $(SRC) $(AOT_PROGRAMS)
	$(CC) $(CFLAGS) $(SRC) -o $(EXE) $(OPENCL_INC) $(OPENCL_LIB) $(EXTRA_FLAGS)

clean:
	$(RM) $(EXE) *.aoco *.aocx $(KERNEL_NAMES)
