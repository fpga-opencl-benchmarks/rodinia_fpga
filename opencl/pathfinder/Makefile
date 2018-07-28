include ../../common/make.config

NAME = pathfinder
ifeq ($(OS),Windows_NT)
	EXE = $(NAME).exe
else
	EXE = $(NAME)
endif
SRC = main.cpp OpenCL.cpp

VERSIONS = 0 1 2 3 4 5 7 9
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
	$(CXX) $(CXXFLAGS) $(SRC) -o $(EXE) $(OPENCL_INC) $(OPENCL_LIB) $(EXTRA_FLAGS) $(LDFLAGS)

clean:
	$(RM) $(EXE) *.aoco *.aocx $(KERNEL_NAMES)
