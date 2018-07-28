include ../../common/make.config

NAME = bfs
ifeq ($(OS),Windows_NT)
	EXE = $(NAME).exe
else
	EXE = $(NAME)
endif
SRC = bfs.cpp

VERSIONS = 0 1 2 3 5
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

# the following is defined to avoid compilation warning for reading size_t values
# size_t is considered as unsigned in ALTERA's compiler while other compilers seem to recognize it as unsigned long
ifdef ARM
	EXTRAFLAGS = -D ARM
else
	ifneq ($(OS),Windows_NT)
		ifneq ($(OSX_TARGET), 1)
			EXTRAFLAGS = -fopenmp
		endif
	endif
endif

release: $(SRC) $(AOT_PROGRAMS)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(EXE) $(OPENCL_INC) $(OPENCL_LIB) $(EXTRAFLAGS)

ptx: $(SRC) $(AOT_PROGRAMS)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(EXE) $(OPENCL_INC) $(OPENCL_LIB) $(EXTRAFLAGS) -D PTX_MSG

res: $(SRC) $(AOT_PROGRAMS)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(EXE) $(OPENCL_INC) $(OPENCL_LIB) $(EXTRAFLAGS) -D RES_MSG

debug: $(SRC) $(AOT_PROGRAMS)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(EXE) $(OPENCL_INC) $(OPENCL_LIB) $(EXTRAFLAGS) 

run:
	./$(EXE)

clean: $(SRC)
	$(RM) $(EXE) $(EXE).linkinfo result* *.aoco *.aocx $(KERNEL_NAMES)
