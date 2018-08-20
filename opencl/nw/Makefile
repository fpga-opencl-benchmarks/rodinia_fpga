include ../../common/make.config

MAIN = nw
KERNEL = $(MAIN)_kernel

ifeq ($(OS),Windows_NT)
	EXE = $(MAIN).exe
else
	EXE = $(MAIN)
endif

ifdef BSIZE
	CXXFLAGS += -DBSIZE=$(BSIZE)
endif

ifdef PAR
	CXXFLAGS += -DPAR=$(PAR)
endif

ifdef NO_INTERLEAVE
	CXXFLAGS += -DNO_INTERLEAVE
endif

all: $(MAIN).c 
	$(CXX) ${KERNEL_DIM} $(CXXFLAGS) $(MAIN).c $(OPENCL_INC) $(OPENCL_LIB) -o $(EXE)
	
verify: $(MAIN).c 
	$(CXX) ${KERNEL_DIM} $(CXXFLAGS) $(MAIN).c $(OPENCL_INC) $(OPENCL_LIB) -o $(EXE) -DVERIFY

trace: $(MAIN).c
	$(CXX) ${KERNEL_DIM} $(CXXFLAGS) $(MAIN).c $(OPENCL_INC) $(OPENCL_LIB) -o $(EXE) -DTRACEBACK

ifndef HOST_ONLY
all: kernel
trace: kernel
endif

kernel:

ifdef ALTERA
# default version
v ?= 9
# default block size
BSIZE ?= 128
# default SIMD size
SIMD ?= 4

# version
KERNEL_VER = $(KERNEL)_v$(v)
KERNEL_VER_AOCO = $(KERNEL_VER)$(PROFILE_SUFFIX).aoco
KERNEL_VER_AOCX = $(KERNEL_VER)$(PROFILE_SUFFIX).aocx
$(KERNEL_VER_AOCO): $(KERNEL_VER).cl
	$(RM) $(KERNEL_VER)$(PROFILE_SUFFIX)
	$(CFPGA) -c $< $(CFPGA_FLAGS) -o $@

# BSIZE
KERNEL_VER_BSIZE = $(KERNEL_VER)_BSIZE$(BSIZE)
KERNEL_VER_BSIZE_AOCO = $(KERNEL_VER_BSIZE)$(PROFILE_SUFFIX).aoco
KERNEL_VER_BSIZE_AOCX = $(KERNEL_VER_BSIZE)$(PROFILE_SUFFIX).aocx
$(KERNEL_VER_BSIZE_AOCO): DEFINE_MACROS += -DBSIZE=$(BSIZE)
$(KERNEL_VER_BSIZE_AOCO): $(KERNEL_VER).cl
	$(RM) $(KERNEL_VER_BSIZE)$(PROFILE_SUFFIX)
	$(CFPGA) -c $< $(CFPGA_FLAGS) -o $@

# SIMD
KERNEL_VER_BSIZE_SIMD = $(KERNEL_VER_BSIZE)_SIMD$(SIMD)
KERNEL_VER_BSIZE_SIMD_AOCO = $(KERNEL_VER_BSIZE_SIMD)$(PROFILE_SUFFIX).aoco
KERNEL_VER_BSIZE_SIMD_AOCX = $(KERNEL_VER_BSIZE_SIMD)$(PROFILE_SUFFIX).aocx
$(KERNEL_VER_BSIZE_SIMD_AOCO): DEFINE_MACROS += -DBSIZE=$(BSIZE) -DSIMD=$(SIMD)
$(KERNEL_VER_BSIZE_SIMD_AOCO): $(KERNEL_VER).cl
	$(RM) $(KERNEL_VER_BSIZE_SIMD)$(PROFILE_SUFFIX)
	$(CFPGA) -c $< $(CFPGA_FLAGS) -o $@


# target definitions
kernel: aocx
ifneq (,$(findstring $(v),1 3))
aocx: $(KERNEL_VER_AOCX)
aoco: $(KERNEL_VER_AOCO)
else ifneq (,$(findstring $(v),0 7 9 11 13 15))
aocx: $(KERNEL_VER_BSIZE_AOCX)
aoco: $(KERNEL_VER_BSIZE_AOCO)
else ifneq (,$(findstring $(v),2 4))
aocx: $(KERNEL_VER_BSIZE_SIMD_AOCX)
aoco: $(KERNEL_VER_BSIZE_SIMD_AOCO)
endif

endif # ifdef ALTERA

clean:
	$(RM) $(EXE) *.aoco
