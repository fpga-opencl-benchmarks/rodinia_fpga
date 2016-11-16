include ../../common/make.config

MAIN = euler3d pre_euler3d
KERNEL_PRE = pre_euler3d_kernel
KERNEL_NON_PRE = euler3d_kernel
KERNEL = $(KERNEL_NON_PRE)

ifeq ($(OS),Windows_NT)
	EXE = $(addsuffix .exe,$(MAIN))
else
	EXE = $(MAIN)
endif

all: $(EXE)

CXXFLAGS += $(KERNEL_DIM) $(OPENCL_INC)
LDFLAGS += $(OPENCL_LIB)

euler3d: euler3d.o host_common.o CLHelper2.o
	$(CXX) $^ -o $@ $(LDFLAGS)

euler3d.o: CLHelper2.h common.h host_common.h

pre_euler3d: pre_euler3d.o host_common.o CLHelper2.o
	$(CXX) $^ -o $@ $(LDFLAGS)

pre_euler3d.o: CLHelper2.h common.h host_common.h

host_common.o: host_common.h CLHelper2.h

ifndef HOST_ONLY
all: kernel
trace: kernel
endif

kernel:

clean:
	$(RM) $(EXE)  *.o

ifdef ALTERA
# default version
v ?= 0
# default block size
BSIZE ?= 16
# num_compute_units
CUSIZE ?= 1

# version
KERNEL_VER = $(addsuffix _v$(v),$(KERNEL))
KERNEL_VER_AOCO = $(addsuffix .aoco,$(KERNEL_VER))
KERNEL_VER_AOCX = $(addsuffix .aocx,$(KERNEL_VER))

# BSIZE
KERNEL_VER_BSIZE = $(addsuffix _BSIZE$(BSIZE),$(KERNEL_VER))
KERNEL_VER_BSIZE_AOCO = $(addsuffix .aoco,$(KERNEL_VER_BSIZE))
KERNEL_VER_BSIZE_AOCX = $(addsuffix .aocx,$(KERNEL_VER_BSIZE))
$(KERNEL_VER_BSIZE_AOCO): DEFINE_MACROS += -DBSIZE=$(BSIZE)
euler3d_kernel_v$(v)_BSIZE$(BSIZE).aoco: euler3d_kernel_v$(v).cl
	$(RM) euler3d_kernel_v$(v)_BSIZE$(BSIZE)
	$(CFPGA) -c $< $(CFPGA_FLAGS) -o $@
pre_euler3d_kernel_v$(v)_BSIZE$(BSIZE).aoco: pre_euler3d_kernel_v$(v).cl
	$(RM) pre_euler3d_kernel_v$(v)_BSIZE$(BSIZE)
	$(CFPGA) -c $< $(CFPGA_FLAGS) -o $@

# CUSIZE
KERNEL_VER_BSIZE_CUSIZE = $(addsuffix _CUSIZE$(CUSIZE),$(KERNEL_VER_BSIZE))
KERNEL_VER_BSIZE_CUSIZE_AOCO = $(addsuffix .aoco,$(KERNEL_VER_BSIZE_CUSIZE))
KERNEL_VER_BSIZE_CUSIZE_AOCX = $(addsuffix .aocx,$(KERNEL_VER_BSIZE_CUSIZE))
$(KERNEL_VER_BSIZE_CUSIZE_AOCO): DEFINE_MACROS += -DBSIZE=$(BSIZE) -DCUSIZE=$(CUSIZE)
euler3d_kernel_v$(v)_BSIZE$(BSIZE)_CUSIZE$(CUSIZE).aoco: euler3d_kernel_v$(v).cl
	$(RM) euler3d_kernel_v$(v)_BSIZE$(BSIZE)_CUSIZE$(CUSIZE)
	$(CFPGA) -c $< $(CFPGA_FLAGS) -o $@
pre_euler3d_kernel_v$(v)_BSIZE$(BSIZE)_CUSIZE$(CUSIZE).aoco: pre_euler3d_kernel_v$(v).cl
	$(RM) pre_euler3d_kernel_v$(v)_BSIZE$(BSIZE)_CUSIZE$(CUSIZE)
	$(CFPGA) -c $< $(CFPGA_FLAGS) -o $@


# target definitions
kernel: aocx
ifneq (,$(findstring $(v),0 1 3 5 7))
aocx: $(KERNEL_VER_AOCX)
aoco: $(KERNEL_VER_AOCO)
else ifneq (,$(findstring $(v),2 4))
aocx: $(KERNEL_VER_BSIZE_AOCX)
aoco: $(KERNEL_VER_BSIZE_AOCO)
else ifneq (,$(findstring $(v),6 8))
aocx: $(KERNEL_VER_BSIZE_CUSIZE_AOCX)
aoco: $(KERNEL_VER_BSIZE_CUSIZE_AOCO)
endif

clean: clean_aocl

clean_aocl:
	rm -f *.aoco *.aocx
	find .  -maxdepth 1 -name "$(KERNEL_PRE)_v[0-9]*" "$(KERNEL_NON_PRE)_v[0-9]*" -type d | xargs rm -rf

endif # ifdef ALTERA
