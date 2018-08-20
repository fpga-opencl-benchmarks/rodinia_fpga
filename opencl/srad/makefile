include ../../common/make.config

CFLAGS += $(OPENCL_INC) $(KERNEL_DIM) -fopenmp
SOURCES=main.c kernel/kernel_gpu_opencl_wrapper.c util/graphics/graphics.c util/graphics/resize.c util/opencl/opencl.c

ifdef RED_UNROLL
	CFLAGS += -DRED_UNROLL=$(RED_UNROLL)
	AOC_FLAGS = -DRED_UNROLL=$(RED_UNROLL)
endif

ifdef SSIZE
	CFLAGS += -DSSIZE=$(SSIZE)
	AOC_FLAGS = -DSSIZE=$(SSIZE)
endif

ifdef BSIZE
	CFLAGS += -DBSIZE=$(BSIZE)
	AOC_FLAGS = -DBSIZE=$(BSIZE)
endif

ifdef NO_INTERLEAVE
	CFLAGS += -DNO_INTERLEAVE
	AOC_FLAGS += --no-interleaving default
endif

ifdef ALTERA
	ifdef HOST_ONLY
		AOT_PROGRAMS = 
	else
		AOT_PROGRAMS = AOCX
	endif
else
	AOT_PROGRAMS =
endif

ifeq ($(OS),Windows_NT)
	EXE=srad.exe
	LDFLAGS+=/out:$@
	CFLAGS+=/Fo$@
	# flip the path separator for Windows (/ -> \)
	SOURCESTOBUILD=$(subst /,\,$(SOURCES))
	OBJECTS=$(SOURCESTOBUILD:.c=.obj)
else
	EXE=srad
	LDFLAGS+=-lm -o $@
	CFLAGS+=-o $@
	SOURCESTOBUILD=$(SOURCES)
	OBJECTS=$(SOURCESTOBUILD:.c=.o)
endif

all: $(EXE) $(AOT_PROGRAMS)

$(EXE): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) $(OPENCL_LIB) $(LDFLAGS)

%.o %.obj: %.c
	$(CC) $(CFLAGS) -c $< 

clean:
	$(RM) $(OBJECTS) $(EXE)
	$(MAKE) -C kernel clean

ifdef ALTERA
AOCX:
	$(MAKE) -C kernel $(AOC_FLAGS)
else
AOCX:
endif
