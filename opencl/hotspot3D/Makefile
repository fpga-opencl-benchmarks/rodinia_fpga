include ../../common/make.config

NAME = hotspot3D
ifeq ($(OS),Windows_NT)
	EXE = $(NAME).exe
else
	EXE = $(NAME)
endif
SRC = hotspot3D.c CL_helper.c

ifdef TIME
	CFLAGS += -DTIME=$(TIME)
endif

ifdef SSIZE
	CFLAGS += -DSSIZE=$(SSIZE)
endif

ifdef BSIZE
	CFLAGS += -DBLOCK_X=$(BSIZE)
	CFLAGS += -DBLOCK_Y=$(BSIZE)
endif

ifdef BLOCK_X
	CFLAGS += -DBLOCK_X=$(BLOCK_X)
endif

ifdef BLOCK_Y
	CFLAGS += -DBLOCK_Y=$(BLOCK_Y)
endif

ifdef PAD
	CFLAGS += -DPAD=$(PAD)
endif

VERSIONS = 0 1 3 5 7
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

ifdef VERIFY
	CFLAGS += -DVERIFY
endif

OUTPUT = *.txt

EXTRA_FLAGS += -lm

all: $(SRC) $(AOT_PROGRAMS)
	$(CC) $(CFLAGS) $(SRC) -o $(EXE) $(OPENCL_INC) $(OPENCL_LIB) $(EXTRA_FLAGS)

clean:
	$(RM) $(EXE) *.aoco $(KERNEL_NAMES) $(OUTPUT)
