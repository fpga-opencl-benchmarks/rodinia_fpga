include ../../common/make.config
INC = $(OPENCL_INC) -Icommon
LIB = $(OPENCL_LIB)

CFLAGS += $(INC)
CXXLAGS += $(INC)
CSOURCES=common/common.c
CPPSOURCES=ocl/lud.cpp

ifeq ($(OS),Windows_NT)
	EXE=lud.exe
	LDFLAGS+=/out:$@
	CFLAGS+=/Fo$@
	# flip the path seperator for Windows (/ -> \)
	CSOURCESTOBUILD=$(subst /,\,$(CSOURCES))
	CPPSOURCESTOBUILD=$(subst /,\,$(CPPSOURCES))
	OBJECTS=$(CSOURCESTOBUILD:.c=.obj)
	OBJECTS+=$(CPPSOURCESTOBUILD:.cpp=.obj)
else
	EXE=lud
	LDFLAGS+=-lm -o $@
	CFLAGS+=-o $@
	CSOURCESTOBUILD=$(CSOURCES)
	CPPSOURCESTOBUILD=$(CPPSOURCES)
	OBJECTS=$(CSOURCESTOBUILD:.c=.o)
	OBJECTS+=$(CPPSOURCESTOBUILD:.cpp=.o)
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

all: $(EXE) $(AOT_PROGRAMS)

profile: CFLAGS += -DPROFILE
profile: $(EXE) $(AOT_PROGRAMS)

$(EXE): $(OBJECTS)
	$(CXX) $(OBJECTS) $(KERNEL_DIM) $(LDFLAGS) $(LIB) $(CFLAGS)

AOCX:
	cd ocl; make

clean:
	cd ocl; make clean

%.o %.obj: %.c
	$(CC) $(CFLAGS) -c $<

%.o %.obj: %.cpp
	$(CXX) $(CFLAGS) -c $<

export OBJECTS EXE CFLAGS
