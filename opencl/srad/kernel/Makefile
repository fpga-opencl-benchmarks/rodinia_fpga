include ../../../common/make.config

NAME = srad

KERNEL_VERSIONS = 0 1 2 3 5
KERNEL_NAMES = $(foreach var,$(KERNEL_VERSIONS),$(NAME)_kernel_v$(var))
KERNEL_ALTERA_NAMES = $(addsuffix .aocx, $(KERNEL_NAMES))

ifdef ALTERA
	AOT_PROGRAMS =  $(KERNEL_ALTERA_NAMES)
else
	AOT_PROGRAMS =
endif

AOCX: $(AOT_PROGRAMS)

# Set equal to RD_WG_SIZE_0_0, RD_WG_SIZE_0, or RD_WG_SIZE. See common.h
CFPGA_FLAGS += $(KERNEL_DIM) -I..

clean:
	$(RM) *.aoco $(KERNEL_NAMES)
