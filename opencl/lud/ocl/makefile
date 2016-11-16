include ../../../common/make.config

NAME = lud

KERNEL_VERSIONS = 0 1 2 3 4 5 6
KERNEL_NAMES = $(foreach var,$(KERNEL_VERSIONS),$(NAME)_kernel_v$(var))
KERNEL_ALTERA_NAMES = $(addsuffix .aocx, $(KERNEL_NAMES))

ifdef ALTERA
	AOT_PROGRAMS =  $(KERNEL_ALTERA_NAMES)
else
	AOT_PROGRAMS =
endif

AOCX: $(AOT_PROGRAMS)

clean:
	$(RM) ../$(EXE) *.aoco *.aocx lud_kernel
	cd ../; $(RM) $(OBJECTS)
