include ../../../common/make.config

KERNEL_NAMES = kernel_gpu_opencl_altera_v0 \
kernel_gpu_opencl_altera_v1 \
kernel_gpu_opencl_altera_v2 \
kernel_gpu_opencl_altera_v3 \
kernel_gpu_opencl_altera_v4

KERNEL_ALTERA_NAMES = $(addsuffix .aocx, $(KERNEL_NAMES))

# Set equal to RD_WG_SIZE_0_0, RD_WG_SIZE_0, or RD_WG_SIZE. See common.h
CFPGA_FLAGS += $(KERNEL_DIM)

AOCX: $(KERNEL_ALTERA_NAMES)

kernel_gpu_opencl_altera_v0.aoco: kernel_gpu_opencl_altera_v0.cl \
	kernel_gpu_opencl_v0.cl	kernel_gpu_opencl_2_v0.cl

kernel_gpu_opencl_altera_v1.aoco: kernel_gpu_opencl_altera_v1.cl \
	kernel_gpu_opencl_v1.cl	kernel_gpu_opencl_2_v1.cl

kernel_gpu_opencl_altera_v2.aoco: kernel_gpu_opencl_altera_v2.cl \
	kernel_gpu_opencl_v2.cl	kernel_gpu_opencl_2_v2.cl

kernel_gpu_opencl_altera_v3.aoco: kernel_gpu_opencl_altera_v3.cl \
	kernel_gpu_opencl_v3.cl	kernel_gpu_opencl_2_v3.cl

kernel_gpu_opencl_altera_v4.aoco: kernel_gpu_opencl_altera_v4.cl \
	kernel_gpu_opencl_v4.cl	kernel_gpu_opencl_2_v4.cl

clean:
	$(RM) *.aoco $(KERNEL_NAMES)
