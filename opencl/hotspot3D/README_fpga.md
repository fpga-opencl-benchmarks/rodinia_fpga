# Compilation

Use the build script as follows:

```
./build.sh
```

This will build a default set of kernels.

```
make ALTERA=1 board=*board_name*
```

To build the host program, run the command as follows:

```
make ALTERA=1 HOST_ONLY
```

# Execution

The benchmark has a number of arguments:

```
./hotspot3D <grid-rows> <grid-cols> <grid-layers> <number-of-iterations> <input-power-file> <input-temperature-file> <output-file> [kernel-version]
```

The arguments except for the last are the same as the original version.

Eg.:

```
./hotspot3D 512 512 8 100 ../../data/hotspot3D/power_512x8 ../../data/hotspot3D/temp_512x8 output.txt v5
```

# Kernel variations

## hotspot_kernel_original

## v0

- The original kernel with restrict and computation reordering to
match the output of the OpenMP implementation.

## v1

- Straightforward sequential implementation with a triply-nested loop.

## v2

- Based on v0
- Parameters: SSIZE
- Adds reqd_work_group_size and SIMD.

## v3

- Based on v1
- Parameters: SSIZE
- Unrolls innermost loop. Replaces address arithmetic with loads to temporary
buffers to reduce number of ports to external memory in presence of unrolling.

## v5

- Complete kernel rewrite.
- Parameters: BLOCK_X, BLOXK_Y, SSIZE. BLOCK_X needs to be divisible by SSIZE.
- Implements spatial block with rectangular blocks and shift register-based
on-chip buffers. Used loop collapse and exit condition optimization. Also
unrolls the loop to vectorize. Performance will not scale after memory
bandwidth is saturated.

## v7 (BEST)

- Based on v5.
- Parameters: BLOCK_X, BLOXK_Y, SSIZE, TIME. BSIZE needs to be divisible by SSIZE.
- Implements temporal blocking on top of spatial blocking.
Decouples memory accesses from compute. Defines compute kernel as autorun and
replicates it by the degree of temporal parallelism (TIME). Halo size is adjusted
based on TIME. Complex performance and area trade-off between all the parameters.
Refer to FPGA'18 paper for more info.