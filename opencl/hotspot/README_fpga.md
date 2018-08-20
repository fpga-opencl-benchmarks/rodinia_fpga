# Compilation

Use the build script as follows:

```
./build.sh
```

This will build a default set of kernels. Alternatively, the make command can be used to build a specific kernel variation:

```
make ALTERA=1 board=*board_name* v=9 kernels
```

To build the host program, run the command as follows:

```
make ALTERA=1 HOST_ONLY
```

# Execution

The benchmark has a number of arguments:

```
./hotspot <matrix-size> <temporal-blocking-degree> <number-of-iterations> <input-temperature-file> <input-power-file> <output-file> <block-size> [kernel-version]
```

The arguments except for the last two are the same as the original version. The block-size argument is the BSIZE parameter in the kernels.

Eg.:

```
./hotspot 1024 1 1000 ../../data/hotspot/temp_1024 ../../data/hotspot/power_1024 output.txt v5
```

# Kernel variations

## v0

- Original NDRange kernel with restrict.
- Parameters: BSIZE

## v1

- Straightforward single work-item implementation with a doubly-nested
loop and restrict.

## v2

- Based on v0
- Parameters: BSIZE, SSIZE
- Adds reqd_work_group_size and SIMD and moves calculation of constants out of the
kernel and to the host.

## v3

- Based on v1
- Parameters: SSIZE
- Unrolls innermost loop. Replaces address arithmetic with loads to temporary
buffers to reduce number of ports to external memory in presence of unrolling.
Moves calculation of constants out of the kernel and to the host.

## v4

- Based on v2
- Parameters: BLOCK_X, BLOCK_Y, SSIZE, UNROLL
- temp_t and power_on_cuda buffers are removed and replaced with private registers
since each is written and read only once by the same thread. All accesses to the
remaining local buffer inside the iteration loop are replaced with writes to and
reads from registers to allow correct coalescing on accesses to the local buffer,
reducing number of reads from the buffer from 35 to 5. Furthermore, the write to
the remaining local buffer from external memory and the updated output of the previous
iteration are combined into one access using a conditional statement over an extra
register. This halves Block RAM replication factor due to reducing number of write
ports from two to one. This also adds the possibility of removing on of the barriers;
however, doing so reduces the number of simultaneous work-groups the compiler allows,
which results in performance reduction. Hence, we keep the false barrier to achieve
better performance, at the cost of slightly higher Block RAM usage. Finally, possibility
of using non-square blocks and unrolling the iteration loop was added. With all these
optimizations, up to an unroll factor of three can be used before the number of write
ports reaches four, forcing port sharing.

## v5

- Complete kernel rewrite.
- Parameters: BLOCK_X, BLOXK_Y, SSIZE. BLOCK_X needs to be divisible by SSIZE.
- Implements spatial block with rectangular blocks and shift register-based
on-chip buffers. Used loop collapse and exit condition optimization. Also
unrolls the loop to vectorize. Performance will not scale after memory
bandwidth is saturated.

## v7 (BEST)

- Based on v5.
- Parameters: BSIZE, SSIZE, TIME. BSIZE needs to be divisible by SSIZE.
- Implements temporal blocking on top of spatial blocking.
Decouples memory accesses from compute. Defines compute kernel as autorun and
replicates it by the degree of temporal parallelism (TIME). Halo size is adjusted
based on TIME. Complex performance and area trade-off between all the parameters.
Refer to FPGA'18 paper for more info.
