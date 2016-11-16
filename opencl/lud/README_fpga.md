# Compilation

Default:

```
make Altera=1 BOARD=[board_name]
```

Host only:
```
make ALTERA=1 HOSTONLY=1
```

Custom kernel:

```
aoc [kernel_name] -g -v --report --board [board_name] -I../../ -DUSE_RESTRICT -D[parameter_name]=[parameter_value]
```


# Execution

Default run:

```
./run v[version_number]
```

Custom run:

```
./lud -i [input_file] v[version_number]

or

./lud -s [input_size] v[version_number]
```


# Kernel variations

See the github Wiki page for more general information. 

## v1

Straightforward single work-item kernel based on the "base" implementation
provided by the benchmark author. Also uses restrict.

## v2

Adds reqd_work_group_size and unrolls loop in internal kernel.

## v3

Uses Altera's shit register method for efficient reduction.

## v4 (Best)

Parameters: BLOCK_SZIE, DIA_UNROLL, PERI_UNROLL, PERI_SIMD, PERI_COMPUTE, INTER_SIMD, INTER_COMPUTE

Compared to v2, increases block size, uses unrolling for the diameter and
perimeter kernels and multiple compute units for the perimeter and internal
kernels. Also uses a temp sum variable to reduce accesses to the shadow buffer
in the diameter kernel and the peri_col and peri_row buffers in the perimeter
kernel which reduces number of accesses and replication factor for these buffers.
Furthermore, disables private cache for the diameter and perimeter kernels by
using the volatile keyword to save some BLOCK RAMs without any negative performance
impact. The existence of this cache has a small positive effect on the internal
kernel, though, if compute units are used. With SIMD for internal, the cache is
not necessary anymore. Also moves local buffers inside the kernels so that AOC
would print implementation details of the local buffers in the area report. Finally,
stores data in the peri_row buffer in the perimeter kernel column-wise instead
of row-wise which allows accesses to this buffer in the first unrolled loop to
be coalesced, reducing the total number of accesses and the replication factor
for this buffer. Similarly, two dia buffers are used instead of one, with one
being loaded row-wise (as the original buffer was), and the other one being loaded
column-wise, for more efficient memory access with unrolling. Using two buffers
like this actually reduces Block RAM usage compared to one buffer with inefficient
access. Same technique is also applied to the shadow buffer in the diameter
kernel and Altera's memory attributes are used to prevent the compiler from
over-replicating the shadow buffer to enable stall-free access since this kernel
is least time-consuming part of the benchmark. Attributes are manually tuned by
compiling and timing tens of different kernels with different values for block size,
unroll, compute units and SIMD.

## v5

Single work-item implementation using the same blocking technique that is
used in the NDRange version. Combines all kernels. Localizes memory accesses
by manually loading data from global memory to local memory. Memory load
loops are unrolled so that the compiler will coalesce the accesses. Inners loops
whose header depends on the iteration of the outer loop are modified in a way
that moves this dependency from the inner loop header to the inner loop body
and uses if/else, which can be pipelined, to handle the change. This results
in slightly more efficient pipelining at the cost of higher area usage.
Fully unrolls the two innermost loops in internal computation. Single work-item
does not work well with this algorithm and even this v5 performs horribly.

## v6

Parameters: BLOCK_SZIE, DIA_UNROLL, PERI_UNROLL, PERI_SIMD, PERI_COMPUTE, INTER_SIMD, INTER_COMPUTE

This version changes the way the buffers are loaded from off-chip memory to un-chip
memory in the perimeter kernel to remove thread-id-dependent branching and merge the
loads into one single loop. This version merges write back to memory with computation
for peri_row. This is possible here since each thread computing peri_row will eventually
write back the same memory location as it computed. This is a by-product of changing
peri_row load direction that was first used in v4, but is not possible for peri_col since
it is still using its original direction and changing its direction will result in
inefficient usage of Block RAMs. Also this version starts peri_row computation from i=0
instead of i=1, which, despite resulting in redundant computation, magically enables the
compiler to properly coalesce accesses to local memory in the j loop and reduce number of
accesses and Block RAM usage. All in all, Block RAM usage is reduced by 3-4% and performance
is improved in the perimeter kernel but in the end, this version fails to achieve better
performance than v4 due to lower operating frequency and the improvement in the perimeter
run time not being substantial enough to make up for the lower operating frequency.

## v8

Parameters: BLOCK_SZIE, DIA_UNROLL, PERI_UNROLL, PERI_SIMD, PERI_COMPUTE, INTER_SIMD, INTER_COMPUTE

This kernel breaks the perimeter kernel into two kernels, one performing the peri_row
computation and the other performing peri_col, but instead of running them sequentially,
the host code is modified to use two separate queues, with each of these two kernels
running in a separate queue, and uses OpenCL events to synchronize the kernels. This way,
due to each kernel having its own pipeline on the FPGA, the two kernels run in parallel
with a very small overhead. In the perimeter_row kernel, this version removes the barrier
and merges write back to memory with computation, similar to v6. This version also
changes memory addressing direction in the perimeter_col kernel and manually unrolls
memory write back to maximize usage of the available memory bandwidth. This kernel fails
to achieve better performance compared to v4 for the same reason as v6.