# Compilation

To build the host code:

```
make ALTERA=1 HOST_ONLY
```

To build a kernel:

```
make ALTERA=1 pathfinder_kernel_vN.aocx
```

where `vN` corresponds to the version number.

# Execution

To run this benchmark, run the pathfinder command as follows:

```
./pathfinder <row-length> <column-length> <pyramid_height>
```

Parameter `<row-lenght>` specifies the length of each row, `<column-length>` the number of rows, and `<pyramid_height>` the number of rows that are processed in a batch by a single kernel invocation.

# Kernel variations

## v0

Same as original Rodinia kernel with restrict and removal of unused
kernel arguments. Local buffers definition is also moved from kernel
argument space to inside of the kernel. The former case, apart from being
a very bad design practice, also breaks area estimation in earlier versions
of Intel's OpenCL Compiler since the area usage by these buffers is not
taken into account. Finally, when such buffers are defined as a kernel
argument, their size must be a power of two or else compilation will fail,
while such limitation does not exist for local buffers defined inside of
the kernel.

## v1

Single work-item version based on OpenMP version of the benchmark
with strict. Similar to the NDRange kernels, the loop on rows is
kept in the host code and the source and destinations buffers are
swapped row by row.

## v2

On top of v0, adds reqd_work_group_size attribute alongside with SIMD,
compute unit replication and unrolling.

## v3

Compared to v1, moves external memory accesses out of branches to allow
correct access coalescing under the presence of loop unrolling and then
unrolls in the innermost loop. Also adds __attribute__((max_global_work_dim(0))).

## v4

Based on v2. Replaces result buffer with private register. Corrects access
coalescing to prev buffer by moving access to the remaining local buffer
outside of conditional statements and replacing them with temporary registers.
Also merges the write from external memory and the one from the output of the
previous iteration to the local buffer into one right using conditional statement
on a temporary register. Reduces number of reads and writes to 3 and 1 regardless
of SIMD size. Also keeps an unnecessary barrier to increase number of simultaneous
work-groups. Overall, exact same strategy as Hotspot 2D v4.

## v5

This is basically a single work-item implementation of the NDRange code. It uses
2D blocking, one dimension of which can be changed at run-time similar to the NDRange
code. Since the computation exhibits a triangular/cone-like dependency pattern,
blocks are overlapped by twice more columns than the height of the block (i.e.
number of combined rows chosen at run-time). This results in a huge amount of redundancy,
but is amortized by the very large block size. In contrast to the NDRange implementation,
the on-chip buffer in the single work-item version is shift register which allows
implementing much larger blocks with even lower Block RAM usage. Furthermore, this
implementation achieves better operating frequency compared to the NDRange implementation;
however, it performs around 30% slower, very likely due to memory access alignment problems
which do not seem to affect NDRange kernels as much.