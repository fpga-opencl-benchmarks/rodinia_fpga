# Compilation

Default:

```
make Altera=1 BOARD=[board_name]
```

Host only:
```
make ALTERA=1 HOST_ONLY=1
```

Custom kernel:

```
aoc [kernel_name] -g -v --report --board [board_name] -I../../ -D[parameter_name]=[parameter_value]
```

# Execution

Default run:

```
./run v[version_number]
```

Custom run:

```
./nw [input_size] [penalty] v[version_number]
```

# Kernel variations

See the github Wiki page for more general information.

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

A straightforward single work-item kernel plus restrict keyword.
Iterates through the 2-dimensional space from top left to bottom right.
Due to load/store dependency on the input_itemsets variable, the outer
loop is run sequentially and the inner loop is run with an II of 328
(latency of two memory accesses).

## v2

Extends v0 with reqd_work_group_size, SIMD and unrolling. Due to naive kernel
design and huge number of accesses to the local buffers, it is impossible to go
beyond a BSIZE of 64 or SIMD size of 2 with this kernel. Unrolling is completely
out of the question since it blows the Block RAM usage through the roof.

## v3

This version saves the output of each iteration in a backup buffer
to be used for left dependency in the next iteration. This, coupled
with #pragma ivdep on the input_itemsets variable allows full pipelining
on the inner loop. The outer loop is still run sequentially due to
dependency on the same variable which is unavoidable in this implementation.
Unrolling the inner loop results in new load/store dependencies and hence,
was avoided.

## v5 (BEST)

Parameters: BSIZE, PAR

This version uses blocking and diagonal parallelism with a block height of
BSIZE and a parallelism degree of PAR. Computation starts from top-left and
moves downwards, computing one chunk of columns at a time with a chunk width
of PAR. The chunk of columns is calculated in a diagonal fashion, with the first
diagonal starting from an out-of-bound point and ending on the top-left cell in
the grid. Then computation moves downwards, calculating one diagonal with PAR
cells at a time until the bottom-cell in the diagonal falls on the bottom-left
cell in the block. For the next diagonal, the cells that would fall inside the
next block instead wrap around the block and start computing cells from the next
chunk of columns in the current block. When the first cell in the diagonal falls
on the block boundary, the computation of the first chunk of columns is finished
and every cell computed after that will be from the second chunk of columns. When
all the columns in a block are computed, computation moves to the next block and
computation repeats in the same fashion. To correctly handle the dependencies,
each newly-computed cell is buffered on-chip using shift registers for two iterations
to resolve the dependency on the top cell in the next iteration (diagonal) and
top-left in the iteration after that. Furthermore, the cells on the right-most column
in the current chunk of columns are also buffered in a large shift register with the
size of BSIZE so that they can be reused in the next chunk of columns. Finally, the
blocks are also overlapped by one row to provide the possibility to re-read the
cells on the boundary computed in the previous block and handle the dependencies for
the first new row in the new block.
Since diagonal computation prevents memory access coalescing, we manually insert a
set of shift registers between the memory accesses (both read and write) and
computation to delay memory accesses and convert diagonal accesses to consecutive
coalesceable ones. For reading, the shift register for the first column in the chunk
has a size of par and as we move towards the last column in the chunk, the shift
registers get smaller by one cell until the last column where the shift register
will turn into a single register. For writes, the set of shift registers instead
starts from a single register and ends with a shift register of size par. Furthermore,
writes start PAR iterations after reads and hence, the input width is padded by PAR
cells to allow the cells in the rightmost chunk of columns to be written back to
external memory correctly. Finally, to improve alignment of external memory accesses,
we pass the first column of the input that is read-only using a separate buffer so
that reads from and writes to the main buffer start from the address 0 instead of 1.
