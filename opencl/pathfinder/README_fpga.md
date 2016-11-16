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

## v1

Single work-item version based on OpenMP version of the benchmark
with strict. Also moved for loop on rows from host code to device
code and added clEnqueueWriteBuffer to actually write the buffers
from host to device instead of using host pointers.

## v2

Restrict, 16 SIMD, 2 compute units, automatic local memory sharing
between compute units performed by aoc

## v3

v1 with some loop unrolling.

## v4

Same as v2 but with 4 compute units (maximum number of compute units
with sub-80% logic usage on Startix V). Local memory sharing is
automatically disabled by aoc due to block ram overutilization.

## v5

Each iteration computes a sub section of a row, and proceeds to the
net row at the diagonal position. The length of the section is
determined by BLOCK_SIZE. Once the iterations reach the end of
rows, the next sub section of the first row is visited. No global
memory access is done except for the first row.

## v7

Extends v5 with decomposition of the section computation. Parameter
XS is used to specify the decomposition factor. For example, when BLOCK_SIZE
is 64 and XS is 4, a continuous section of 128 items is computed
in the four consecutive iterations. This is expected to improve
memory access performance since it reduces strided accesses.

## v9

Extends v7 so that the problem size is not a multiple of the section
length.
