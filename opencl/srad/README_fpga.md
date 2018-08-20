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
aoc [kernel_name] -g -v --report --board [board_name] -I../../ -DUSE_RESTRICT -DBSIZE=[BSZIE] -DSSIZE=[SSIZE]
```


# Execution

Default run:

```
./run v[version_number]
```

Custom run:

```
./srad <iteration_number> <lambda> <height> <width> v[version_number]
```


# Kernel variations

See the github Wiki page for more general information.

## v0

Original Rodinia kernel with restrict and removal of unused kernel
parameters. The "compress" and "extract" kernels are also moved to the
host since they take a very small portion of the total run time and are
not timed anyway.

## v1

Straightforward single work-item implementation created by wrapping
the NDRange kernels in for loops and adding restrict. Also adds one
instance of ivdep to the srad2 kernel to avoid false load/store
dependency.

## v2

Compared to v0, adds reqd_work_group_size for all kernels and SIMD for
all kernels except the reduction kernel; the latter has thread-id-dependent
branching and SIMD doesn't work for it. Used unrolling in that kernel instead.
One simple loop in this kernel was also fully unrolled.

## v3

Compared to v1, uses shift register for efficient reduction and unrolls
all the innermost loops which also requires another ivdep in the srad2
kernel.

## v5 (Best)

Parameter: BSIZE, SSIZE, RED_UNROLL

Kernels are pretty much completely rewritten. All compute kernels are
combined into one single kernel. For loops of original prepare and
reduction kernels are combined into one loop, resulting in much less
global memory usage and access. Global memory-based fetching of neighbor
indexes is replaced with very simple math, which removes 4 more buffers
(d_iX). SRAD and SRAD2 loops are combined and more buffers are removed
(d_dX). Computation direction is changed from top right, downwards, to
bottom left, upwards, since this is the only way the SRAD2 computation
could start right after the SRAD1 computation. 1D blocking and overlapping
is used with shift registers. Since two kernels are involved, a halo size
of two is required. loop collapse and exit condition optimization are also
employed. Arria 10 also uses single-cycle floating-point accumulation.
Due to some issue with Quartus v16.1.2, multiplication of constants with
floating-point variables resulted in abnormally-low operating frequency.
To work-around this problem, all such multiplications were replaced with
division which also reduced DSP usage with no loss of computation accuracy.