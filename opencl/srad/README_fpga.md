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

## v1

Straightforward single work-item kernel created by wrapping kernels
in for loops and adding strict.

## v2

Basic NDRange optimizations with SIMD and reqd_work_group_size.

## v3

Uses Altera's shit register method for efficient reduction. Also uses
some unrolling

## v5 (Best)

Parameter: BSIZE, SSIZE

Kernels are pretty much completely rewritten. All compute kernels are
combined into one single kernel.

For loops of original prepare and reduction kernels are combined into
one loop, resulting in much less global memory usage and access.

Global memory-based fetching of neighbor indexes is replaced with very
simple math, which removes 4 more buffers (d_iX).

SRAD and SRAD2 loops are combined and more buffers are removed (d_dX).

Computation direction is changed from top right, downwards, to bottom
left, upwards, to easily address dependency to the computation of right
and bottom indexes.

Computation is partitioned and a two shift registers are employed,
one for the SRAD1 computation (I_SR) to be used as a local cache to
prevent redundant off-chip memory accesses, and the other (c_loc_SR)
to store output of past iterations for use in next iterations.

A small c_boundary buffer is used to hold the row on the block boundaries
and address bottom dependency for upper blocks.

A new I_out buffer is added to store output, instead of overwriting the
input.

A single while loop is used to process each block, instead of nested loops,
to make sure the compiler would correctly infer the shift registers.

The loop on blocks is kept outside of the while loop over the block, to make
sure blocks will be processed sequentially due to dependency on the c_boundary
variable. False dependency on this variable inside the while loop is removed
by using #pragma ivdep.

The "I" input has been falsely marked as volatile to disable Altera's cache.
The cache is completely useless here and just wastes space since we are
caching everything manually using the shift registers.

Finally, unsigned variables are used whenever possible and scope of variables
is minimized as much as possible.