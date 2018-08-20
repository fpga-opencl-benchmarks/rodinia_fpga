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
./lud -i [input_file] v[version_number]

or

./lud -s [input_size] v[version_number]
```


# Kernel variations

See the github Wiki page for more general information.

## v0
Original Rodinia kernel but definition of local buffers is moved from kernel
arguments to inside of the kernel and automatic loop unrolling by the compiler
on two loops has been disabled to prevent explosion of ports to external memory.

## v1

Direct port of the OpenMP version with restrict. Also adds ivdep to some loops
to avoid false dependencies detected by the compiler, and swaps the two innermost
loops in the internal kernel to avoid a functional bug in Altera's compiler.
After this change, the temporary sum variable is changed from an array to a single
variable, and the memory write back loop is also merged into the compute loop.

## v2

On top of v0, add reqd_work_group_size, SIMD and CU, fully unrolls the loop in internal
kernel and adds possibility of partial unrolling to other kernels. SIMD is not usable
on diagonal and perimeter kernels due thread-id-dependent branches in the kernels, and
compute unit replication is also useless on the diameter kernel since it is only run by
one work-group. Unrolling loops in the diameter and perimeter kernels is nearly unusable
since it increases number of write ports to the local buffers to over 3, forcing port
sharing and lowering performance.

## v3

Uses shift register for floating-point reduction. Furthermore, unrolls the reduction
loop by creating a new fully unrolled loop inside the original loop, and correcting
the header of the original loop. The innermost loop in internal kernel is also fully
unrolled and memory load loops in perimeter and internal kernels are partially unrolled.
Finally, the "chunk" loops in the perimeter and internal kernels which were
unpipelineable due to dependency were moved into the host which also eliminated the
need for using expensive mod and mul operations in the kernel to calculate the loop
bounds.

## v4 (Best)

Parameters: BSZIE, DIA_UNROLL, PERI_UNROLL, PERI_COMPUTE, INTER_SIMD, INTER_COMPUTE

Compared to v2, uses a temp sum variable to reduce accesses to the shadow buffer
in the diameter kernel and the peri_col and peri_row buffers in the perimeter
kernel which reduces number of accesses and replication factor for these
buffers. Splits the shadow buffer in the diameter kernel and the dia buffer in
the perimeter kernel into to buffers, one of which is loaded row-wise and the
other, column-wise. This allows correct access coalescing under the presence
of loop unrolling in both kernels. peri_row is also transposed in the perimeter
kernel for the same purpose. Loads from external memory and writes to it modified
in the perimeter kernel to remove thread-id-dependent branching. Furthermore,
writing back the content of the peri_row buffer is merged into the compute loop to
remove one extra read port from this buffer. The same could be done with the write-back
of the content of the peri_col buffer; however, that would have resulted in a memory
access pattern that was not consecutive based on work-group ID and hence, resulted
in performance slow-down. This write-back was kept outside of the compute loop after
a barrier so that data can then be written back in a way that accesses are consecutive.
Also common subexpression elimination has been performed in all kernel,
and constant common subexpressions have been moved to the host code,
to minimize logic and DSPs used for integer arithmetic.