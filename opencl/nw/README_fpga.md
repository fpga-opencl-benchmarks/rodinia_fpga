# Compilation

The simplest way to build kernels is to use build.sh as:

```
./build.sh
```

This will build the default set of kernels. 

Each kernel has some performance-sensitive parameters, which are
described below. To compile those kernels with its parameter
configuration, pass optional argument to the make command. For
example, build the kernel of version 9 with BSIZE set to 128:

```
make ALTERA=1 board=de5net_a7 v=9 BSIZE=128 kernel
```

This will build a binary named nw_kernel_v9_BSIZE128.aocx from
nw_kernel_v9.cl with BSIZE set to 128.

The build.sh script also allows to set those parameters and
additionally it can be instructed to send email after compilation. 

# Execution

To run the benchmark with problem size of 1024^2:

```
./nw 1024 10 [version]
```

For example, to run the v9 kernel:

```
./nw 1024 10 128 nw_kernel_v9_BSIZE128.aocx
```

# Kernel variations

See the github Wiki page for more general information. 

## v1

A straightforward single work-item kernel plus restrict keyword.
Iterates through the 2-dimensional space from top left to bottom right.
Due to load/store dependency on the input_itemsets variable, the outer
loop is run sequentially and the inner loop is run with an II of 328
(latency of two memory accesses).

## v3

This version saves the output of each iteration in a backup buffer
to be used for left dependency in the next iteration. This, coupled
with #pragma ivdep on the input_itemsets variable allows full pipelining
on the inner loop. The outer loop is still run sequentially due to
dependency on the same variable which is unavoidable in this implementation.
Unrolling the inner loop results in new load/store dependencies and hence,
was avoided.

## nw_kernel_single_work_item_volatile.cl (formerly v5)

Similer to the method presented in [Settle, HPEC'13]
(http://hgpu.org/?p=10816), it uses a local fixed-sized array to
propagate the computed values down to the vertical iterations. The
array should work as a shift register.

Since the shift register size is limited, this version uses 1-D
blocking, where each kernel invocation only computes a sub block of
max_rows*BLOCK_SIZE. The block size can be changed as follows:

```
make KERNEL_DIM="-BLOCK_SIZE=64"
```

The size of the block affects the performance and the compilation time
considerably. A smaller size results in less usage of FPGA resources
and a faster clock speed, but requires a larger number of kernel
invocations. Block size of 64, which is the default for de5net_a7,
seems to works best with Stratix 5 for a 2D matrix of 2048^2. Block
size of 128 or larger resulted in fitting errors.

Note that since the kernel uses the single array as input and output,
the compiler was originally unable to pipeline the loop. The
workaround used in this version is to mimic the compiler by passing
the array as two separate (restricted) pointer parameters for input
and output. This way, the compiler can reason that the loop does not
have data dependency and it seems to pipeline the loop. However,
likely due to this technically incorrect usage of restrict parameters,
the kernel invoked second or thereafter seems to have incorrect
caching behavior. Specifying "volatile" to input_itemsets removes the
problem.

## v7

Parameters: BSIZE

This version basically uses the same pipelining scheme as v5, but does
not use the trick to hide the data dependency. It correctly uses
restrict and volatile is not necessary here. To do so, it uses two 1-D
arrays that correspond to vertical columns at the two ends of the sub
region. It also uses a 1-D array that corresponds to the top-most
horizontal row. This way, there is no read to the final output matrix,
so the compiler can safely pipeline the loop.

It also uses const for constant arrays, which seems to result in a
little faster performance.

Unlike v5, de5net_a7 can fit the block size of 128. Performance does
not differ much for 2048^2, but about 10% faster for 4096 (16 ms
vs. 18 ms). It can't fit the block size of 256.

## single_work_item_2d_blocking

This used to be v7 at commit
f4dea53a210b606eb589932adb03ae42e8eeae12. It was the first attempt to
apply shift registers, and used 2-D blocking, while the current
version uses 1-D blocking. It seems just 1-D blocking is better than
2-D. 

## v9 (BEST)

Based on v7.

Parameters: BSIZE

Mostly the same as v7, but the dimensions of the output_itemsets array
and others are smaller by one, which is slightly more efficient than
v7 because of better memory alignment. The original kernel has one
extra row and column, but since we are using saparate 1D arrays for
them, those extra areas are not necessary.

## v11

Based on v7.

This version uses channels to directly transfer the boundary data from
one kernel to the next kernel. Less efficient than v7.

## v13

Based on v11.

Similar to v11, but uses 3 channels to create 4 pipelines directly
connected with channels. Slower than v7.

Note that nw.c does not seem to have code to correctly use this
version of the kernel.

## v15

Based on v7.

It creates 2 pipelines to exploit the pipeline parallelism
across those 2 pipelines. The first pipeline computes the upper half
of the matrix, and the other does the lower half. There is a
dependency from the upper pipeline to the lower when computing the
same band of columns, but there is no depency diagnorally. The two
pipelines are invoked in a wavefront way as illustrated below:

Time step 1:

```
[X] [ ]
[ ] [ ]
```

The sub matrix with "X" is computed at time step 1.

Time step 2:

```
[ ] [X]
[X] [ ]
```

Next, both the upper-right and lower-left sub matrices can be computed
in parallel.

Time step 3:

```
[ ] [ ]
[ ] [X]
```

Finally, the remaining lower-right sub matrix is comuputed by the
second pipeline.

Turned out that this is not effective. Still v9 is faster on Stratix V.

