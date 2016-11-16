# Compilation

Use the build script as follows:

```
./build.sh
```

This will build a default set of kernels. Alternatively, the make command can be used to build a specific kernel variation:

```
make ALTERA=1 board=de5net_a7 v=9 kernels
```

To build the host program, run the command as follwos:

```
make ALTERA=1 HOST_ONLY
```

# Execution

The benchmark has a number of arguments:

```
./hotspot <matrix-size> <temporal-blocking-degree> <number-of-iterations> <input-temperature-file> <input-power-file> <output-file> <block-size> [kernel-version]
```

The arguments except for the last two are the same as the original version. The block-size argument is the BSIZE parameter in the kernels. The last kernel-version argument is the string appended to the kernel name in the compiled aocx file name.

Eg.:

```
./hotspot 1024 1 1000 ../../data/hotspot/temp_1024 ../../data/hotspot/power_1024 output.txt 512 v9_BSIZE512_SSIZE8
```

Note that in the current implementation the temporal blocking parameter must be 1 as the blocking is omitted.

# Kernel variations

## hotspot_kernel_original

- The original kernel.

## v0

- Based on the original kernel.
- Simplified by undoing the temporal blocking optimization in order to
simplify performance analysis. The temporal blocking optimization may
be applied again as an advanced optimization.

## v1

- Straightforward sequential implementation with a doubly nested loop.

## v2

- Based on v0
- Adds restrict to v0
- Parameters: BSIZE, SIMD, and CUSIZE

## v3

- Based on v1
- Parameters: BSIZE
- Add restrict. Unrolling the inner loop by BSIZE.

## v5

- Based on v3
- Parameters: BSIZE
- Vertically decomposes a given 2D matrix to columns of length BSIZE,
  and attempt to pipeline the processing of vertical loop with the
  BSIZE-length horizontal loop completely unrolled.

## v7

- Parameters: BSIZE, SSIZE. BSIZE needs to be divisible by SSIZE.
- Similar to v5, but does not completely unroll the horizontal loop,
  but just unrolls SSIZE iterations. It has one loop to traverse the
  sub matrix, and one outerloop that traverses across the columns.

## v9

- Parameters: BSIZE, SSIZE. BSIZE needs to be divisible by SSIZE.
- Similar to v7, a given matrix is decomposed vertically to BSIZE
  columns. Unlike v7, it does not use the loop nest since the Altera
  compiler seems to have problems with generating efficient pipelines
  from nested loops.

## v11 (BEST)
- Parameters: BSIZE, SSIZE. BSIZE needs to be divisible by SSIZE.
- This version uses the v9 implementation as base and adds a second
  very similar kernel so that the process is done two iterations at
  a time, with data traveling between the two kernels using channels.
  This significantly reduces memory load since both the output of the
  first iteration and the content of the power buffer are forwarded
  to the second kernel through channels.
  The second kernel starts from x = -1, instead of 0. This way, it is
  always one column behind the first kernel, while having the same
  BSIZE and SSIZE. On the boundary, the last column of each block from
  the first kernel will be processed as the first column of the next
  block in the second kernel, when its right neighbor has also already
  been computed by the first kernel. Though the left and current indexes
  for that column are written to and read from a global buffer since they
  cannot be transferred via channels due to possible large size of the
  necessary buffer which depends on input size.
