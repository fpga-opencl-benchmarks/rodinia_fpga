# Compilation

The simplest way to build kernels is to use build.sh as:

```
./build.sh
```

This will build the default set of kernels.

# Execution

To run the benchmark with problem size of 1024^2:

```
./euler3d input-file iteration-count block-size [version]
```

The block-size parameter is ignored in single-thread kernels.

# Kernel variations

See the github Wiki page for more general information.

## Single-thread versions

### v1

A straightforward single work-item kernel that is created by wrapping the original multi-threaded kernel with a loop for iterating all the elements.

### v3

Based on v1. Add restrict to input arrays.

### v5 (BEST on Stratix V)

Based on v3. Improves the throughput of the inner-most loop where a reduction is performed into a variable in v3. A shift register-based reduction is used instead in this version.

### v7

Based on v3. Unrolls the inner-most loop for iterating the neighbor elements. Unrolling makes the shift register-based reduction unnecessary, so this is effectively v3 with just unrolling. Does not fit Terasic Stratix V DE5NET_A7.

## Multi-thread versions

### v2

Add RESTRICT and reqd_work_group_size to v0. BSIZE is used to specify the block size. Note that the output slightly depends on the value of block size.

### v2_float3

Based on v3. Uses the builtin float3 instead of its own FLOAT3 type. Expected the compiler can perhaps generate efficient pipeline  for the builtin vector type instead of the custom type, but no difference with the v15 SDK.

### v4

Unrolls the inner-most loop in v2. Does not fit Terasic Stratix V.

### v6

Based on v4. Replicates the kernel pipeline by CUSIZE times. Does not fit Terasic Stratix V.

### v8

Based on v6. Use volatile to disable caching. This may be necessary to reduce resource usage of v6 to fit Arria 10.  Not confirmed whether the caching is actually disabled.
