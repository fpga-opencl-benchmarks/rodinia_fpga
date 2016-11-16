# Kernel variations

See the github Wiki page for more general information. 

## v1

Straightforward single work-item kernel created by wrapping v0 in a
for loop from 0 to numRecords and adding strict.

## v2

Optimized NDrange kernel by adding restrict, reqd_work_group_size,
num_simd_work_items and num_compute_units to v0. SIMD_LANES 16 is
the maximum number allowed by the compiler. COMPUTE_UNITS=3 was
chosen to avoid fully utilizing the DSPs.

## v3

Single work-item kernel created by adding #pragma unroll 48 to v1.
#pragma unroll 64 fully utilizes the DSP units, so it was avoided
at this point.

## v4

Same as v2 but with COMPUTE_UNITS=4 (resulting in 100% DSP utlization
and lower operating frequency).

## v5

Same as v3 but with #pragma unroll 64 (resulting in 100% DSP utlization
and lower operating frequency).

