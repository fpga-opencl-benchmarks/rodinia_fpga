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
aoc [kernel_name] -g -v --report --board [board_name] -I../../ -DUSE_RESTRICT -D[parameter_name]=[parameter_value]
```


# Execution

Default run:

```
./run v[version_number]
```

Custom run:

```
./backprop <network_size> v[version_number]
```


# Kernel variations

See the github Wiki page for more general information. 

## v1

Straightforward single work-item kernel created based on the OpenMP
version plus restrict.

## v3

Uses shift register-based optimization for floating-point reduction
in the first three kernels and also adds #pragma ivdep to both loops
in the last kernel to avoid false load/store dependency on w and
oldw global buffers. "__attribute__((max_global_work_dim(0)))" is
also used.
