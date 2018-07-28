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
aoc [kernel_name] -g -v --report --board [board_name] -I../../ -DUSE_RESTRICT -DUNROLL=[UNROLL]
```


# Execution

Default run:

```
./run v[version_number]
```

Custom run:

```
./lavaMD -boxes1d <box_number> v[version_number]
```


# Kernel variations

See the github Wiki page for more general information. 

## v1

Straightforward single work-item kernel created based on the OpenMP
version. For the first two kernel arguments, only the necessary variable
is passed instead of the whole struct.

## v3

Uses shift register-based optimization for floating-point reduction,
"__attribute__((max_global_work_dim(0)))," and unrolling.
