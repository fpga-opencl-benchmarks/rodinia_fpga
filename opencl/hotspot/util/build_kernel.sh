#!/usr/bin/env bash

kernel=hotspot_kernel
v=$1
bsize=$2
par=$3
board=$4
email=$5
aocx=${kernel}_v${v}_blocksize_${bsize}_parpoints_$par.aocx
#aocx=${kernel}_v${v}_blocksize_${bsize}_parpoints_${par}_fp_relaxed_fpc.aocx
make_log=make.v$v_$bsize_$par.log

{
    date
    make $aocx ALTERA=1 BOARD=$board EXTRA_MACROS="-DBLOCK_SIZE=$bsize -DPAR_POINTS=$par"
    date
} 2>&1 | tee $make_log

mailx -s $aocx $email < $make_log
