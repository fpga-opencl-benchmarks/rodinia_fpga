#!/bin/bash

export CL_CONTEXT_COMPILER_MODE_ALTERA=3

runs=5
iter=1000
bench=hotspot3D
version=7
input_size=490
no_inter=""
folder=arria10
board=p385a_sch_ax115

echo kernel | xargs printf "%-50s"
echo freq | xargs printf "%-10s"
echo last_x | xargs printf "%-10s"
echo last_y | xargs printf "%-10s"
echo time | xargs printf "%-10s"
echo bytes | xargs printf "%-10s"
if [[ "$board" == "p385a_sch_ax115" ]]
then
	echo energy | xargs printf "%-10s"
	echo power | xargs printf "%-10s"
fi
echo

for i in `ls $folder | grep aocx | grep v$version | sort -V`
do
	name="${i%.*}"
	echo "$name" | xargs printf "%-50s"
	TIME=`echo $name | cut -d "_" -f 4 | cut -c 5-`
	SSIZE=`echo $name | cut -d "_" -f 5 | cut -c 6-`
	BSIZE=`echo $name | cut -d "_" -f 6 | cut -c 6-`
	freq=`cat $folder/$name/acl_quartus_report.txt | grep Actual | cut -d " " -f 4`
	if [ -n `echo $name | grep nointer` ]
	then
		no_inter="NO_INTERLEAVE=1"
	fi
	if [ -z `echo $BSIZE | grep x` ]
	then
		BLOCK_X=$BSIZE
		BLOCK_Y=$BSIZE
	else
		BLOCK_X=`echo $BSIZE | cut -d "x" -f 1`
		BLOCK_Y=`echo $BSIZE | cut -d "x" -f 2`
	fi

	compute_bsize_x=$(($BLOCK_X - (2 * $TIME)))
	compute_bsize_y=$(($BLOCK_Y - (2 * $TIME)))
	last_col_x=$(($input_size + $compute_bsize_x - $input_size % $compute_bsize_x))
	last_col_y=$(($input_size + $compute_bsize_y - $input_size % $compute_bsize_y))

	timesum=0
	bytessum=0
	energysum=0
	powersum=0

	rm $bench >/dev/null 2>&1; make ALTERA=1 HOST_ONLY=1 BOARD=$board TIME=$TIME SSIZE=$SSIZE BLOCK_X=$BLOCK_X BLOCK_Y=$BLOCK_Y "$no_inter" >/dev/null 2>&1
	rm "$bench"_kernel_"v$version".aocx >/dev/null 2>&1
	ln -s "$folder"/"$i" "$bench"_kernel_"v$version".aocx
	aocl program acl0 "$bench"_kernel_"v$version".aocx >/dev/null 2>&1

	for (( k=1; k<=$runs; k++ ))
	do
		out=`CL_DEVICE_TYPE=CL_DEVICE_TYPE_ACCELERATOR ./$bench $last_col_x $last_col_y $last_col_x $iter v$version 2>&1`
		time=`echo "$out" | grep "Computation" | cut -d " " -f 4`
		bytes=`echo "$out" | grep "Throughput" | cut -d " " -f 3`
		if [[ "$board" == "p385a_sch_ax115" ]]
		then
			energy=`echo "$out" | grep energy | cut -d " " -f 5`
			power=`echo "$out" | grep power | cut -d " " -f 5`
		fi

		timesum=`echo $timesum+$time | bc -l`
		bytessum=`echo $bytessum+$bytes | bc -l`
		if [[ "$board" == "p385a_sch_ax115" ]]
		then
			energysum=`echo $energysum+$energy | bc -l`
			powersum=`echo $powersum+$power | bc -l`
		fi
	done

	timeaverage=`echo $timesum/$runs | bc -l`
	bytesaverage=`echo $bytessum/$runs | bc -l`
	if [[ "$board" == "p385a_sch_ax115" ]]
	then
		energyaverage=`echo $energysum/$runs | bc -l`
		poweraverage=`echo $powersum/$runs | bc -l`
	fi

	echo $freq | xargs printf "%-10.3f"
	echo $last_col_x | xargs printf "%-10d"
	echo $last_col_y | xargs printf "%-10d"
	echo $timeaverage | xargs printf "%-10.3f"
	echo $bytesaverage | xargs printf "%-10.3f"
	if [[ "$board" == "p385a_sch_ax115" ]]
	then
		echo $energyaverage | xargs printf "%-10.3f"
		echo $poweraverage | xargs printf "%-10.3f"
	fi
	echo
done

unset CL_CONTEXT_COMPILER_MODE_ALTERA