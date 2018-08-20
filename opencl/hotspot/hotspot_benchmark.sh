#!/bin/bash

export CL_CONTEXT_COMPILER_MODE_ALTERA=3

runs=5
iter=1000
bench=hotspot
version=7
input_size=15000
folder=stratixv
board=de5net_a7

echo kernel | xargs printf "%-50s"
echo freq | xargs printf "%-10s"
echo last_col | xargs printf "%-10s"
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
	
	compute_bsize=$(($BSIZE - (2 * $TIME)))
	last_col=$(($input_size + $compute_bsize - $input_size % $compute_bsize))
	
	timesum=0
	bytessum=0
	energysum=0
	powersum=0

	rm $bench >/dev/null 2>&1; make ALTERA=1 HOST_ONLY=1 BOARD=$board TIME=$TIME SSIZE=$SSIZE BSIZE=$BSIZE >/dev/null 2>&1
	rm "$bench"_kernel_"v$version".aocx >/dev/null 2>&1
	ln -s "$folder"/"$i" "$bench"_kernel_"v$version".aocx
	aocl program acl0 "$bench"_kernel_"v$version".aocx >/dev/null 2>&1

	for (( k=1; k<=$runs; k++ ))
	do
		out=`CL_DEVICE_TYPE=CL_DEVICE_TYPE_ACCELERATOR ./$bench $last_col 1 $iter ../../data/hotspot/temp_16384 ../../data/hotspot/power_16384 v$version 2>&1`
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
	echo $last_col | xargs printf "%-10d"
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