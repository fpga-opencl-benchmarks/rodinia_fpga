#!/bin/bash

CC=gcc
CXX=g++

benchmarks=(nw hotspot pathfinder srad lud cfd)
runs=5
if [ "$#" -ne 1 ]
then
	echo -e "Missing thread number!\nUsage: ./openmp_benchmark.sh num_threads"
	exit 1
else
	threads=$1
fi

# If the NUMA target is changed here, the core number used in the CPU power measurement header should also be changed
# or else the benchmarks will run on one CPU while the power usage is read from another
RUN_PREFIX="sudo LD_LIBRARY_PATH=$LD_LIBRARY_PATH OMP_NUM_THREADS=$threads numactl -N 0 -m 0"
MAKE_LOG=make.log
echo "Benchmark     Time (ms)     Energy Usage (J)  Average Power Consumption (Watts)"

for i in "${benchmarks[@]}"
do
	timesum=0
	energysum=0
	powersum=0
	timesum1=0
	energysum1=0
	powersum1=0

	if [[ "$i" == "srad" ]]
	then
		cd $i/srad_v1
	else
		cd $i
	fi

	make clean > /dev/null 2>&1
	echo `$CC --version` > $MAKE_LOG
	if [[ "$i" == "cfd" ]]
	then
		make OMP_NUM_THREADS=$threads CC=$CC CXX=$CXX >> $MAKE_LOG 2>&1
	else
		make CC=$CC CXX=$CXX >> $MAKE_LOG 2>&1
	fi

	if [[ "$i" == "cfd" ]]
	then
		for (( k=1; k<=$runs; k++ ))
		do
			output=`$RUN_PREFIX ./run`
			time=`echo "$output" | grep -m1 Computation | tail -n 1 | cut -d " " -f 4`
			time1=`echo "$output" | grep -m2 Computation | tail -n 1 | cut -d " " -f 4`

			energy=`echo "$output" | grep -m1 energy | tail -n 1 | cut -d " " -f 5`
			energy1=`echo "$output" | grep -m2 energy | tail -n 1 | cut -d " " -f 5`

			power=`echo "$output" | grep -m1 power | tail -n 1 | cut -d " " -f 5`
			power1=`echo "$output" | grep -m2 power | tail -n 1 | cut -d " " -f 5`

			timesum=`echo $timesum+$time | bc -l`
			timesum1=`echo $timesum1+$time1 | bc -l`

			energysum=`echo $energysum+$energy | bc -l`
			energysum1=`echo $energysum1+$energy1 | bc -l`

			powersum=`echo $powersum+$power | bc -l`
			powersum1=`echo $powersum1+$power1 | bc -l`
		done
	else
		for (( k=1; k<=$runs; k++ ))
		do
			output=`$RUN_PREFIX ./run $threads`
			time=`echo "$output" | grep Computation | cut -d " " -f 4`
			energy=`echo "$output" | grep energy | cut -d " " -f 5`
			power=`echo "$output" | grep power | cut -d " " -f 5`

			timesum=`echo $timesum+$time | bc -l`
			energysum=`echo $energysum+$energy | bc -l`
			powersum=`echo $powersum+$power | bc -l`
		done
	fi

	if [[ "$i" == "srad" ]]
	then
		cd ../..
	else
		cd ..
	fi

	averagetime=`echo $timesum/$runs | bc -l`
	averageenergy=`echo $energysum/$runs | bc -l`
	averagepower=`echo $powersum/$runs | bc -l`

	if [[ "$i" == "cfd" ]]
	then
		echo "cfd_eu" | xargs printf "%-14s"
	else
		echo $i | xargs printf "%-14s"
	fi

	echo $averagetime | xargs printf "%-14.3f"
	echo $averageenergy | xargs printf "%-18.3f"
	echo $averagepower | xargs printf "%-18.3f"
	echo

	if [[ "$i" == "cfd" ]]
	then
		averagetime=`echo $timesum1/$runs | bc -l`
		averageenergy=`echo $energysum1/$runs | bc -l`
		averagepower=`echo $powersum1/$runs | bc -l`

		echo "cfd_pre_eu" | xargs printf "%-14s"

		echo $averagetime | xargs printf "%-14.3f"
		echo $averageenergy | xargs printf "%-18.3f"
		echo $averagepower | xargs printf "%-18.3f"
		echo
	fi
done

