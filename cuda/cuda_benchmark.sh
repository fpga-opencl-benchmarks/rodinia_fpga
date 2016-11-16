#!/bin/bash

benchmarks=(nw hotspot pathfinder srad lud cfd)
runs=5

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
	if [[ "$i" == "cfd" ]]
	then
		make KERNEL_DIM="-DRD_WG_SIZE_1=128 -DRD_WG_SIZE_2=192 -DRD_WG_SIZE_3=128 -DRD_WG_SIZE_4=256" > /dev/null 2>&1
	elif [[ "$i" == "nw" || "$i" == "pathfinder" ]]
	then
		make FOR=1 > /dev/null 2>&1
	else
		make > /dev/null 2>&1
	fi

	if [[ "$i" == "cfd" ]]
	then
		for (( k=1; k<=$runs; k++ ))
		do
			output=`./run`
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
			output=`./run`
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

