#!/bin/bash

version=(4)
iteration=5

echo -e "Version" | xargs printf "%-10s"
echo -e "Time" | xargs printf "%-10s"
echo -e "Power" | xargs printf "%-10s"
echo -e "Energy"

for j in "${version[@]}"
do
	echo -e "$j" | xargs printf "%-10s"
	timesum=0
	powersum=0
	energysum=0
	for (( k=1; k<=$iteration; k++ ))
	do
		output=`CL_DEVICE_TYPE=CL_DEVICE_TYPE_ACCELERATOR ./run v$j 2>&1`

		time=`echo "$output" | grep Computation | cut -d " " -f 4`
		power=`echo "$output" | grep power | cut -d " " -f 5`
		energy=`echo "$output" | grep energy | cut -d " " -f 5`

		timesum=`echo $timesum+$time | bc -l`
		powersum=`echo $powersum+$power | bc -l`
		energysum=`echo $energysum+$energy | bc -l`
	done
	averagetime=`echo $timesum/$iteration | bc -l | xargs printf "%.3f"`
	averagepower=`echo $powersum/$iteration | bc -l | xargs printf "%.2f"`
	averageenergy=`echo $energysum/$iteration | bc -l | xargs printf "%.2f"`

	echo $averagetime | xargs printf "%-10s"
	echo $averagepower | xargs printf "%-10s"
	echo $averageenergy
done

