#!/bin/bash

export CL_CONTEXT_COMPILER_MODE_ALTERA=3

benchmarks=(nw hotspot hotspot3D pathfinder srad lavaMD lud backprop cfd_eu cfd_pre_eu)

nw_version=(0 1 2 3 5)
hotspot_version=(0 1 2 3 4 5)
hotspot3D_version=(0 1 2 3 5)
pathfinder_version=(1 2 3 4)
srad_version=(0 1 2 3 5)
lavaMD_version=(0 1 3)
lud_version=(0 1 2 3 4)
backprop_version=(0 1 3)
cfd_eu_version=(0 1 2 3 4 5 7)
cfd_pre_eu_version=(0 1 2 3)

runs=5
power_ok=0

for i in "${benchmarks[@]}"
do
	echo "$i" | xargs printf "%-15s"
	cd $i

	version_array="$i"_version[@]
	for j in ${!version_array}
	do
		echo -n "v$j: "
		if [[ "$i" == "srad" ]]
		then
			aocl program acl0 kernel/"$i"_kernel_"v$j".aocx >/dev/null 2>&1
		elif [[ "$i" == "lud" ]]
		then
			aocl program acl0 ocl/"$i"_kernel_"v$j".aocx >/dev/null 2>&1
		else
			aocl program acl0 "$i"_kernel_"v$j".aocx >/dev/null 2>&1
		fi
		timesum=0
		powersum=0
		for (( k=1; k<=$runs; k++ ))
		do
			out=`CL_DEVICE_TYPE=CL_DEVICE_TYPE_ACCELERATOR ./run v$j 2>&1`

			time=`echo "$out" | grep Computation | cut -d " " -f 4`
			timesum=`echo $timesum+$time | bc -l`
			
			if [[ ! -z `echo "$out" | grep power | cut -d " " -f 5` ]]
			then
				power_ok=1
				power=`echo "$out" | grep power | cut -d " " -f 5`
				powersum=`echo $powersum+$power | bc -l`
			fi
		done

		timeaverage=`echo $timesum/$runs | bc -l`
		poweraverage=`echo $powersum/$runs | bc -l`

		echo $timeaverage | xargs printf "%-15.3f"
		if [[ $power_ok -eq 1 ]]
		then
			echo $poweraverage | xargs printf "%-10.3f"
		fi
	done

	echo
	cd ..
done

unset CL_CONTEXT_COMPILER_MODE_ALTERA
