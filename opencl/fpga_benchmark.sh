#!/bin/bash

benchmarks=(nw hotspot hotspot3D pathfinder srad lavaMD lud backprop cfd_eu cfd_pre_eu)

nw_version=(0 1 2 3 4 7 9 11 13 15)
hotspot_version=(0 1 2 3 5 7 9 11)
hotspot3D_version=(0 1 3)
pathfinder_version=(0 1 2 3 4 11 13 15)
srad_version=(0 1 2 3 5)
lavaMD_version=(0 1 3)
lud_version=(0 1 2 3 4 5)
backprop_version=(0 1 3)
cfd_eu_version=(0 1 2 3 4 5 7)
cfd_pre_eu_version=(0 1 2 3)

iteration=5

for i in "${benchmarks[@]}"
do
	echo "$i" | xargs printf "%-15s"
	cd $i

	version_array="$i"_version[@]
	for j in ${!version_array}
	do
		echo -n "v$j: "
		sum=0
		for (( k=1; k<=$iteration; k++ ))
		do
			time=`CL_DEVICE_TYPE=CL_DEVICE_TYPE_ACCELERATOR ./run v$j 2>&1 | grep Computation | cut -d " " -f 4`
			sum=`echo $sum+$time | bc -l`
		done
		average=`echo $sum/$iteration | bc -l`
		echo $average | xargs printf "%-15.3f"
	done

	echo
	cd ..
done
