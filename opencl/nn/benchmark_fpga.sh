#!/bin/bash

filelist=`ls | grep filelist`
version=(0 1 2 3 4 5)

echo -n "File list/Version     "
for i in "${version[@]}"
do
	echo -n "$i          "
done
echo

for i in $filelist
do
	echo $i | xargs printf "%-22s"
	for j in "${version[@]}"
	do
		sum=0
		for k in {1..3}
		do
			time=`CL_DEVICE_TYPE=CL_DEVICE_TYPE_ACCELERATOR ./nn $i -r 5 -lat 30 -lng 90 v$j 2>&1 | grep Records -A 1 | grep -v Records | cut -f 3`
			sum=`echo $sum+$time | bc -l`
		done
		average=`echo $sum/3 | bc -l | xargs printf "%.3f"`
		echo $average | xargs printf "%-11.3f"
	done
	echo
done
