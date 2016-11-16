#!/usr/bin/env bash

EMAIL=$1
shift
OUT=$1
shift

{
	date
	make $*
	date
} 2>&1 | tee $OUT | mailx -s "make $*" $EMAIL 

