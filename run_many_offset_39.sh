#!/bin/bash
cd ~/local/src/victor/out2dcourse/codes_python/
for (( i=63; i<=99; i++ ))
do
	echo $i | python3 39_ray_tracing_interp_test_from_37.py
done


