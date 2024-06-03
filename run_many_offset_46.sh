#!/bin/bash
cd ~/local/src/victor/out2dcourse/codes_python/
for (( i=1; i<=99; i++ ))
do
	echo $i | python3 46_ray_tracing_interp_test_from_39.py
done


