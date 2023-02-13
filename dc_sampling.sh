#!/bin/bash
trap "kill 0" EXIT 

echo "experiments for contingency sampling size to measure the effectiveness of sampling both on accuracy and run time"

x=1
while [ $x -le 5 ]
do
	echo "iteration: $x"
	python main.py -U dc -p no -R True -N $RANDOM -C dc_sample_10
	python main.py -U dc -p no -R True -N $RANDOM -C dc_sample_20
	python main.py -U dc -p no -R True -N $RANDOM -C dc_sample_30
	x=$(( $x + 1 ))
done

mv *_run_* experiments/


# python main.py -U dc -p True -R True -C dc_sample_30

rbbm -