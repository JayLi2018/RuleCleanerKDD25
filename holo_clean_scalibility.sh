#!/bin/bash
trap "kill 0" EXIT 

echo "experiments for runtime vs filesize and runtime vs dcsize"

TIMEFORMAT=%R

dc_sizes=(10 20 40 80 160)
file_sizes=(500 1000 2000 4000 8000 16000)

for d in ${dc_sizes[@]}
	do
		for f in ${file_sizes[@]}
			do
				# start=`date +%s`
				start=`date +%s.%N`

				# echo -e "fsize=${f}, dcsize=${d}, time=${foo}" 
				time python main.py -s /home/opc/adult_samples/ -S adult_${f}.csv \
				-t /home/opc/adult_samples/ -T adult_${f}_clean.csv  \
				-c /home/opc/adult_rules/ -C rules_${d}
				
				# end=`date +%s`
				end=`date +%s.%N`

				runtime=$( echo "$end - $start" | bc -l )
				
				# runtime=$((end-start))
				echo -e "fsize=${f}, dcsize=${d}, time=${runtime}" >> holoclean_runtimes.txt 
			done
	done

		# -h, --help            show this help message and exit
		# -l, --log_level   loglevel: debug/info/warning/error/critical (default: critical)
		# -p, --user_provide 
		#                       user select from all wrong labels(for LF) or all wrong repairs(for DC)? (default: True)
		# -U, --use_case    use case of the run, is it for dc or lf? (default: dc)
		# -R, --sample_contingency 
		#                       when evaluating responsibility, sample contingency? (default: True)
		# -Z, --contingency_sample_times 
		#                       if choosing uniform sampling contingency, how many times do you want to sample? (default: 3)
		# -N, --random_number_for_complaint 
		#                       random number chosen to select a complaint: given a list of complaint L with size N, we choose L[N mod len(L)](default: 999)
		# -M, --training_model_type 
		#                       the model used to get the label: majority/snorkel (default: snorkel)
		# -W, --word_threshold 
		#                       word threshold when evaluating inflences of words(default: 0)
		# -A, --cardinality_thresh 
		#                       cardinality threshold if non greedy (i.e. exhaustive), ONLY userful when greedy==False (default: 4)
		# -L, --using_lattice 
		#                       using lattice when fiding rule influences? (default: False)
		# -E, --eval_mode   method used to evaluate the model (default: single_func)
		# -d, --dbname      dbname used during holoclean application (default: holo)
		# -u, --dbuser      dbuser used during holoclean application (default: holocleanuser)
		# -P, --dbpaswd     dbname used during holoclean application (default: abcd1234)
		# -c, --dc_dir      holoclean needs a input csv file as the starting point, this will be the dir it finds the file
		# -C, --dc_file     holoclean needs a input csv file as the starting point, this will be the dir it finds the file
		# -s, --input_csv_dir 
		#                       holoclean needs a input csv file as the starting point, this will be the dir it finds the file
		# -S, --input_csv_file 
		#                       holoclean needs a input csv file as the starting point, this will be the dir it finds the file
		# -t, --ground_truth_dir 
		#                       holoclean needs a input csv file as the starting point, this will be the dir it finds the file
		# -T, --ground_truth_file 
		#                       holoclean needs a input csv file as the starting point, this will be the dir it finds the file
		# -H, --contingency_size_threshold 
		#                       if enumerate and test contingency, up to what size do you want to try (default: 4)

# x=1
# while [ $x -le 5 ]
# do
# 	echo "iteration: $x"
# 	python main.py -U dc -p no -R True -N $RANDOM -C dc_sample_10
# 	python main.py -U dc -p no -R True -N $RANDOM -C dc_sample_20
# 	python main.py -U dc -p no -R True -N $RANDOM -C dc_sample_30
# 	x=$(( $x + 1 ))
# done

# mv *_run_* experiments/


# python main.py -U dc -p True -R True -C dc_sample_30