#!/bin/bash

big_sizes=(110 130)
optimal_sizes=(10 30 50 70 90)
# num_funcs=(10 20 30 40 50)
complaint_ratio=(0.1 0.3 0.5 0.7 0.9)
# deletion_factor=(0.0 0.3 0.5 0.7)
dabsolutes=(4 6 8 10 12 14)

declare -i with_optimal_iterations=8
declare -i no_optimal_small_size_iterations=50
declare -i no_optimal_big_size_iterations=12


declare -i rseed
# filename="seed_file_930_all_lf_experiments.txt" 
# all_lf_experiments_1006

filename="seed_file_1017_new_2_algo_absolute.txt" 

global_runs=1

x=1
while [ $x -le $no_optimal_small_size_iterations ]
do
	for n in ${optimal_sizes[@]}
		do
			for c in ${complaint_ratio[@]}
				do
					for d in ${dabsolutes[@]}
						do
							echo "global_runs: $global_runs"
							rseed=$(shuf -i 1-2000 -n 1)

							python main.py -U lf -e experiment_results_folders/2_algo_absolute_del_1017/2_algo/ -R true -x ${rseed} -X test_cases_seed.txt -T 0 -s ${n} -r ${c} -G naive -b absolute -W ${d} -l critical \
							-K /home/opc/chenjie/RBBM/rbbm_src/labelling_func_src/src/pickled_funcs_720 -n youtube -P 5432 -k true -P 5432
							echo "python main.py -U lf -e experiment_results_folders/2_algo_absolute_del_1017/2_algo/ -R true -x ${rseed} -X test_cases_seed.txt -T 0 -s ${n} -r ${c} -G naive -b absolute -W ${d} -l critical \
							-K /home/opc/chenjie/RBBM/rbbm_src/labelling_func_src/src/pickled_funcs_720 -n youtube -P 5432  -k true " >> "$filename"

							global_runs=$(( $global_runs+1 ))

							python main.py -U lf -e experiment_results_folders/2_algo_absolute_del_1017/2_algo/ -R true -x ${rseed} -X test_cases_seed.txt -T 0 -s ${n} -r ${c} -G information_gain -b absolute -W ${d} -l critical \
							-K /home/opc/chenjie/RBBM/rbbm_src/labelling_func_src/src/pickled_funcs_720 -n youtube -P 5432 -k true 
							echo "python main.py -U lf -e experiment_results_folders/2_algo_absolute_del_1017/2_algo/ -R true -x ${rseed} -X test_cases_seed.txt -T 0 -s ${n} -r ${c} -G information_gain -b absolute -W ${d} -l critical \
							-K /home/opc/chenjie/RBBM/rbbm_src/labelling_func_src/src/pickled_funcs_720 -n youtube  -P 5432 -k true " >> "$filename" 

							global_runs=$(( $global_runs+1 ))
						done
				done
		done
	x=$(( $x+1 ))
done

x=1
while [ $x -le $no_optimal_big_size_iterations ]
do
	for n in ${big_sizes[@]}
		do
			for c in ${complaint_ratio[@]}
				do
					for d in ${dabsolutes[@]}
						do
							echo "global_runs: $global_runs"
							rseed=$(shuf -i 1-2000 -n 1)
							
							python main.py -U lf -e experiment_results_folders/2_algo_absolute_del_1017/2_algo/ -R true -x ${rseed} -X test_cases_seed.txt -T 0 -s ${n} -r ${c} -G naive -b absolute -W ${d} -l critical \
							-K /home/opc/chenjie/RBBM/rbbm_src/labelling_func_src/src/pickled_funcs_720 -n youtube -P 5432 -k true 
							echo "python main.py -U lf -e experiment_results_folders/2_algo_absolute_del_1017/2_algo/ -R true -x ${rseed} -X test_cases_seed.txt -T 0 -s ${n} -r ${c} -G naive -b absolute -W ${d} -l critical \
							-K /home/opc/chenjie/RBBM/rbbm_src/labelling_func_src/src/pickled_funcs_720 -n youtube -P 5432 -k true " >> "$filename"

							global_runs=$(( $global_runs+1 ))

							python main.py -U lf -e experiment_results_folders/2_algo_absolute_del_1017/2_algo/ -R true -x ${rseed} -X test_cases_seed.txt -T 0 -s ${n} -r ${c} -G information_gain -b absolute -W ${d} -l critical \
							-K /home/opc/chenjie/RBBM/rbbm_src/labelling_func_src/src/pickled_funcs_720 -n youtube -P 5432 -k true 
							echo "python main.py -U lf -e experiment_results_folders/2_algo_absolute_del_1017/2_algo/ -R true -x ${rseed} -X test_cases_seed.txt -T 0 -s ${n} -r ${c} -G information_gain -b absolute -W ${d} -l critical \
							-K /home/opc/chenjie/RBBM/rbbm_src/labelling_func_src/src/pickled_funcs_720 -n youtube -P 5432 -k true " >> "$filename"

							global_runs=$(( $global_runs+1 ))
						done
				done
		done
	x=$(( $x+1 ))
done