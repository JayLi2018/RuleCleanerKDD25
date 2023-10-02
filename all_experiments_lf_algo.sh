#!/bin/bash


small_sizes=(20 40 80)
big_sizes=(160 320)
optimal_sizes=(10 30 50 70 100)
# num_funcs=(10 20 30 40 50)
complaint_ratio=(0.1 0.3 0.5 0.7 0.9)

deletion_factor=(0.0 0.3 0.5 0.7)

deletion_factor_optimal=(0.0 0.5 0.7)
complaint_ratio_optimal=(0.3 0.7 0.9)

declare -i with_optimal_iterations=2
declare -i no_optimal_small_size_iterations=50
declare -i no_optimal_big_size_iterations=20


declare -i rseed
filename="seed_file_930_all_lf_experiments.txt" 


global_runs=1


x=1
while [ $x -le $no_optimal_small_size_iterations ]
do
	for n in ${small_sizes[@]}
		do
			for c in ${complaint_ratio[@]}
				do
					for d in ${deletion_factor[@]}
						do
							echo "global_runs: $global_runs"
							rseed=$(shuf -i 1-2000 -n 1)
							# echo "rand= ${rseed}"
							python main.py -U lf -e experiment_results_folders/all_lf_experiments/2_algo/ -R true -x ${rseed} -X test_cases_seed.txt -T 0 -s ${n} -r ${c} -G naive -D ${d} -l critical \
							-K /home/opc/chenjie/RBBM/rbbm_src/labelling_func_src/src/pickled_funcs_720 -n youtube 
							echo "python main.py -U lf -e experiment_results_folders/all_lf_experiments/2_algo/ -R true -x ${rseed} -X test_cases_seed.txt -T 0 -s ${n} -r ${c} -G naive -D ${d} -l critical \
							-K /home/opc/chenjie/RBBM/rbbm_src/labelling_func_src/src/pickled_funcs_720 -n youtube " >> "$filename3"
							global_runs=$(( $global_runs+1 ))
							python main.py -U lf -e experiment_results_folders/all_lf_experiments/2_algo/ -R true -x ${rseed} -X test_cases_seed.txt -T 0 -s ${n} -r ${c} -G information_gain -D ${d} -l critical \
							-K /home/opc/chenjie/RBBM/rbbm_src/labelling_func_src/src/pickled_funcs_720 -n youtube 
							echo "python main.py -U lf -e experiment_results_folders/all_lf_experiments/2_algo/ -R true -x ${rseed} -X test_cases_seed.txt -T 0 -s ${n} -r ${c} -G information_gain -D ${d} -l critical \
							-K /home/opc/chenjie/RBBM/rbbm_src/labelling_func_src/src/pickled_funcs_720 -n youtube " >> "$filename3"
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
					for d in ${deletion_factor[@]}
						do
							echo "global_runs: $global_runs"
							rseed=$(shuf -i 1-2000 -n 1)
							# echo "rand= ${rseed}"
							python main.py -U lf -e experiment_results_folders/all_lf_experiments/2_algo/ -R true -x ${rseed} -X test_cases_seed.txt -T 0 -s ${n} -r ${c} -G naive -D ${d} -l critical \
							-K /home/opc/chenjie/RBBM/rbbm_src/labelling_func_src/src/pickled_funcs_720 -n youtube 
							echo "python main.py -U lf -e experiment_results_folders/all_lf_experiments/2_algo/ -R true -x ${rseed} -X test_cases_seed.txt -T 0 -s ${n} -r ${c} -G naive -D ${d} -l critical \
							-K /home/opc/chenjie/RBBM/rbbm_src/labelling_func_src/src/pickled_funcs_720 -n youtube " >> "$filename3"
							global_runs=$(( $global_runs+1 ))
							python main.py -U lf -e experiment_results_folders/all_lf_experiments/2_algo/ -R true -x ${rseed} -X test_cases_seed.txt -T 0 -s ${n} -r ${c} -G information_gain -D ${d} -l critical \
							-K /home/opc/chenjie/RBBM/rbbm_src/labelling_func_src/src/pickled_funcs_720 -n youtube 
							echo "python main.py -U lf -e experiment_results_folders/all_lf_experiments/2_algo/ -R true -x ${rseed} -X test_cases_seed.txt -T 0 -s ${n} -r ${c} -G information_gain -D ${d} -l critical \
							-K /home/opc/chenjie/RBBM/rbbm_src/labelling_func_src/src/pickled_funcs_720 -n youtube " >> "$filename3"
							global_runs=$(( $global_runs+1 ))
						done
				done
		done
	x=$(( $x+1 ))
done

x=1
while [ $x -le $no_optimal_small_size_iterations ]
do
	for n in ${brute_sizes[@]}
		do
			echo "global_runs: $global_runs"
			rseed=$(shuf -i 1-1000 -n 1)
			# echo "rand= ${rseed}"
			python main.py -U lf -e experiment_results_folders/all_lf_experiments/3_algo/ -R true -x ${rseed} -X test_cases_seed.txt -T 0 -s ${n} -r 0.5 -G naive -D 0 -l critical \
			-K /home/opc/chenjie/RBBM/rbbm_src/labelling_func_src/src/pickled_funcs_720 -n youtube 
			echo "python main.py -U lf -e experiment_results_folders/all_lf_experiments/3_algo/ -R true -x ${rseed} -X test_cases_seed.txt -T 0 -s ${n} -r ${c} -G naive -D 0 -l critical \
			-K /home/opc/chenjie/RBBM/rbbm_src/labelling_func_src/src/pickled_funcs_720 -n youtube" >> "$filename"
			global_runs=$(( $global_runs+1 ))
			
			python main.py -U lf -e experiment_results_folders/all_lf_experiments/3_algo/ -R true -x ${rseed} -X test_cases_seed.txt -T 0 -s ${n} -r 0.5 -G information_gain -D 0 -l critical \
			-K /home/opc/chenjie/RBBM/rbbm_src/labelling_func_src/src/pickled_funcs_720 -n youtube 
			echo "python main.py -U lf -e experiment_results_folders/all_lf_experiments/3_algo/ -R true -x ${rseed} -X test_cases_seed.txt -T 0 -s ${n} -r 0.5 -G information_gain -D 0 -l critical \
			-K /home/opc/chenjie/RBBM/rbbm_src/labelling_func_src/src/pickled_funcs_720 -n youtube" >> "$filename"
			
			global_runs=$(( $global_runs+1 ))
		done
	x=$(( $x+1 ))
done


x=1
while [ $x -le $with_optimal_iterations ]
do
	for n in ${brute_sizes[@]}
		do
			echo "global_runs: $global_runs"
			rseed=$(shuf -i 1-1000 -n 1)
			# echo "rand= ${rseed}"
			python main.py -U lf -e experiment_results_folders/all_lf_experiments/3_algo/ -R true -x ${rseed} -X test_cases_seed.txt -T 0 -s ${n} -r 0.5 -G naive -D 0 -l critical \
			-K /home/opc/chenjie/RBBM/rbbm_src/labelling_func_src/src/pickled_funcs_720 -n youtube 
			echo "python main.py -U lf -e experiment_results_folders/all_lf_experiments/3_algo/ -R true -x ${rseed} -X test_cases_seed.txt -T 0 -s ${n} -r ${c} -G naive -D 0 -l critical \
			-K /home/opc/chenjie/RBBM/rbbm_src/labelling_func_src/src/pickled_funcs_720 -n youtube" >> "$filename"
			global_runs=$(( $global_runs+1 ))
			
			python main.py -U lf -e experiment_results_folders/all_lf_experiments/3_algo/ -R true -x ${rseed} -X test_cases_seed.txt -T 0 -s ${n} -r 0.5 -G information_gain -D 0 -l critical \
			-K /home/opc/chenjie/RBBM/rbbm_src/labelling_func_src/src/pickled_funcs_720 -n youtube 
			echo "python main.py -U lf -e experiment_results_folders/all_lf_experiments/3_algo/ -R true -x ${rseed} -X test_cases_seed.txt -T 0 -s ${n} -r 0.5 -G information_gain -D 0 -l critical \
			-K /home/opc/chenjie/RBBM/rbbm_src/labelling_func_src/src/pickled_funcs_720 -n youtube" >> "$filename"
			
			global_runs=$(( $global_runs+1 ))
			
			python main.py -U lf -e experiment_results_folders/all_lf_experiments/3_algo/ -R true -x ${rseed} -X test_cases_seed.txt -T 0 -s ${n} -r 0.5 -G optimal -D 0 -l critical \
			-K /home/opc/chenjie/RBBM/rbbm_src/labelling_func_src/src/pickled_funcs_720 -n youtube 
			echo "python main.py -U lf -e experiment_results_folders/all_lf_experiments/3_algo/ -R true -x ${rseed} -X test_cases_seed.txt -T 0 -s ${n} -r 0.5 -G optimal -D 0 -l critical \
			-K /home/opc/chenjie/RBBM/rbbm_src/labelling_func_src/src/pickled_funcs_720 -n youtube" >> "$filename"

			global_runs=$(( $global_runs+1 ))
		done
	x=$(( $x+1 ))
done



# python main.py -U lf -e experiment_results_folders/all_lf_experiments/2_algo/ -R true -x 1 -X test_cases_seed.txt -T 0 -s 20 -r 0.5 -G naive -D ${d} -l critical -K /home/opc/chenjie/RBBM/rbbm_src/labelling_func_src/src/pickled_funcs_720 -n youtube 