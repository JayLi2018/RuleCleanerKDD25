#!/bin/bash

# brute_sizes=(10 20 30 40)
brute_sizes=(60 80 100)
brute_sizes_80=(80)
sizes=(20 40 80 160 320)

# num_funcs=(10 20 30 40 50)
complaint_ratio=(0.1 0.3 0.5 0.7 0.9)

deletion_factor=(0.0 0.3 0.5 0.7)

deletion_factor_optimal=(0.0 0.5 0.7)
complaint_ratio_optimal=(0.3 0.7 0.9)

declare -i brute_iterations=15
declare -i non_brute_iterations=10
declare -i brute_iterations_80=1


declare -i rseed
filename3="seed_file_922_with_optimal.txt" 

# x=1
global_runs=1
# while [ $x -le $non_brute_iterations ]
# do
# 	for n in ${sizes[@]}
# 		do
# 			for c in ${complaint_ratio[@]}
# 				do
# 					for d in ${deletion_factor[@]}
# 						do
# 							echo "global_runs: $global_runs"
# 							rseed=$(shuf -i 1-1000 -n 1)
# 							# echo "rand= ${rseed}"
# 							python main.py -U lf -e experiment_results_folders/basics -R true -x  ${rseed} -X test_cases_seed.txt -T 0 -s ${n} -r ${c} -G naive -D ${d} -l critical \
# 							-K /home/opc/chenjie/RBBM/rbbm_src/labelling_func_src/src/pickled_funcs_720 -n youtube >> "$filename1"
# 							global_runs=$(( $global_runs+1 ))
# 							python main.py -U lf -e experiment_results_folders/basics -R true -x ${rseed} -X test_cases_seed.txt -T 0 -s ${n} -r ${c} -G information_gain -D ${d} -l critical \
# 							-K /home/opc/chenjie/RBBM/rbbm_src/labelling_func_src/src/pickled_funcs_720 -n youtube >> "$filename1"
# 							global_runs=$(( $global_runs+1 ))
# 							# echo "python main.py -U lf -e experiment_results_folders/basics -R true -x ${c} -X test_cases_seed.txt -T 0 -s ${n} -r ${c} -G naive -D ${d} -K /home/opc/chenjie/RBBM/rbbm_src/labelling_func_src/src/pickled_funcs_720 -n youtube "
# 							# echo "python main.py -U lf -e experiment_results_folders/basics -R true -x ${c} -X test_cases_seed.txt -T 0 -s ${n} -r ${c} -G information_gain -D ${d} -K /home/opc/chenjie/RBBM/rbbm_src/labelling_func_src/src/pickled_funcs_720 -n youtube "
# 						done
# 				done
# 		done
# 	x=$(( $x+1 ))
# done




# x=1
# while [ $x -le $brute_iterations_80 ]
# do
# 	for n in ${brute_sizes_80[@]}
# 		do
# 			for c in ${complaint_ratio[@]}
# 				do
# 					for d in ${deletion_factor[@]}
# 						do
# 							echo "global_runs: $global_runs"
# 							rseed=$(shuf -i 1-1000 -n 1)
# 							# echo "rand= ${rseed}"
# 							python main.py -U lf -e experiment_results_folders/with_optimal/basics -R true -x ${rseed} -X test_cases_seed.txt -T 0 -s ${n} -r ${c} -G naive -D ${d} -l critical \
# 							-K /home/opc/chenjie/RBBM/rbbm_src/labelling_func_src/src/pickled_funcs_720 -n youtube >> "$filename2"
# 							global_runs=$(( $global_runs+1 ))
# 							python main.py -U lf -e experiment_results_folders/with_optimal/basics -R true -x ${rseed} -X test_cases_seed.txt -T 0 -s ${n} -r ${c} -G information_gain -D ${d} -l critical \
# 							-K /home/opc/chenjie/RBBM/rbbm_src/labelling_func_src/src/pickled_funcs_720 -n youtube >> "$filename2"
# 							global_runs=$(( $global_runs+1 ))
# 							python main.py -U lf -e experiment_results_folders/with_optimal/basics -R true -x ${rseed} -X test_cases_seed.txt -T 0 -s ${n} -r ${c} -G optimal -D ${d} -l critical \
# 							-K /home/opc/chenjie/RBBM/rbbm_src/labelling_func_src/src/pickled_funcs_720 -n youtube >> "$filename2"
# 							global_runs=$(( $global_runs+1 ))

# 							# echo "python main.py -U lf -e experiment_results_folders/basics -R true -x ${c} -X test_cases_seed.txt -T 0 -s ${n} -r ${c} -G naive -D ${d} -K /home/opc/chenjie/RBBM/rbbm_src/labelling_func_src/src/pickled_funcs_720 -n youtube"
# 							# echo "python main.py -U lf -e experiment_results_folders/basics -R true -x ${c} -X test_cases_seed.txt -T 0 -s ${n} -r ${c} -G information_gain -D ${d} -K /home/opc/chenjie/RBBM/rbbm_src/labelling_func_src/src/pickled_funcs_720 -n youtube"
# 							# echo "python main.py -U lf -e experiment_results_folders/basics -R true -x ${c} -X test_cases_seed.txt -T 0 -s ${n} -r ${c} -G optimal -D ${d} -K /home/opc/chenjie/RBBM/rbbm_src/labelling_func_src/src/pickled_funcs_720 -n youtube"
# 						done
# 				done
# 		done
# 	x=$(( $x+1 ))
# done

x=1
while [ $x -le $brute_iterations ]
do
	for n in ${brute_sizes[@]}
		do
			echo "global_runs: $global_runs"
			rseed=$(shuf -i 1-1000 -n 1)
			# echo "rand= ${rseed}"
			python main.py -U lf -e experiment_results_folders/3_strats/ -R true -x ${rseed} -X test_cases_seed.txt -T 0 -s ${n} -r 0.5 -G naive -D 0 -l critical \
			-K /home/opc/chenjie/RBBM/rbbm_src/labelling_func_src/src/pickled_funcs_720 -n youtube 
			echo "python main.py -U lf -e experiment_results_folders/3_strats/ -R true -x ${rseed} -X test_cases_seed.txt -T 0 -s ${n} -r ${c} -G naive -D 0 -l critical \
			-K /home/opc/chenjie/RBBM/rbbm_src/labelling_func_src/src/pickled_funcs_720 -n youtube" >> "$filename3"
			global_runs=$(( $global_runs+1 ))
			
			python main.py -U lf -e experiment_results_folders/3_strats/ -R true -x ${rseed} -X test_cases_seed.txt -T 0 -s ${n} -r 0.5 -G information_gain -D 0 -l critical \
			-K /home/opc/chenjie/RBBM/rbbm_src/labelling_func_src/src/pickled_funcs_720 -n youtube 
			echo "python main.py -U lf -e experiment_results_folders/3_strats/ -R true -x ${rseed} -X test_cases_seed.txt -T 0 -s ${n} -r 0.5 -G information_gain -D 0 -l critical \
			-K /home/opc/chenjie/RBBM/rbbm_src/labelling_func_src/src/pickled_funcs_720 -n youtube" >> "$filename3"
			
			global_runs=$(( $global_runs+1 ))
			
			python main.py -U lf -e experiment_results_folders/3_strats/ -R true -x ${rseed} -X test_cases_seed.txt -T 0 -s ${n} -r 0.5 -G optimal -D 0 -l critical \
			-K /home/opc/chenjie/RBBM/rbbm_src/labelling_func_src/src/pickled_funcs_720 -n youtube 
			echo "python main.py -U lf -e experiment_results_folders/3_strats/ -R true -x ${rseed} -X test_cases_seed.txt -T 0 -s ${n} -r 0.5 -G optimal -D 0 -l critical \
			-K /home/opc/chenjie/RBBM/rbbm_src/labelling_func_src/src/pickled_funcs_720 -n youtube" >> "$filename3"

			global_runs=$(( $global_runs+1 ))
		done
	x=$(( $x+1 ))
done

# python main.py -U lf -e experiment_results_folders/exp_test_case_naive -R true -x 123 -X test_cases_seed.txt -T 0 -s 40 -r 0.5 -G naive -D 0 -K /home/opc/chenjie/RBBM/rbbm_src/labelling_func_src/src/pickled_funcs_720 -n youtube 


# optional arguments:
#   -h, --help            show this help message and exit
#   -U, --use_case    use case of the run, is it for dc or lf? (default: dc)
#   -e, --experiment_name 
#                         the name of the experiment, the results will be stored in the directory named with experiment_name_systime (default: test_blah)
#   -R, --repeatable  repeatable? (default: true)
#   -x, --seed        if repeatable, specify a seed number here (default: 123)
#   -X, --seed_file   if repeatable, specify a seed number here (default: seeds.txt)
#   -E, --retrain_every_percent 
#                         retrain over every (default: 1*100), the default order is sorted by treesize ascendingly
#   -A, --retrain_accuracy_thresh 
#                         when retrain over every retrain_every_percent, the algorithm stops when the fix rate is over this threshold (default: 0.5)
#   -T, --pre_filter_thresh 
#                         prefilter those DCs that have number of tuples involved in violations above this thresh (default: 0)
#   -l, --log_level   loglevel: debug/info/warning/error/critical (default: debug)
#   -s, --user_input_size 
#                         user input size total (the complaint size is decided by user_input_size*complaint_ratio) and confirm size is decided by user_input_size-complaint_size(default: 20)
#   -r, --complaint_ratio 
#                         user input complaint ratio (default: 0.5)
#   -G, --strategy    method used to repair the rules (naive, information_gain, optimal) (default: information gain)
#   -D, --deletion_factor 
#                         this is a factor controlling how aggressive the algorithm chooses to delete the rule from the rulset (default: 0.5)
#   -d, --dbname      database name which stores the dataset, (default: label)
#   -P, --port        database port, (default: 5432)
#   -p, --password    database password, (default: 5432)
#   -u, --user        database user, (default: postgres)
#   -f, --lf_source   the source of labelling function (intro / system generate) (default: intro)
#   -O, --number_of_funcs 
#                         if if_source is selected as system generate, how many do you want(default: 20)
#   -i, --run_intro   do you want to run the intro example with pre selected user input? (default: false)
#   -k, --load_funcs_from_pickle 
#                         (flag indicating if we want to load functions from a pickle file default: true)
#   -K, --pickle_file_name 
#                         (if load_funcs_from_pickle, then heres the pickle file name : placeholder_name)
#   -M, --training_model_type 
#                         the model used to get the label: majority/snorkel (default: snorkel)
#   -n, --dataset_name 
#                         dataset used in the use case of labelling functions (default: youtube)
#   -C, --dc_file     holoclean needs a input text file which contains the denial constraints, this will be the file inside dc_dir (default: /home/opc/chenjie/RBBM/rbbm_src/muse/data/mas/tax_rules.txt)
#   -S, --semantic_version 
#                         muse semantic version (ind/stage/end/step) (default: ind)
#   -t, --table_name  the table name from database cr that you want to work with (default: tax)
#   -F, --desired_dcs_file 
#                         the ground truth DCs that so called user think is correct (default: /home/opc/chenjie/RBBM/rbbm_src/dc_src/user_desired_dcs.txt)
#   -I, --user_specify_pairs 
#                         user specify pairs of violations to repair? (default: True)
