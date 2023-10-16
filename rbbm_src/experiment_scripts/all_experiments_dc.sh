#!/bin/bash


sizes=(10 20 30 40 50)
# sizes=(10 20 30 40 50)

declare -i repetitions=1
declare -i rseed
# filename="seed_file_dc_10.txt" 


global_runs=1

x=1
while [ $x -le $repetitions ]
do
	for n in ${sizes[@]}
		do
			python main.py -U dc -e experiment_results_folders/muse_repeatable_1012/ -R true -x 123 -X test_cases_seed.txt -T 0 -s ${n} -r 0.5  -D 0 -t tax -C /home/opc/chenjie/RBBM/rbbm_src/muse/data/mas/tax_rules.txt  -F /home/opc/chenjie/RBBM/rbbm_src/user_desired_dcs.txt -S ind -l critical -B True

			# echo "global_runs: $global_runs"
			# rseed=$(shuf -i 1-1000 -n 1)
			# echo "rand= ${rseed}"
			# python main.py -U dc -e experiment_results_folders/dc_quality_1012 -R true -x 123 -X test_cases_seed.txt -T 0 -s ${n} -r 0.5 -G naive -D 0 -t tax \
			# -C /home/opc/chenjie/RBBM/rbbm_src/muse/data/mas/tax_rules.txt   -F /home/opc/chenjie/RBBM/rbbm_src/user_desired_dcs.txt -S ind -l critical 
			# echo "python main.py -U dc -e experiment_results_folders/dc_quality_1012 -R true -x 123 -X test_cases_seed.txt -T 0 -s 20 -r 0.5 -G naive -D 0 -t tax \
			# -C /home/opc/chenjie/RBBM/rbbm_src/muse/data/mas/tax_rules.txt   -F /home/opc/chenjie/RBBM/rbbm_src/user_desired_dcs.txt -S ind -l critical " >>$filename
			# global_runs=$(( $global_runs+1 ))

			# python main.py -U dc -e experiment_results_folders/dc_quality_1012 -R true -x 123 -X test_cases_seed.txt -T 0 -s ${n} -r 0.5 -G information_gain -D 0 -t tax \
			# -C /home/opc/chenjie/RBBM/rbbm_src/muse/data/mas/tax_rules.txt   -F /home/opc/chenjie/RBBM/rbbm_src/user_desired_dcs.txt -S ind -l critical 
			# echo "python main.py -U dc -e experiment_results_folders/dc_quality_1009 -R true -x 123 -X test_cases_seed.txt -T 0 -s 20 -r 0.5 -G information_gain -D 0 -t tax \
			# -C /home/opc/chenjie/RBBM/rbbm_src/muse/data/mas/tax_rules.txt   -F /home/opc/chenjie/RBBM/rbbm_src/user_desired_dcs.txt -S ind -l critical " >>$filename
			# global_runs=$(( $global_runs+1 ))
			
			# python main.py -U dc -e experiment_results_folders/dc_quality_1012 -R true -x 123 -X test_cases_seed.txt -T 0 -s ${n} -r 0.5 -G optimal -D 0 -t tax \
			# -C /home/opc/chenjie/RBBM/rbbm_src/muse/data/mas/tax_rules.txt   -F /home/opc/chenjie/RBBM/rbbm_src/user_desired_dcs.txt -S ind -l critical 
			# echo "python main.py -U dc -e experiment_results_folders/dc_quality_1012 -R true -x 123 -X test_cases_seed.txt -T 0 -s 20 -r 0.5 -G optimal -D 0 -t tax \
			# -C /home/opc/chenjie/RBBM/rbbm_src/muse/data/mas/tax_rules.txt   -F /home/opc/chenjie/RBBM/rbbm_src/user_desired_dcs.txt -S ind -l critical " >>$filename
			# global_runs=$(( $global_runs+1 ))
		done
	x=$(( $x+1 ))
done

# 12:49

			# python main.py -U dc -e experiment_results_folders/dc_quality_1009 -R true -x 123 -X test_cases_seed.txt -T 0 -s 20 -r 0.5 -G optimal -D 0 -t tax -C /home/opc/chenjie/RBBM/rbbm_src/muse/data/mas/tax_rules.txt   -F /home/opc/chenjie/RBBM/rbbm_src/user_desired_dcs.txt -S ind -l critical 