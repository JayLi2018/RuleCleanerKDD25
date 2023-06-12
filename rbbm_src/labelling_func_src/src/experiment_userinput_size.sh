#!/bin/bash

sizes=(10 20 40 80 160 320)
sizes_brute=(5 10 20 30)
num_funcs=(10 20 30 40 50)
complaint_ratio=(0.1 0.3 0.5 0.7 0.9)
deletion_factor=(0.0 0.3 0.5 0.7)

declare -i brute_iterations=1
declare -i non_brute_iterations=1

# x=1
# while [ $x -le $brute_iterations ]
# do
# 	for n in ${num_funcs[@]}
# 		do
# 			for c in ${complaint_ratio[@]}
# 				do
# 					for d in ${deletion_factor[@]}
# 						do
# 							python LFRepair.py -U 10 -t ${c}  -e experiment_diff_num_lfs_611 -n ${n} -r brute_force -f sysgen -D ${d}
# 							echo "python LFRepair.py -U 10 -t ${c}  -e experiment_diff_num_lfs_611 -n ${n} -r brute_force -f sysgen -D ${d}"
# 						done
# 				done
# 		done
# 	x=$(( $x+1 ))
# done


# x=1
# while [ $x -le $brute_iterations ]
# do
# 	for s in ${sizes_brute[@]}
# 		do
# 			for c in ${complaint_ratio[@]}
# 				do
# 					for d in ${deletion_factor[@]}
# 						do
# 							python  LFRepair.py -U ${s} -t ${c} -e exp_vary_user_input_and_comp_ratio611 -r brute_force -f intro -D ${d}
# 							echo "python  LFRepair.py -U ${s} -t ${c} -e exp_vary_user_input_and_comp_ratio611 -r brute_force -f intro -D ${d}"
# 						done
# 				done
# 		done
# 	x=$(( $x+1 ))
# done




x=1
while [ $x -le $non_brute_iterations ]
do
	for n in ${num_funcs[@]}
		do
			for c in ${complaint_ratio[@]}
				do
					for d in ${deletion_factor[@]}
						do
							python LFRepair.py -U 50 -t ${c}  -e experiment_diff_num_lfs_611 -n ${n} -r information_gain -f sysgen -D ${d}
							echo "python LFRepair.py -U 50 -t ${c}  -e experiment_diff_num_lfs_611 -n ${n} -r information_gain -f sysgen -D ${d}"
							python LFRepair.py -U 50 -t ${c}  -e experiment_diff_num_lfs_611 -n ${n} -r naive -f sysgen -D ${d}
							echo "python LFRepair.py -U 50 -t ${c}  -e experiment_diff_num_lfs_611 -n ${n} -r naive -f sysgen -D ${d}"
						done
				done
		done
	x=$(( $x+1 ))
done


x=1
while [ $x -le $non_brute_iterations ]
do
	for s in ${sizes[@]}
		do
			for c in ${complaint_ratio[@]}
				do
					for d in ${deletion_factor[@]}
						do
							python LFRepair.py -U ${s} -t ${c} -e exp_vary_user_input_and_comp_ratio611 -r information_gain -f intro -D ${d}
							echo "python LFRepair.py -U ${s} -t ${c} -e exp_vary_user_input_and_comp_ratio611 -r information_gain -f intro -D ${d}"
							python LFRepair.py -U ${s} -t ${c} -e exp_vary_user_input_and_comp_ratio611 -r naive -f intro -D ${d}
							echo "python LFRepair.py -U ${s} -t ${c} -e exp_vary_user_input_and_comp_ratio611 -r naive -f intro -D ${d}"
						done
				done
		done	
	x=$(( $x+1 ))
done
