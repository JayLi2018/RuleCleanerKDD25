#!/bin/bash

# brute_sizes=(40 80)
sizes=(320)

# sizes_brute=(5 10 20 30)
# num_funcs=(10 20 30 40 50)
complaint_ratio=(0.1 0.3 0.5 0.7 0.9)
# deletion_factor=(0.0 0.3 0.5 0.7)
deletion_factor=(0.0)

declare -i brute_iterations=5
declare -i non_brute_iterations=1500

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




# x=1
# while [ $x -le $non_brute_iterations ]
# do
# 	for n in ${num_funcs[@]}
# 		do
# 			for c in ${complaint_ratio[@]}
# 				do
# 					for d in ${deletion_factor[@]}
# 						do
# 							python LFRepair.py -U 50 -t ${c}  -e experiment_diff_num_lfs_611 -n ${n} -r information_gain -f sysgen -D ${d}
# 							echo "python LFRepair.py -U 50 -t ${c}  -e experiment_diff_num_lfs_611 -n ${n} -r information_gain -f sysgen -D ${d}"
# 							python LFRepair.py -U 50 -t ${c}  -e experiment_diff_num_lfs_611 -n ${n} -r naive -f sysgen -D ${d}
# 							echo "python LFRepair.py -U 50 -t ${c}  -e experiment_diff_num_lfs_611 -n ${n} -r naive -f sysgen -D ${d}"
# 						done
# 				done
# 		done
# 	x=$(( $x+1 ))
# done


# x=1
# while [ $x -le $non_brute_iterations ]
# do
# 	for s in ${sizes[@]}
# 		do
# 			for c in ${complaint_ratio[@]}
# 				do
# 					for d in ${deletion_factor[@]}
# 						do
# 							python LFRepair.py -U ${s} -t ${c} -e exp_vary_user_input_and_comp_ratio611 -r information_gain -f intro -D ${d}
# 							echo "python LFRepair.py -U ${s} -t ${c} -e exp_vary_user_input_and_comp_ratio611 -r information_gain -f intro -D ${d}"
# 							python LFRepair.py -U ${s} -t ${c} -e exp_vary_user_input_and_comp_ratio611 -r naive -f intro -D ${d}
# 							echo "python LFRepair.py -U ${s} -t ${c} -e exp_vary_user_input_and_comp_ratio611 -r naive -f intro -D ${d}"
# 						done
# 				done
# 		done	
# 	x=$(( $x+1 ))
# done






# filename="seed_file_613.txt" 

# x=1
# declare -i rseed
# while [ $x -le $non_brute_iterations ]
# do
# 	for s in ${sizes[@]}
# 		do
# 			for c in ${complaint_ratio[@]}
# 				do
# 					for d in ${deletion_factor[@]}
# 						do
# 							rseed=$(shuf -i 1-1000 -n 1)
# 							echo "rand= ${rseed}"
# 							echo -e "LFRepair.py -U ${s} -t ${c} -e exp_613_repeated -r information_gain -f intro -D ${d} -k true -K picked_funcs_613 -R true -s ${rseed} -S seed_file_613.txt" >> "$filename"
# 							python LFRepair.py -U ${s} -t ${c} -e exp_613_repeated -r information_gain  -D ${d} -k true -K picked_funcs_613 -R true -s ${rseed} -S seed_file_613.txt
# 							echo -e "LFRepair.py -U ${s} -t ${c} -e exp_613_repeated -r naive -f intro -D ${d} -k true -K picked_funcs_613 -R true -s ${rseed} -S seed_file_613.txt" >> "$filename"
# 							python LFRepair.py -U ${s} -t ${c} -e exp_613_repeated -r naive  -D ${d} -k true -K picked_funcs_613 -R true -s ${rseed} -S seed_file_613.txt
# 						done
# 				done
# 		done	
# 	x=$(( $x+1 ))
# done



# filename="seed_file_zero_deletes.txt" 

# x=1
# declare -i rseed
# while [ $x -le $non_brute_iterations ]
# do
# 	for s in ${sizes[@]}
# 		do
# 			rseed=$(shuf -i 1-1000 -n 1)
# 			echo "rand= ${rseed}"
# 			echo -e "LFRepair.py -U ${s} -t 0.5 -e exp_zero_deletion -r information_gain -D 0 -k true -K picked_funcs_613 -R true -s ${rseed} -S seed_file_zero_deletes.txt" >> "$filename"
# 			python LFRepair.py -U ${s} -t 0.5 -e exp_zero_deletion -r information_gain  -D 0 -k true -K picked_funcs_613 -R true -s ${rseed} -S seed_file_zero_deletes.txt
# 			echo -e "LFRepair.py -U ${s} -t 0.5 -e exp_zero_deletion -r naive -D 0 -k true -K picked_funcs_613 -R true -s ${rseed} -S seed_file_zero_deletes.txt" >> "$filename"
# 			python LFRepair.py -U ${s} -t 0.5 -e exp_zero_deletion -r naive  -D 0 -k true -K picked_funcs_613 -R true -s ${rseed} -S seed_file_zero_deletes.txt
# 		done	
# 	x=$(( $x+1 ))
# done


# filename="seed_file_compare_good_vs_bad_funcs_620.txt" 

# x=1
# declare -i rseed
# while [ $x -le $non_brute_iterations ]
# do
# 	for s in ${sizes[@]}
# 		do
# 			rseed=$(shuf -i 1-1000 -n 1)
# 			echo "rand= ${rseed}"
# 			echo -e "LFRepair.py -U ${s} -t 0.5 -e exp_zero_deletion -r information_gain -D 0.5 -k true -K pickled_funcs_620 -R true -s ${rseed} -S seed_file_compare_good_vs_bad_funcs_620.txt" >> "$filename"
# 			python LFRepair.py -U ${s} -t 0.5 -e bad_funcs_results_620 -r information_gain  -D 0.5 -k true -K pickled_funcs_620 -R true -s ${rseed} -S seed_file_compare_good_vs_bad_funcs_620.txt
# 			echo -e "LFRepair.py -U ${s} -t 0.5 -e exp_zero_deletion -r naive -D 0.5 -k true -K pickled_funcs_620 -R true -s ${rseed} -S seed_file_compare_good_vs_bad_funcs_620.txt" >> "$filename"
# 			python LFRepair.py -U ${s} -t 0.5 -e bad_funcs_results_620 -r naive  -D 0.5 -k true -K pickled_funcs_620 -R true -s ${rseed} -S seed_file_compare_good_vs_bad_funcs_620.txt
# 			echo -e "LFRepair.py -U ${s} -t 0.5 -e exp_zero_deletion -r information_gain -D 0 -k true -K picked_funcs_613 -R true -s ${rseed} -S seed_file_compare_good_vs_bad_funcs_620.txt" >> "$filename"
# 			python LFRepair.py -U ${s} -t 0.5 -e good_funcs_results_620 -r information_gain  -D 0 -k true -K picked_funcs_613 -R true -s ${rseed} -S seed_file_compare_good_vs_bad_funcs_620.txt
# 			echo -e "LFRepair.py -U ${s} -t 0.5 -e exp_zero_deletion -r naive -D 0 -k true -K picked_funcs_613 -R true -s ${rseed} -S seed_file_compare_good_vs_bad_funcs_620.txt" >> "$filename"
# 			python LFRepair.py -U ${s} -t 0.5 -e good_funcs_results_620 -r naive  -D 0 -k true -K picked_funcs_613 -R true -s ${rseed} -S seed_file_compare_good_vs_bad_funcs_620.txt
# 		done	
# 	x=$(( $x+1 ))
# done

# pickled_funcs_620.pkl



# filename="seed_file_710.txt" 

# x=1
# declare -i rseed
# while [ $x -le $non_brute_iterations ]
# do
# 	for s in ${sizes[@]}
# 		do
# 			for c in ${complaint_ratio[@]}
# 				do
# 					for d in ${deletion_factor[@]}
# 						do
# 							rseed=$(shuf -i 1-1000 -n 1)
# 							echo "rand= ${rseed}"
# 							echo -e "LFRepair.py -U ${s} -t ${c} -e exp_710_repeated -r information_gain -f intro -D ${d} -k true -K picked_funcs_613 -R true -s ${rseed} -S seed_file_710.txt" >> "$filename"
# 							python LFRepair.py -U ${s} -t ${c} -e exp_710_repeated -r information_gain  -D ${d} -k true -K picked_funcs_613 -R true -s ${rseed} -S seed_file_710.txt
# 							echo -e "LFRepair.py -U ${s} -t ${c} -e exp_710_repeated -r naive -f intro -D ${d} -k true -K picked_funcs_613 -R true -s ${rseed} -S seed_file_710.txt" >> "$filename"
# 							python LFRepair.py -U ${s} -t ${c} -e exp_710_repeated -r naive  -D ${d} -k true -K picked_funcs_613 -R true -s ${rseed} -S seed_file_710.txt
# 						done
# 				done
# 		done	
# 	x=$(( $x+1 ))
# done


# filename="seed_file_725.txt" 

# x=1
# declare -i rseed
# while [ $x -le $non_brute_iterations ]
# do
# 	for s in ${sizes[@]}
# 		do
# 			for c in ${complaint_ratio[@]}
# 				do
# 					for d in ${deletion_factor[@]}
# 						do
# 							rseed=$(shuf -i 1-1000 -n 1)
# 							echo "rand= ${rseed}"
# 							echo -e "LFRepair.py -U ${s} -t ${c} -e exp_725_repeated -r information_gain -f intro -D ${d} -k true -K pickled_funcs_720 -R true -s ${rseed} -S seed_file_725.txt" >> "$filename"
# 							python LFRepair.py -U ${s} -t ${c} -e exp_725_repeated -r information_gain  -D ${d} -k true -K pickled_funcs_720 -R true -s ${rseed} -S seed_file_725.txt
# 							echo -e "LFRepair.py -U ${s} -t ${c} -e exp_725_repeated -r naive -f intro -D ${d} -k true -K pickled_funcs_720 -R true -s ${rseed} -S seed_file_725.txt" >> "$filename"
# 							python LFRepair.py -U ${s} -t ${c} -e exp_725_repeated -r naive  -D ${d} -k true -K pickled_funcs_720 -R true -s ${rseed} -S seed_file_725.txt
# 						done
# 				done
# 		done	
# 	x=$(( $x+1 ))
# done



# filename="seed_file_726.txt" 

# x=1
# declare -i rseed
# while [ $x -le $non_brute_iterations ]
# do
# 	for s in ${sizes[@]}
# 		do
# 			for c in ${complaint_ratio[@]}
# 				do
# 					for d in ${deletion_factor[@]}
# 						do
# 							rseed=$(shuf -i 1-1000 -n 1)
# 							echo "rand= ${rseed}"
# 							echo -e "LFRepair.py -U ${s} -t ${c} -e exp_726_repeated -r information_gain -f intro -D ${d} -k true -K pickled_funcs_720 -R true -s ${rseed} -S seed_file_726.txt" >> "$filename"
# 							python LFRepair.py -U ${s} -t ${c} -e exp_726_repeated -r information_gain  -D ${d} -k true -K pickled_funcs_720 -R true -s ${rseed} -S seed_file_726.txt
# 							echo -e "LFRepair.py -U ${s} -t ${c} -e exp_726_repeated -r naive -f intro -D ${d} -k true -K pickled_funcs_720 -R true -s ${rseed} -S seed_file_726.txt" >> "$filename"
# 							python LFRepair.py -U ${s} -t ${c} -e exp_726_repeated -r naive  -D ${d} -k true -K pickled_funcs_720 -R true -s ${rseed} -S seed_file_726.txt
# 						done
# 				done
# 		done	
# 	x=$(( $x+1 ))
# done


# filename="seed_file_727.txt" 

# x=1
# declare -i rseed
# while [ $x -le $non_brute_iterations ]
# do
# 	for s in ${sizes[@]}
# 		do
# 			for c in ${complaint_ratio[@]}
# 				do
# 					for d in ${deletion_factor[@]}
# 						do
# 							rseed=$(shuf -i 1-1000 -n 1)
# 							echo "rand= ${rseed}"
# 							echo -e "LFRepair.py -U ${s} -t ${c} -e exp_727_repeated_320 -r information_gain -f intro -D ${d} -k true -K pickled_funcs_720 -R true -s ${rseed} -S seed_file_727.txt" >> "$filename"
# 							python LFRepair.py -U ${s} -t ${c} -e exp_727_repeated_320 -r information_gain  -D ${d} -k true -K pickled_funcs_720 -R true -s ${rseed} -S seed_file_727.txt
# 							echo -e "LFRepair.py -U ${s} -t ${c} -e exp_727_repeated_320 -r naive -f intro -D ${d} -k true -K pickled_funcs_720 -R true -s ${rseed} -S seed_file_727.txt" >> "$filename"
# 							python LFRepair.py -U ${s} -t ${c} -e exp_727_repeated_320 -r naive  -D ${d} -k true -K pickled_funcs_720 -R true -s ${rseed} -S seed_file_727.txt
# 						done
# 				done
# 		done	
# 	x=$(( $x+1 ))
# done



# filename="seed_file_804.txt" 

# x=1
# declare -i rseed
# while [ $x -le $brute_iterations ]
# do
# 	for s in ${brute_sizes[@]}
# 		do
# 			for c in ${complaint_ratio[@]}
# 				do
# 					for d in ${deletion_factor[@]}
# 						do
# 							rseed=$(shuf -i 1-1000 -n 1)
# 							echo "rand= ${rseed}"
# 							echo -e "LFRepair.py -U ${s} -t ${c} -e exp_804_all_methods -r brute_force -f intro -D ${d} -k true -K pickled_funcs_720 -R true -s ${rseed} -S seed_file_804.txt" >> "$filename"
# 							python LFRepair.py -U ${s} -t ${c} -e exp_804_all_methods -r brute_force  -D ${d} -k true -K pickled_funcs_720 -R true -s ${rseed} -S seed_file_804.txt
# 							echo -e "LFRepair.py -U ${s} -t ${c} -e exp_727_repeated_320 -r information_gain -f intro -D ${d} -k true -K pickled_funcs_720 -R true -s ${rseed} -S seed_file_804.txt" >> "$filename"
# 							python LFRepair.py -U ${s} -t ${c} -e exp_804_all_methods -r information_gain  -D ${d} -k true -K pickled_funcs_720 -R true -s ${rseed} -S seed_file_804.txt
# 							echo -e "LFRepair.py -U ${s} -t ${c} -e exp_727_repeated_320 -r naive -f intro -D ${d} -k true -K pickled_funcs_720 -R true -s ${rseed} -S seed_file_804.txt" >> "$filename"
# 							python LFRepair.py -U ${s} -t ${c} -e exp_804_all_methods -r naive  -D ${d} -k true -K pickled_funcs_720 -R true -s ${rseed} -S seed_file_804.txt
# 						done
# 				done
# 		done	
# 	x=$(( $x+1 ))
# done

# x=1
# declare -i rseed
# while [ $x -le $non_brute_iterations ]
# do
# 	for s in ${sizes[@]}
# 		do
# 			for c in ${complaint_ratio[@]}
# 				do
# 					for d in ${deletion_factor[@]}
# 						do
# 							rseed=$(shuf -i 1-1000 -n 1)
# 							echo "rand= ${rseed}"
# 							echo -e "LFRepair.py -U ${s} -t ${c} -e exp_727_repeated_320 -r information_gain -f intro -D ${d} -k true -K pickled_funcs_720 -R true -s ${rseed} -S seed_file_804.txt" >> "$filename"
# 							python LFRepair.py -U ${s} -t ${c} -e exp_804_all_methods -r information_gain  -D ${d} -k true -K pickled_funcs_720 -R true -s ${rseed} -S seed_file_804.txt
# 							echo -e "LFRepair.py -U ${s} -t ${c} -e exp_727_repeated_320 -r naive -f intro -D ${d} -k true -K pickled_funcs_720 -R true -s ${rseed} -S seed_file_804.txt" >> "$filename"
# 							python LFRepair.py -U ${s} -t ${c} -e exp_804_all_methods -r naive  -D ${d} -k true -K pickled_funcs_720 -R true -s ${rseed} -S seed_file_804.txt
# 						done
# 				done
# 		done	
# 	x=$(( $x+1 ))
# done



filename="seed_file_808_1500.txt" 

x=1
declare -i rseed
while [ $x -le $non_brute_iterations ]
do
	for s in ${sizes[@]}
		do
			for c in ${complaint_ratio[@]}
				do
					for d in ${deletion_factor[@]}
						do
							rseed=$(shuf -i 1-1000 -n 1)
							echo "rand= ${rseed}"
							echo -e "LFRepair.py -U ${s} -t ${c} -e exp808_repeat1500 -r information_gain -f intro -D ${d} -k true -K pickled_funcs_720 -R true -s ${rseed} -S seed_file_808_1500.txt" >> "$filename"
							python LFRepair.py -U ${s} -t ${c} -e exp808_repeat1500 -r information_gain  -D ${d} -k true -K pickled_funcs_720 -R true -s ${rseed} -S seed_file_808_1500.txt
							echo -e "LFRepair.py -U ${s} -t ${c} -e exp808_repeat1500 -r naive -f intro -D ${d} -k true -K pickled_funcs_720 -R true -s ${rseed} -S seed_file_808_1500.txt" >> "$filename"
							python LFRepair.py -U ${s} -t ${c} -e exp808_repeat1500 -r naive  -D ${d} -k true -K pickled_funcs_720 -R true -s ${rseed} -S seed_file_808_1500.txt
						done
				done
		done	
	x=$(( $x+1 ))
done