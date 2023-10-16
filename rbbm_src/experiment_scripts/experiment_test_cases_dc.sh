#!/bin/bash

# # brute_sizes=(40 80)
# sizes=(320)

# # sizes_brute=(5 10 20 30)
# # num_funcs=(10 20 30 40 50)
# complaint_ratio=(0.1 0.3 0.5 0.7 0.9)
# # deletion_factor=(0.0 0.3 0.5 0.7)
# deletion_factor=(0.0)

# declare -i brute_iterations=5
# declare -i non_brute_iterations=1500

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
python main.py -U dc -e experiment_results_folders/dc_test_case1004 -R true -x 123 -X test_cases_seed.txt -T 0 -s 40 -r 0.5 -G optimal -D 0 -t tax -C /home/opc/chenjie/RBBM/rbbm_src/muse/data/mas/tax_rules.txt \
  -F /home/opc/chenjie/RBBM/rbbm_src/user_desired_dcs.txt -S ind -l critical 

python main.py -U dc -e experiment_results_folders/dc_test_case1004 -R true -x 123 -X test_cases_seed.txt -T 0 -s 40 -r 0.5 -G information_gain -D 0 -t tax -C /home/opc/chenjie/RBBM/rbbm_src/muse/data/mas/tax_rules.txt \
   -F /home/opc/chenjie/RBBM/rbbm_src/user_desired_dcs.txt -S ind -l critical 

python main.py -U dc -e experiment_results_folders/dc_test_case1004 -R true -x 123 -X test_cases_seed.txt -T 0 -s 40 -r 0.5 -G naive -D 0 -t tax -C /home/opc/chenjie/RBBM/rbbm_src/muse/data/mas/tax_rules.txt \
-F /home/opc/chenjie/RBBM/rbbm_src/user_desired_dcs.txt -S ind -l critical 

