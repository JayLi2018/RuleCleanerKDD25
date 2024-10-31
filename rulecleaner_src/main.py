import psycopg2
import math
import time 
import sys
import os
import datetime
import pickle
sys.path.append(os.path.join(os.getcwd(), ".."))
from lf_solver import  lf_constraint_solve_no_new_lf
from utils import (construct_input_df_to_solver,
                   create_solver_input_df_copies,
                   select_user_input

)
from rulecleaner_src.LFRepair import fix_rules_with_solver_input
from rulecleaner_src.utils import run_snorkel_with_funcs

def main(user_input_size,
         lf_acc_thresh,
         instance_acc_thresh,
         min_non_abstain_thresh,
        dataset_name,
        random_state,
        funcs_dictionary,
       instance_acc_on_valid,
       use_non_abstain,
       pickle_result_file_name_prefix):
    
    for arg_name, value in locals().items():
        print(f"{arg_name} = {value}")
    
    run_times = ['snorkel_first_run','snorkel_run_after_fix', 'solver_runtime','repair_time']
    runtime_dict = {r:0 for r in run_times}

    gen_input_tree_rules_func = funcs_dictionary[dataset_name]
    
    conn = psycopg2.connect(dbname='label', user='postgres')
    
    user_complaint_size = math.floor(user_input_size * 0.5)
    user_confirm_size = user_input_size - user_complaint_size
         
    treerules = gen_input_tree_rules_func()
    
    funcs = [f.gen_label_rule() for f in treerules]
    
    first_snorkel_run_start = time.time()
    df_sentences_filtered, correct_preds_by_snorkel, wrong_preds_by_snorkel, filtered_vectors, correct_predictions, incorrect_predictions, global_accuracy, global_accuracy_on_valid =run_snorkel_with_funcs(dataset_name=dataset_name, funcs=funcs, conn=conn)
    first_snorkel_run_end = time.time()
    first_snorkel_run_time = first_snorkel_run_end - first_snorkel_run_start
    runtime_dict['snorkel_first_run'] = first_snorkel_run_time

    user_vecs, gts, user_input_df = select_user_input(user_confirm_size, user_complaint_size, random_state,
                      filtered_vectors,correct_preds_by_snorkel,
                      wrong_preds_by_snorkel, correct_predictions, incorrect_predictions)

        
    combined_df = construct_input_df_to_solver(user_vecs, gts)
    
    solver_runtime_start = time.time()
    
    res_df, res_flip_cost = lf_constraint_solve_no_new_lf(df=combined_df, 
                lf_acc_thresh=lf_acc_thresh,
                instance_acc_thresh=instance_acc_thresh,
                min_non_abstain_thresh=min_non_abstain_thresh,      
                expected_label_col='expected_label',
                instance_acc_on_valid=instance_acc_on_valid,
               use_non_abstain=use_non_abstain)
    
    
    solver_runtime_end = time.time()
    solver_runtime = solver_runtime_end - solver_runtime_start
    runtime_dict['solver_runtime'] = solver_runtime
    
    fix_book_keeping_dict = {'original_'+str(k.id):{'rule':k, 'deleted':False,
                       'pre_fix_size':k.size, 
                       'after_fix_size':k.size, 
                       'pre-deleted': False} for k in treerules}
    
    lfs_witan = [l for l in list(combined_df) if ('nlf' not in l and l!='expected_label')]
#     lfs_manual_added =  [x for x in inclusion_dict if inclusion_dict[x]==1]
#     lf_names_after_fix = lfs_witan +lfs_manual_added

    df_copies = create_solver_input_df_copies(lf_names_after_fix=lfs_witan,
                                     user_input_df=user_input_df,
                                     res_df=res_df)
    df_list = list(df_copies.values())

    book_keeping_dict_list = list(fix_book_keeping_dict)
    
    for i in range(len(df_list)):
        fix_book_keeping_dict[book_keeping_dict_list[i]]['user_input'] = df_list[i]
        fix_book_keeping_dict[book_keeping_dict_list[i]]['user_input']['id'] = \
        fix_book_keeping_dict[book_keeping_dict_list[i]]['user_input'].reset_index().index
    
    
    repair_alghorithm_start = time.time()
    fix_rules_with_solver_input(fix_book_keeping_dict=fix_book_keeping_dict)
    repair_alghorithm_end = time.time()
    repair_alghorithm_time = repair_alghorithm_end - repair_alghorithm_start
    runtime_dict['repair_time'] = repair_alghorithm_time
    
    new_trees = [x['rule'] for x in fix_book_keeping_dict.values()]
    funcs_after_fix = [f.gen_label_rule() for f in new_trees]

    snorkel_run_after_fix_start = time.time()
    new_df_sentences_filtered, correct_preds_by_snorkel, wrong_preds_by_snorkel, filtered_vectors, correct_predictions, incorrect_predictions, new_global_accuracy, new_global_accuracy_on_valid =run_snorkel_with_funcs(dataset_name=dataset_name, funcs=funcs_after_fix, conn=conn) 
    snorkel_run_after_fix_end = time.time()
    snorkel_run_after_fix_time = snorkel_run_after_fix_end - snorkel_run_after_fix_start
    runtime_dict['snorkel_run_after_fix'] = snorkel_run_after_fix_time
    
    complaints = user_input_df[user_input_df['expected_label']!=user_input_df['model_pred']]
    complant_ids = complaints['cid'].to_list()
    confirms = user_input_df[user_input_df['expected_label']==user_input_df['model_pred']]
    confirm_ids = confirms['cid'].to_list()
    
    df_confirms_after_fix = new_df_sentences_filtered[(new_df_sentences_filtered['cid'].isin(confirm_ids))]
    df_complaints_after_fix = new_df_sentences_filtered[(new_df_sentences_filtered['cid'].isin(complant_ids))]
    
    confirm_preserv_rate = len(df_confirms_after_fix[df_confirms_after_fix['expected_label']==df_confirms_after_fix['model_pred']])/len(df_confirms_after_fix)
    complain_fix_rate = len(df_complaints_after_fix[df_complaints_after_fix['expected_label']==df_complaints_after_fix['model_pred']])/len(df_complaints_after_fix)
    
    ret = {'before_fix_global_accuracy':global_accuracy,
           'user_input_size':user_input_size,
           'lf_acc_thresh':lf_acc_thresh,
           'instance_acc_thresh':instance_acc_thresh,
           'min_non_abstain_thresh':min_non_abstain_thresh,
           'dataset_name':dataset_name,
           'random_state':random_state,
           'confirm_prev_rate':confirm_preserv_rate,
           'complain_fix_rate':complain_fix_rate,
           'new_global_accuracy':new_global_accuracy,
           'global_accuracy_on_valid_data': global_accuracy_on_valid,
          'new_global_accuracy_on_valid': new_global_accuracy_on_valid,
           'valid_global_data_size': len(df_sentences_filtered),
           'new_valid_global_data_size': len(new_df_sentences_filtered),
           'runtimes': runtime_dict,
           'optimal_objective_value': res_flip_cost,
           }
    
    res_to_save = {'summary': ret, 'fix_details': fix_book_keeping_dict}

    directory = os.path.dirname(pickle_result_file_name_prefix)

    if directory: 
        os.makedirs(directory, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(f'{pickle_result_file_name_prefix}-{dataset_name}_sample_params_{user_input_size}-{lf_acc_thresh}-{instance_acc_thresh}-{min_non_abstain_thresh}-{random_state}-{timestamp}.pkl', 'wb') as resf:
        pickle.dump(res_to_save, resf)
    
    conn.close()
    
    
    return fix_book_keeping_dict, res_df, gts, user_input_df, df_sentences_filtered, ret