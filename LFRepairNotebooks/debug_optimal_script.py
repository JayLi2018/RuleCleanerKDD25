#!/usr/bin/env python
# coding: utf-8

# In[1]:


# from rbbm_src.labelling_func_src.src.utils import lf_constraint_solve
from rbbm_src.labelling_func_src.src.lfs_tree import keyword_labelling_func_builder
from rbbm_src.labelling_func_src.src.TreeRules import SPAM, HAM, ABSTAIN, PredicateNode
from rbbm_src.labelling_func_src.src.LFRepair import populate_violations, fix_rules_with_solver_input
from rbbm_src.labelling_func_src.src.classes import clean_text

import re
import psycopg2
import pandas as pd
from snorkel.labeling import (
	LabelingFunction, 
	labeling_function, 
	PandasLFApplier, 
	LFAnalysis,
	filter_unlabeled_dataframe
	)
from snorkel.labeling.model import MajorityLabelVoter, LabelModel
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.patches as mpatches
import pulp
from sklearn.metrics import accuracy_score, classification_report

from rbbm_src.labelling_func_src.src.KeyWordRuleMiner import KeyWordRuleMiner 
# sample user confirmation and complaints
import random
from collections import deque
import numpy as np
import pickle
import pydot
from IPython.display import Image, display 

import datetime


# In[2]:


from collections import defaultdict


# In[3]:


def run_snorkel_with_funcs(dataset_name, funcs, conn):
    
    sentences_df=pd.read_sql(f'SELECT * FROM {dataset_name}', conn)
    sentences_df = sentences_df.rename(columns={"class": "expected_label", "content": "old_text"})
    sentences_df['text'] = sentences_df['old_text'].apply(lambda s: clean_text(s))
    sentences_df = sentences_df[~sentences_df['text'].isna()]
    applier = PandasLFApplier(lfs=funcs)
    initial_vectors = applier.apply(df=sentences_df, progress_bar=False)
    model = LabelModel(cardinality=2, verbose=True, device='cpu')
    model.fit(L_train=initial_vectors, n_epochs=500, log_freq=100, seed=123)
    probs_test= model.predict_proba(L=initial_vectors)
    df_sentences_filtered, probs_test_filtered, filtered_vectors, df_no_signal  = filter_unlabeled_dataframe(
        X=sentences_df, y=probs_test, L=initial_vectors
    )	

    df_sentences_filtered = df_sentences_filtered.reset_index(drop=True)
    prob_diffs = [abs(t[0]-t[1]) for t in probs_test_filtered]
    prob_diffs_tuples = [(t[0],t[1]) for t in probs_test_filtered]
    df_sentences_filtered['model_pred_diff'] = pd.Series(prob_diffs)
    df_sentences_filtered['model_pred_prob_tuple'] = pd.Series(prob_diffs_tuples)
    df_sentences_filtered['model_pred'] = pd.Series(model.predict(L=filtered_vectors))

    wrong_preds = df_sentences_filtered[(df_sentences_filtered['expected_label']!=df_sentences_filtered['model_pred'])]
    # df_sentences_filtered.to_csv('predictions_shakira.csv', index=False)
    # logger.critical(wrong_preds)
    global_accuray_on_valid=(len(df_sentences_filtered)-len(wrong_preds))/len(df_sentences_filtered)

    print(f"""
        out of {len(sentences_df)} sentences, {len(df_sentences_filtered)} actually got at least one signal to \n
        make prediction. Out of all the valid predictions, we have {len(wrong_preds)} wrong predictions, \n
        accuracy = {(len(df_sentences_filtered)-len(wrong_preds))/len(df_sentences_filtered)} 
    """)
    
    global_accuracy = (len(df_sentences_filtered)-len(wrong_preds))/len(sentences_df)
    
    
    ground_truth = df_sentences_filtered['expected_label']
    snorkel_predictions = df_sentences_filtered['model_pred']
    snorkel_probs = df_sentences_filtered['model_pred_diff']
    df_sentences_filtered['vectors'] = pd.Series([",".join(map(str, t)) for t in filtered_vectors])
    correct_predictions = (snorkel_predictions == ground_truth)
    incorrect_predictions = (snorkel_predictions != ground_truth)
    correct_preds_by_snorkel = df_sentences_filtered[correct_predictions].reset_index(drop=True)
    wrong_preds_by_snorkel = df_sentences_filtered[incorrect_predictions].reset_index(drop=True)
    
    return df_sentences_filtered, correct_preds_by_snorkel, wrong_preds_by_snorkel, filtered_vectors, correct_predictions, incorrect_predictions, global_accuracy, global_accuray_on_valid 


# In[4]:


def select_user_input(user_confirm_size,
                     user_complaint_size,
                     random_state,
                     filtered_vectors,
                     correct_preds_by_snorkel,
                     wrong_preds_by_snorkel,
                      correct_predictions,
                      incorrect_predictions ):

    user_confirm_df = correct_preds_by_snorkel.sample(n=user_confirm_size, random_state=random_state)
    user_complaints_df = wrong_preds_by_snorkel.sample(n=user_complaint_size, random_state=random_state)
    
    random_confirm_indices = user_confirm_df.index
    random_complaints_indices = user_complaints_df.index
    random_user_confirms_vecs = filtered_vectors[correct_predictions][random_confirm_indices]
    random_user_complaints_vecs = filtered_vectors[incorrect_predictions][random_complaints_indices]
    user_input_df = pd.concat([user_confirm_df, user_complaints_df])
    gts = user_input_df['expected_label'].reset_index(drop=True)
    user_vecs = np.vstack((random_user_confirms_vecs, random_user_complaints_vecs))
    
    return user_vecs, gts, user_input_df


# In[5]:


def gather_user_input_signals_on_rules(tree_rules, user_input):
    leaf_nodes = []
    
    for atui in tree_rules:
        rids = set([])
        for i, c in user_input.iterrows():
            leaf_node_with_complaints = populate_violations(atui, c)
            if(leaf_node_with_complaints.number not in rids):
                rids.add(leaf_node_with_complaints.number)
                leaf_nodes.append(leaf_node_with_complaints)
            
    uinput_unsatisfied_counts = defaultdict(int)
    
    for ln in leaf_nodes:
        if(ln.label==ABSTAIN):
            for l in [SPAM, HAM]:
                for u in ln.pairs[l]:
                    uinput_unsatisfied_counts[u['cid']]+=1
    
    return uinput_unsatisfied_counts


# In[6]:


def gather_used_keywords(tree_rules):
    
    used_keywords = []
    
    for atui in tree_rules:
        rids = set([])
        queue = deque([atui.root])
        while(queue):
            cur_node = queue.popleft()
            if(isinstance(cur_node, PredicateNode)):
                used_keywords.extend(cur_node.pred.keywords)
            if(cur_node.left):
                queue.append(cur_node.left)
            if(cur_node.right):
                queue.append(cur_node.right)
    
    return used_keywords

#     for i, c in sorted_df.iterrows():
#         leaf_node_with_complaints = populate_violations(atui, c)
#         if(leaf_node_with_complaints.number not in rids):
#             rids.add(leaf_node_with_complaints.number)
#             leaf_nodes.append(leaf_node_with_complaints)


# In[7]:


def apply_new_lfs_to_df(new_funcs, user_input_df):
    new_rules_applier = PandasLFApplier(lfs=new_funcs)
    new_rules_vector = new_rules_applier.apply(df=user_input_df, progress_bar=False)
    
    return new_rules_vector


# In[8]:


def construct_input_df_to_solver(user_vecs, gts):
    
#     df_new_vectors = pd.DataFrame(new_rules_vector, columns=[f'nlf_{i+1}' for i in range(new_rules_vector.shape[1])])
    df_user_vectors = pd.DataFrame(user_vecs, columns=[f'lf_{i+1}' for i in range(user_vecs.shape[1])])
    combined_df= pd.concat([df_user_vectors, gts], axis=1)
    
    return combined_df


# In[9]:


# def lf_constraint_solve(df, lf_acc_thresh=0.5, 
#                         instance_acc_thresh=0.5,
#                         min_non_abstain_thresh=0.8,
#                         nlf_prefix='nlf_',
#                         expected_label_col='expected_label',
#                         new_lf_weight=0.1):
    
#     # Problem initialization
#     prob = pulp.LpProblem("Label_Flip_Minimization", pulp.LpMinimize)

#     # Parameters
# #     labeling_functions = df.columns[:-1]
#     labeling_functions = [lf_name for lf_name in df.columns if lf_name!=expected_label_col]
#     print(f"lf_acc: {lf_acc_thresh}, ins_acc:{instance_acc_thresh}, min_non_abstain_thresh")
#     print(f"labeling_functions: {labeling_functions}")
#     num_instances = len(df)
#     print(f"num_instances: {num_instances}")
#     M = 5
    
#     nlfs = [lf for lf in labeling_functions if nlf_prefix in lf]
#     print(f"nlfs: {nlfs}")
#     x_nlfs = pulp.LpVariable.dicts("x_nlf", nlfs, cat='Binary')

#     P_vars = pulp.LpVariable.dicts("P", (range(num_instances), labeling_functions), 
#                                    lowBound=-1, upBound=1, cat='Integer')
    
#     is_abstain = pulp.LpVariable.dicts("is_abstain", 
#                                (range(num_instances), labeling_functions), 
#                                cat='Binary')

#     flip_1_to_0 = pulp.LpVariable.dicts("flip_1_to_0", 
#                                         (range(num_instances), labeling_functions), cat='Binary')
#     flip_1_to_neg1 = pulp.LpVariable.dicts("flip_1_to_neg1", 
#                                            (range(num_instances), labeling_functions), cat='Binary')
#     flip_0_to_1 = pulp.LpVariable.dicts("flip_0_to_1", 
#                                         (range(num_instances), labeling_functions), cat='Binary')
#     flip_0_to_neg1 = pulp.LpVariable.dicts("flip_0_to_neg1", 
#                                            (range(num_instances), labeling_functions), cat='Binary')
#     flip_neg1_to_1 = pulp.LpVariable.dicts("flip_neg1_to_1", 
#                                            (range(num_instances), labeling_functions), cat='Binary')
#     flip_neg1_to_0 = pulp.LpVariable.dicts("flip_neg1_to_0", 
#                                            (range(num_instances), labeling_functions), cat='Binary')

#     # Binary variables to track correctness of predictions (1 if correct, 0 if not)
#     correctness_vars = pulp.LpVariable.dicts("correct", 
#                                              (range(num_instances), labeling_functions), cat='Binary')
    
#     # Create auxiliary variables to represent active nLF abstains
#     active_abstain = pulp.LpVariable.dicts("active_abstain", 
#                                            (range(num_instances), nlfs), 
#                                            cat='Binary')
    
#     correct_and_active = pulp.LpVariable.dicts("correct_and_active", 
#                                            (range(num_instances), nlfs), 
#                                            cat='Binary')


#     # Objective: Minimize the number of flips
#     flip_cost = pulp.lpSum([flip_1_to_0[i][lf] + flip_1_to_neg1[i][lf] + 
#                             flip_0_to_1[i][lf] + flip_0_to_neg1[i][lf] + 
#                             flip_neg1_to_1[i][lf] + flip_neg1_to_0[i][lf] 
#                             for i in range(num_instances) for lf in labeling_functions])

#     prob += flip_cost + pulp.lpSum([new_lf_weight * x_nlfs[lf] for lf in nlfs]), "Minimize_Flips"


#     # Mutual exclusivity
#     for i in range(num_instances):
#         for lf in labeling_functions:
#             prob += (flip_1_to_0[i][lf] + flip_1_to_neg1[i][lf] + 
#                      flip_0_to_1[i][lf] + flip_0_to_neg1[i][lf] + 
#                      flip_neg1_to_1[i][lf] + flip_neg1_to_0[i][lf]) <= 1, f"Flip_Exclusivity_{i}_{lf}"

#     for i in range(num_instances):
#         for lf in labeling_functions:
#             original_val = df.loc[i, lf]
#             if original_val == 1:
#                 prob += P_vars[i][lf] == 0 * flip_1_to_0[i][lf] + \
#                 (-1) * flip_1_to_neg1[i][lf] + 1 * (1 - flip_1_to_0[i][lf] - flip_1_to_neg1[i][lf]), f"Flip_From_1_{i}_{lf}"
                
#             elif original_val == 0:                
#                 prob += P_vars[i][lf] == 1 * flip_0_to_1[i][lf] + \
#                 (-1) * flip_0_to_neg1[i][lf] + 0 * (1 - flip_0_to_1[i][lf] - flip_0_to_neg1[i][lf]), f"Flip_From_0_{i}_{lf}"
                
#             elif original_val == -1:
#                 prob += P_vars[i][lf] == 1 * flip_neg1_to_1[i][lf] + 0 * flip_neg1_to_0[i][lf] + (-1) * (1 - flip_neg1_to_1[i][lf] - flip_neg1_to_0[i][lf]), f"Flip_From_neg1_{i}_{lf}"
    
#     for i in range(num_instances):
#         for lf in labeling_functions:
#             prob += P_vars[i][lf] >= -1 - (1 - is_abstain[i][lf]) * M, f"Abstain_LowerBound_{i}_{lf}"
#             prob += P_vars[i][lf] <= -1 + (1 - is_abstain[i][lf]) * M, f"Abstain_UpperBound_{i}_{lf}"

#             # If is_abstain[i][lf] == 0, P_vars[i][lf] can only be 0 or 1
#             prob += P_vars[i][lf] >= 0 - is_abstain[i][lf] * M, f"Non_Abstain_LowerBound_{i}_{lf}"
#             prob += P_vars[i][lf] <= 1 + is_abstain[i][lf] * M, f"Non_Abstain_UpperBound_{i}_{lf}"
    
#     # Set up the constraints for the auxiliary variables
#     for i in range(num_instances):
#         for lf in nlfs:
#             # Ensure active_abstain[i][lf] is 1 only if both is_abstain[i][lf] == 1 and x_nlfs[lf] == 1
#             prob += active_abstain[i][lf] <= is_abstain[i][lf], f"ActiveAbstain_LF_{lf}_Instance_{i}_1"
#             prob += active_abstain[i][lf] <= x_nlfs[lf], f"ActiveAbstain_LF_{lf}_Instance_{i}_2"
#             prob += active_abstain[i][lf] >= is_abstain[i][lf] + x_nlfs[lf] - 1, f"ActiveAbstain_LF_{lf}_Instance_{i}_3"

#     for i in range(num_instances):
#         for lf in nlfs:
#             # correct_and_active[i][lf] should be 1 only if both correctness_vars[i][lf] == 1 and x_nlfs[lf] == 1
#             prob += correct_and_active[i][lf] <= correctness_vars[i][lf], f"CorrectAndActive_UpperBound_1_{i}_{lf}"
#             prob += correct_and_active[i][lf] <= x_nlfs[lf], f"CorrectAndActive_UpperBound_2_{i}_{lf}"
#             prob += correct_and_active[i][lf] >= correctness_vars[i][lf] + x_nlfs[lf] - 1, f"CorrectAndActive_LowerBound_{i}_{lf}"
        
    
#     for lf in labeling_functions:
#         num_instances_abstain = pulp.lpSum([is_abstain[i][lf] for i in range(num_instances)])
#         if lf in nlfs:
#             lf_correct_predictions = pulp.lpSum([correctness_vars[i][lf] for i in range(num_instances)])
#             prob += lf_correct_predictions >= lf_acc_thresh * (num_instances-num_instances_abstain) - M * (1 - x_nlfs[lf]), f"LF_{lf}_Accuracy"
#         else:
#             lf_correct_predictions = pulp.lpSum([correctness_vars[i][lf] for i in range(num_instances)])
#             prob += lf_correct_predictions >= lf_acc_thresh * (num_instances-num_instances_abstain), f"LF_{lf}_Accuracy"



#     for i in range(num_instances):
#         for lf in nlfs:
#             # Ensure that correctness_vars[i][lf] is counted only if x_nlf[lf] = 1
#             prob += correctness_vars[i][lf] <= M * x_nlfs[lf], f"{lf}_active_{i}"
            
#         correct_predictions_per_instance = pulp.lpSum([correctness_vars[i][lf] for lf in labeling_functions if lf not in nlfs]) + \
#                                pulp.lpSum([correct_and_active[i][lf] for lf in nlfs])
#         instance_abstain_count = pulp.lpSum([is_abstain[i][lf] for lf in labeling_functions if lf not in nlfs]) + \
#                                  pulp.lpSum([active_abstain[i][lf] for lf in nlfs]) 
        
#         num_labeling_functions_used = len(labeling_functions) - len(nlfs) + pulp.lpSum(x_nlfs.values())
#         prob += correct_predictions_per_instance >= instance_acc_thresh * num_labeling_functions_used, f"Instance_{i}_Accuracy"
#         prob += instance_abstain_count <= num_labeling_functions_used *(1- min_non_abstain_thresh), f"Instance_{i}_NonAbastain"

        
#     for i in range(num_instances):
#         for lf in labeling_functions:
#             true_label = df[expected_label_col][i]
#             # Ensure that correctness_vars[i][lf] is 1 if P_vars[i][lf] equals true_label, else 0
#             prob += P_vars[i][lf] - true_label <= M * (1 - correctness_vars[i][lf]),\
#                                      f"Correctness_UpperBound_{i}_{lf}"
#             prob += true_label - P_vars[i][lf] <= M * (1 - correctness_vars[i][lf]), \
#                                      f"Correctness_LowerBound_{i}_{lf}"


#     # Solve the integer program
#     prob.solve()

#     p_vars_solution = pd.DataFrame(index=df.index, columns=labeling_functions)
#     active_abstain_df = pd.DataFrame(index=df.index, columns=labeling_functions)
#     is_abstain_df = pd.DataFrame(index=df.index, columns=labeling_functions)
    
#     for i in range(num_instances):
#         for lf in labeling_functions:
#             p_vars_solution.loc[i, lf] = int(pulp.value(P_vars[i][lf]))
    
#     correctness_solution = pd.DataFrame(index=df.index, columns=labeling_functions)
#     for i in range(num_instances):
#         for lf in labeling_functions:
#             correctness_solution.loc[i, lf] = int(pulp.value(correctness_vars[i][lf]))
    
#     x_nlfs_solution = {lf: pulp.value(x_nlfs[lf]) for lf in nlfs}
    
#     print(f"Status: {pulp.LpStatus[prob.status]}")
#     print(f"pulp.value(num_labeling_functions_used) : {pulp.value(num_labeling_functions_used)}")
    
#     for i in range(num_instances):
#         for lf in labeling_functions:
#             is_abstain_df.loc[i, lf] = int(pulp.value(is_abstain[i][lf]))
#     for i in range(num_instances):
#         for lf in nlfs:
#             active_abstain_df.loc[i, lf] = int(pulp.value(active_abstain[i][lf]))
    
#     return p_vars_solution, x_nlfs_solution, pulp, prob, active_abstain_df, is_abstain_df


# In[10]:


def lf_constraint_solve_no_new_lf(df, lf_acc_thresh=0.5, 
                        instance_acc_thresh=0.5,
                        min_non_abstain_thresh=0.8,
#                         nlf_prefix='nlf_',
                        expected_label_col='expected_label',
#                         new_lf_weight=0.1
                        instance_acc_on_valid=False,
                        use_non_abstain=True
                       ):
    
    # Problem initialization
    prob = pulp.LpProblem("Label_Flip_Minimization", pulp.LpMinimize)

    # Parameters
#     labeling_functions = df.columns[:-1]
    labeling_functions = [lf_name for lf_name in df.columns if lf_name!=expected_label_col]
    print(f"lf_acc: {lf_acc_thresh}, ins_acc:{instance_acc_thresh}, min_non_abstain_thresh:{min_non_abstain_thresh}")
    print(f"labeling_functions: {labeling_functions}")
    num_instances = len(df)
    print(f"num_instances: {num_instances}")
    M = 5
    
#     nlfs = [lf for lf in labeling_functions if nlf_prefix in lf]
#     print(f"nlfs: {nlfs}")
#     x_nlfs = pulp.LpVariable.dicts("x_nlf", nlfs, cat='Binary')

    P_vars = pulp.LpVariable.dicts("P", (range(num_instances), labeling_functions), 
                                   lowBound=-1, upBound=1, cat='Integer')
    
    is_abstain = pulp.LpVariable.dicts("is_abstain", 
                               (range(num_instances), labeling_functions), 
                               cat='Binary')

    flip_1_to_0 = pulp.LpVariable.dicts("flip_1_to_0", 
                                        (range(num_instances), labeling_functions), cat='Binary')
    flip_1_to_neg1 = pulp.LpVariable.dicts("flip_1_to_neg1", 
                                           (range(num_instances), labeling_functions), cat='Binary')
    flip_0_to_1 = pulp.LpVariable.dicts("flip_0_to_1", 
                                        (range(num_instances), labeling_functions), cat='Binary')
    flip_0_to_neg1 = pulp.LpVariable.dicts("flip_0_to_neg1", 
                                           (range(num_instances), labeling_functions), cat='Binary')
    flip_neg1_to_1 = pulp.LpVariable.dicts("flip_neg1_to_1", 
                                           (range(num_instances), labeling_functions), cat='Binary')
    flip_neg1_to_0 = pulp.LpVariable.dicts("flip_neg1_to_0", 
                                           (range(num_instances), labeling_functions), cat='Binary')

    # Binary variables to track correctness of predictions (1 if correct, 0 if not)
    correctness_vars = pulp.LpVariable.dicts("correct", 
                                             (range(num_instances), labeling_functions), cat='Binary')
    
#     # Create auxiliary variables to represent active nLF abstains
#     active_abstain = pulp.LpVariable.dicts("active_abstain", 
#                                            (range(num_instances), nlfs), 
#                                            cat='Binary')
    
#     correct_and_active = pulp.LpVariable.dicts("correct_and_active", 
#                                            (range(num_instances), nlfs), 
#                                            cat='Binary')


    # Objective: Minimize the number of flips
    flip_cost = pulp.lpSum([flip_1_to_0[i][lf] + flip_1_to_neg1[i][lf] + 
                            flip_0_to_1[i][lf] + flip_0_to_neg1[i][lf] + 
                            flip_neg1_to_1[i][lf] + flip_neg1_to_0[i][lf] 
                            for i in range(num_instances) for lf in labeling_functions])

#     prob += flip_cost + pulp.lpSum([new_lf_weight * x_nlfs[lf] for lf in nlfs]), "Minimize_Flips"
    prob += flip_cost, "Minimize_Flips"


    # Mutual exclusivity
    for i in range(num_instances):
        for lf in labeling_functions:
            prob += (flip_1_to_0[i][lf] + flip_1_to_neg1[i][lf] + 
                     flip_0_to_1[i][lf] + flip_0_to_neg1[i][lf] + 
                     flip_neg1_to_1[i][lf] + flip_neg1_to_0[i][lf]) <= 1, f"Flip_Exclusivity_{i}_{lf}"

    for i in range(num_instances):
        for lf in labeling_functions:
            original_val = df.loc[i, lf]
            if original_val == 1:
                prob += P_vars[i][lf] == 0 * flip_1_to_0[i][lf] + \
                (-1) * flip_1_to_neg1[i][lf] + 1 * (1 - flip_1_to_0[i][lf] - flip_1_to_neg1[i][lf]), f"Flip_From_1_{i}_{lf}"
                
            elif original_val == 0:                
                prob += P_vars[i][lf] == 1 * flip_0_to_1[i][lf] + \
                (-1) * flip_0_to_neg1[i][lf] + 0 * (1 - flip_0_to_1[i][lf] - flip_0_to_neg1[i][lf]), f"Flip_From_0_{i}_{lf}"
                
            elif original_val == -1:
                prob += P_vars[i][lf] == 1 * flip_neg1_to_1[i][lf] + 0 * flip_neg1_to_0[i][lf] + (-1) * (1 - flip_neg1_to_1[i][lf] - flip_neg1_to_0[i][lf]), f"Flip_From_neg1_{i}_{lf}"
    
    for i in range(num_instances):
        for lf in labeling_functions:
            prob += P_vars[i][lf] >= -1 - (1 - is_abstain[i][lf]) * M, f"Abstain_LowerBound_{i}_{lf}"
            prob += P_vars[i][lf] <= -1 + (1 - is_abstain[i][lf]) * M, f"Abstain_UpperBound_{i}_{lf}"

            # If is_abstain[i][lf] == 0, P_vars[i][lf] can only be 0 or 1
            prob += P_vars[i][lf] >= 0 - is_abstain[i][lf] * M, f"Non_Abstain_LowerBound_{i}_{lf}"
            prob += P_vars[i][lf] <= 1 + is_abstain[i][lf] * M, f"Non_Abstain_UpperBound_{i}_{lf}"
    
    # Set up the constraints for the auxiliary variables
#     for i in range(num_instances):
#         for lf in nlfs:
#             # Ensure active_abstain[i][lf] is 1 only if both is_abstain[i][lf] == 1 and x_nlfs[lf] == 1
#             prob += active_abstain[i][lf] <= is_abstain[i][lf], f"ActiveAbstain_LF_{lf}_Instance_{i}_1"
#             prob += active_abstain[i][lf] <= x_nlfs[lf], f"ActiveAbstain_LF_{lf}_Instance_{i}_2"
#             prob += active_abstain[i][lf] >= is_abstain[i][lf] + x_nlfs[lf] - 1, f"ActiveAbstain_LF_{lf}_Instance_{i}_3"

#     for i in range(num_instances):
#         for lf in nlfs:
#             # correct_and_active[i][lf] should be 1 only if both correctness_vars[i][lf] == 1 and x_nlfs[lf] == 1
#             prob += correct_and_active[i][lf] <= correctness_vars[i][lf], f"CorrectAndActive_UpperBound_1_{i}_{lf}"
#             prob += correct_and_active[i][lf] <= x_nlfs[lf], f"CorrectAndActive_UpperBound_2_{i}_{lf}"
#             prob += correct_and_active[i][lf] >= correctness_vars[i][lf] + x_nlfs[lf] - 1, f"CorrectAndActive_LowerBound_{i}_{lf}"
        
    
    for lf in labeling_functions:
        num_instances_abstain = pulp.lpSum([is_abstain[i][lf] for i in range(num_instances)])
#         if lf in nlfs:
#             lf_correct_predictions = pulp.lpSum([correctness_vars[i][lf] for i in range(num_instances)])
#             prob += lf_correct_predictions >= lf_acc_thresh * (num_instances-num_instances_abstain) - M * (1 - x_nlfs[lf]), f"LF_{lf}_Accuracy"
#         else:
        lf_correct_predictions = pulp.lpSum([correctness_vars[i][lf] for i in range(num_instances)])
        prob += lf_correct_predictions >= lf_acc_thresh * (num_instances-num_instances_abstain), f"LF_{lf}_Accuracy"



    for i in range(num_instances):
#         for lf in nlfs:
#             # Ensure that correctness_vars[i][lf] is counted only if x_nlf[lf] = 1
#             prob += correctness_vars[i][lf] <= M * x_nlfs[lf], f"{lf}_active_{i}"
            
#         correct_predictions_per_instance = pulp.lpSum([correctness_vars[i][lf] for lf in labeling_functions if lf not in nlfs]) + \
#                                pulp.lpSum([correct_and_active[i][lf] for lf in nlfs])
        correct_predictions_per_instance = pulp.lpSum([correctness_vars[i][lf] for lf in labeling_functions])
            
#         instance_abstain_count = pulp.lpSum([is_abstain[i][lf] for lf in labeling_functions if lf not in nlfs]) + \
#                                  pulp.lpSum([active_abstain[i][lf] for lf in nlfs]) 
        instance_abstain_count = pulp.lpSum([is_abstain[i][lf] for lf in labeling_functions])
        
#         num_labeling_functions_used = len(labeling_functions) - len(nlfs) + pulp.lpSum(x_nlfs.values())
        num_labeling_functions_used = len(labeling_functions)
        if(instance_acc_on_valid):
            prob += correct_predictions_per_instance >= instance_acc_thresh * (num_labeling_functions_used-instance_abstain_count), f"Instance_{i}_Accuracy"
        else:
            prob += correct_predictions_per_instance >= instance_acc_thresh * (num_labeling_functions_used), f"Instance_{i}_Accuracy"
        if(use_non_abstain):
            prob += instance_abstain_count <= num_labeling_functions_used *(1- min_non_abstain_thresh), f"Instance_{i}_NonAbastain"

        
    for i in range(num_instances):
        for lf in labeling_functions:
            true_label = df[expected_label_col][i]
            # Ensure that correctness_vars[i][lf] is 1 if P_vars[i][lf] equals true_label, else 0
            prob += P_vars[i][lf] - true_label <= M * (1 - correctness_vars[i][lf]),\
                                     f"Correctness_UpperBound_{i}_{lf}"
            prob += true_label - P_vars[i][lf] <= M * (1 - correctness_vars[i][lf]), \
                                     f"Correctness_LowerBound_{i}_{lf}"


    # Solve the integer program
    prob.solve()

    p_vars_solution = pd.DataFrame(index=df.index, columns=labeling_functions)
    active_abstain_df = pd.DataFrame(index=df.index, columns=labeling_functions)
    is_abstain_df = pd.DataFrame(index=df.index, columns=labeling_functions)
    
    for i in range(num_instances):
        for lf in labeling_functions:
            p_vars_solution.loc[i, lf] = int(pulp.value(P_vars[i][lf]))
    
    correctness_solution = pd.DataFrame(index=df.index, columns=labeling_functions)
    for i in range(num_instances):
        for lf in labeling_functions:
            correctness_solution.loc[i, lf] = int(pulp.value(correctness_vars[i][lf]))
    
#     x_nlfs_solution = {lf: pulp.value(x_nlfs[lf]) for lf in nlfs}
    
    print(f"Status: {pulp.LpStatus[prob.status]}")
    print(f"pulp.value(num_labeling_functions_used) : {pulp.value(num_labeling_functions_used)}")
    
#     for i in range(num_instances):
#         for lf in labeling_functions:
#             is_abstain_df.loc[i, lf] = int(pulp.value(is_abstain[i][lf]))
#     for i in range(num_instances):
#         for lf in nlfs:
#             active_abstain_df.loc[i, lf] = int(pulp.value(active_abstain[i][lf]))
    
#     return p_vars_solution, x_nlfs_solution, pulp, prob, active_abstain_df, is_abstain_df

    return p_vars_solution, pulp.value(flip_cost)


# In[11]:


# for c in list(combined_df):
#     print(f"{c}: {combined_df[c].value_counts().to_dict()}")


# In[12]:


def create_solver_input_df_copies(lf_names_after_fix, user_input_df, res_df):
    df_copies = {}

    cols_needed = ['text', 'expected_label', 'cid']

    # Loop through each column in df2 and create a copy of df1 with modified 'expected_label'
    for lf in lf_names_after_fix:
        # Create a deep copy of df1
        df_copy = user_input_df.copy(deep=True)

        # Update the 'expected_label' column based on the corresponding column in df2
        df_copy['expected_label'] = res_df[lf].values

        # Store the modified dataframe in the dictionary with key as the labeling function name
        df_copies[lf] = df_copy[cols_needed]
    
    return df_copies



# In[13]:


import math
import time 


# In[14]:


def main_driver(user_input_size,
         lf_acc_thresh,
         instance_acc_thresh,
         min_non_abstain_thresh,
        dataset_name,
        random_state,
        funcs_dictionary,
       instance_acc_on_valid,
       use_non_abstain,
        repair_strat='information_gain'):
    
    
    run_times = ['snorkel_first_run','snorkel_run_after_fix', 'solver_runtime','repair_time']
    runtime_dict = {r:0 for r in run_times}

    gen_input_tree_rules_func = funcs_dictionary[dataset_name]
    
    conn = psycopg2.connect(dbname='label', user='postgres')
    
    user_complaint_size = math.floor(user_input_size * 0.5)
    user_confirm_size = user_input_size - user_complaint_size
     
    treerules_for_user_input = gen_input_tree_rules_func()
    
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
    fix_rules_with_solver_input(fix_book_keeping_dict=fix_book_keeping_dict, repair_strategy=repair_strat)
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

    # Get the current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'path_repair_tweets-{dataset_name}_sample_params_{user_input_size}-{lf_acc_thresh}-{instance_acc_thresh}-{min_non_abstain_thresh}-{random_state}-{timestamp}.pkl', 'wb') as resf:
        pickle.dump(res_to_save, resf)
    
    conn.close()
    
    
    return fix_book_keeping_dict, res_df, gts, user_input_df, df_sentences_filtered, ret


# In[15]:


# instance accuracy: |correct_predictions_from_included_lfs|/|included_lfs|
# lf accuracy: |correct_predictions_from_each_lf|/|non_abstain_preds_from_the_lf|
# instance_non_abstain_thresh: each instance cant have more than (instance_non_abstain_thresh*100)% abstains


# In[16]:


import signal
import time 

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException

# def run_with_params(params):
#     time.sleep(params)
#     return f"Finished params: {params}"

# def test_params_with_timeout(params_list, time_limit_minutes):
#     time_limit_seconds = int(time_limit_minutes * 60)
#     signal.signal(signal.SIGALRM, timeout_handler)
#     results = []

#     for params in params_list:
#         signal.alarm(time_limit_seconds)  # Set the timeout
#         try:
#             result = run_with_params(params)
#             print(result)
#             results.append(result)
#         except TimeoutException:
#             print(f"Params {params} exceeded time limit, moving to next.")
#         finally:
#             signal.alarm(0)  # Reset the alarm

#     return results

# Example usage
# params_list = [1, 5, 10, 2]  # Parameters that would be passed to the function
# results = test_params_with_timeout(params_list, time_limit_minutes=0.1)


# In[17]:


import concurrent.futures
import time


# In[18]:


def run_main_with_params(user_input_size, lf_acc_thresh, instance_acc_thresh, 
                         min_non_abstain_thresh, random_state, dataset_name, funcs_dictionary,
                         instance_acc_on_valid,use_non_abstain, repair_strat):

    fix_book_keeping_dict, res_df, gts, user_input_df, df_sentences_filtered, summary = main_driver(
        user_input_size=user_input_size,
        lf_acc_thresh=lf_acc_thresh,
        instance_acc_thresh=instance_acc_thresh,
        min_non_abstain_thresh=min_non_abstain_thresh,
        random_state=random_state,
        dataset_name=dataset_name,
        funcs_dictionary=funcs_dictionary,
        instance_acc_on_valid=instance_acc_on_valid,
       use_non_abstain=use_non_abstain,
    repair_strat=repair_strat)
    
    res_to_save = {'summary': summary, 'fix_details': fix_book_keeping_dict}
    return res_to_save


# In[19]:


# inclusion_dict, fix_book_keeping_dict, res_df, gts, user_input_df, df_sentences_filtered, summary = main(user_input_size=40,
#  lf_acc_thresh=0.6,
#  instance_acc_thresh=0.6,
#  min_non_abstain_thresh=0.3,
#  random_state=42,
#  dataset_name='amazon01')



# In[20]:


# def frange(start, stop, step):
#     while start < stop:
#         yield round(start, 10)  # Rounding to avoid floating-point precision issues
#         start += step


# In[21]:


# import itertools
# import random

# # Define the parameter ranges
# user_input_size_range = range(20, 81, 20)  # From 20 to 80 with step 20
# lf_acc_thresh_range = [round(x, 1) for x in frange(0.2, 1.0, 0.2)]  # From 0.2 to 0.8 with step 0.2
# instance_acc_thresh_range = [round(x, 1) for x in frange(0.2, 1.0, 0.2)]
# min_non_abstain_thresh_range = [round(x, 1) for x in frange(0.2, 1.0, 0.2)]
# random_states = [42, 100]
# dataset_names = ['amazon']
# # Select 2 random states (you can choose others if preferred)

# # Generate all combinations of the parameter values
# params_list = list(itertools.product(
#     user_input_size_range,
#     lf_acc_thresh_range,
#     instance_acc_thresh_range,
#     min_non_abstain_thresh_range,
#     random_states,
#     dataset_names,
# ))


# # Now, randomly sample around 25 parameter combinations from the full list
# sampled_params_list = random.sample(params_list, 15)


# In[22]:


# inclusion_dict, fix_book_keeping_dict, res_df, gts, user_input_df, df_sentences_filtered, summary = main(
#     user_input_size=20,
#     lf_acc_thresh=0.7,
#     instance_acc_thresh=0.8,
#     min_non_abstain_thresh=0.1,
#     random_state=42,
#     dataset_name='amazon01'
# )


# In[23]:


from rbbm_src.labelling_func_src.src.example_tree_rules import (
gen_amazon_funcs,
gen_professor_teacher_funcs,
gen_painter_architecht_funcs,
gen_imdb_funcs,
gen_pj_funcs,
gen_pp_funcs,
gen_yelp_funcs,
gen_plots_funcs,
gen_fakenews_funcs,
gen_dbpedia_funcs,
gen_agnews_funcs,
gen_tweets_funcs,
gen_spam_funcs
)


# In[24]:


dataset_dict = {
#     "plots": gen_plots_funcs,
#     "amazon": gen_amazon_funcs,
#     "dbpedia": gen_dbpedia_funcs,
#     "agnews": gen_agnews_funcs,
#     "physician_professor": gen_pp_funcs,
#     "imdb": gen_imdb_funcs,
#     "fakenews": gen_fakenews_funcs,
#     "yelp": gen_yelp_funcs,
#     "photographer_journalist": gen_pj_funcs,
#     "professor_teacher": gen_professor_teacher_funcs,
#     "painter_architect": gen_painter_architecht_funcs,
    "tweets": gen_tweets_funcs,
#     "spam": gen_spam_funcs,
}


# In[25]:


from collections import defaultdict
import psycopg2
import pandas as pd
import concurrent.futures
import time


# In[26]:


def test_main_with_timeout(params_list, time_limit_minutes):
    time_limit_seconds = time_limit_minutes * 60
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for params in params_list:
            future = executor.submit(run_main_with_params, *params)
            try:
                result = future.result(timeout=time_limit_seconds)
                print(f"Params {params} finished successfully.")
                results.append(result)
            except concurrent.futures.TimeoutError:
                print(f"Params {params} exceeded the time limit, moving to the next set.")
    
    return results


# In[27]:


# def frange(start, stop, step):
#     while start < stop:
#         yield round(start, 10)  # Rounding to avoid floating-point precision issues
#         start += step

        
# import itertools
# import random

# # Define the parameter ranges
# user_input_size_range = range(20, 81, 20)  # From 20 to 80 with step 20
# lf_acc_thresh_range = [round(x, 1) for x in frange(0.2, 1.0, 0.2)]  # From 0.2 to 0.8 with step 0.2
# instance_acc_thresh_range = [round(x, 1) for x in frange(0.2, 1.0, 0.2)]
# min_non_abstain_thresh_range = [round(x, 1) for x in frange(0.2, 1.0, 0.2)]
# random_states = [42, 100]
# dataset_names = ['amazon']
# # Select 2 random states (you can choose others if preferred)

# # Generate all combinations of the parameter values
# params_list = list(itertools.product(
#     user_input_size_range,
#     lf_acc_thresh_range,
#     instance_acc_thresh_range,
#     min_non_abstain_thresh_range,
#     random_states,
#     dataset_names,
# ))


# # Now, randomly sample around 25 parameter combinations from the full list
# sampled_params_list = random.sample(params_list, 15)


# In[28]:


# res_storing = defaultdict(dict)


# In[29]:


# for uinput in [100, 150, 80]:
#     for rs in [123, 42]:
#         for mat in [0.1, 0.5, 0.8]:
#             for dd in dataset_dict:
#                 fix_book_keeping_dict, res_df, gts, user_input_df, df_sentences_filtered, summary = main_driver(
#                     user_input_size=uinput,
#                     lf_acc_thresh=0.7,
#                     instance_acc_thresh=0.8,
#                     min_non_abstain_thresh=mat,
#                     random_state=rs,
#                     dataset_name=dd,
#                     gen_input_tree_rules_func=dataset_dict[dd],
#                     conn=conn
#                 )
#                 res_to_store = {   
#                 'fix_book_keeping_dict': fix_book_keeping_dict,
#                 'summary': summary,
#                 }
#                 res_storing[dd] = res_to_store


# In[1]:


# user_input_sizes = [20, 40]
# random_states = [123, 42]
# lf_acc_threshs = [0.7]
# instance_acc_threshs = [0.8]
# non_abstain_threshs = [0.5, 0.8]
# datasets = list(dataset_dict)
# func_dictionary = [dataset_dict]


# testing agnews
user_input_sizes = [3]
random_states = [1]
lf_acc_threshs = [0.7]
instance_acc_threshs = [0.8]
non_abstain_threshs = [0.8]
datasets = list(dataset_dict)
func_dictionary = [dataset_dict]
instance_acc_on_valids=[True]
use_non_abstains=[True]
repair_strats = ['optimal']

# user_input_sizes_20 = [20]
# random_states_20 = [1,321,4,123,6,5,2,7,8,3,42]
# lf_acc_threshs_20 = [0.7]
# instance_acc_threshs_20 = [0.8]
# non_abstain_threshs_20 = [0.8]
# datasets_20 = list(dataset_dict)
# func_dictionary_20 = [dataset_dict]
# instance_acc_on_valids_20=[False]
# use_non_abstains_20=[False]


# testing agnews on 80 , random_state=6 only
# user_input_sizes = [80]
# random_states = [6]
# lf_acc_threshs = [0.7]
# instance_acc_threshs = [0.8]
# non_abstain_threshs = [0.8]
# datasets = list(dataset_dict)
# func_dictionary = [dataset_dict]
# instance_acc_on_valids=[False]
# use_non_abstains=[False]


# user_input_sizes_1 = [80]
# random_states_1 = [6]
# lf_acc_threshs_1 = [0.7]
# instance_acc_threshs_1 = [0.8]
# non_abstain_threshs_1 = [0.8]
# datasets_1 = list(dataset_dict)
# func_dictionary_1 = [dataset_dict]
# instance_acc_on_valids_1=[True]
# use_non_abstains_1=[True]


# In[31]:


import itertools


# In[32]:


input_params = list(itertools.product(
    user_input_sizes,
    lf_acc_threshs,
    instance_acc_threshs,
    non_abstain_threshs,
    random_states,
    datasets,
    func_dictionary,
    instance_acc_on_valids,
    use_non_abstains,
    repair_strats
))


# input_params_20 = list(itertools.product(
#     user_input_sizes_20,
#     lf_acc_threshs_20,
#     instance_acc_threshs_20,
#     non_abstain_threshs_20,
#     random_states_20,
#     datasets_20,
#     func_dictionary_20,
#     instance_acc_on_valids_20,
#     use_non_abstains_20
# ))


# In[34]:


# for i in range(0,3):
#     test_main_with_timeout(input_params, time_limit_minutes=20)

# for i in range(0,3):
test_main_with_timeout(input_params, time_limit_minutes=20)


# In[ ]:


# test_main_with_timeout(input_params_20, time_limit_minutes=20)


# In[ ]:





# In[ ]:


# res_dfs = []


# In[ ]:


# for dname in res_storing:
#     dsummary = pd.DataFrame([res_storing[dname]['summary']])
#     dsummary['dataset'] = dname
#     res_dfs.append(dsummary)


# In[ ]:


# df_res = pd.concat(res_dfs)


# In[ ]:


# df_res.sort_values(by='dataset_name')


# In[ ]:


# df_res[['before_fix_global_accuracy','confirm_prev_rate','complain_fix_rate','new_global_accuracy','dataset']]


# In[ ]:


# user_input_sizes_test = [20]
# random_states_test = [7]
# lf_acc_threshs_test = [0.7]
# instance_acc_threshs_test = [0.8]
# non_abstain_threshs_test = [0.8]
# datasets_test = list(dataset_dict)
# func_dictionary_test = [dataset_dict]


# In[ ]:


# input_params_test = list(itertools.product(
#     user_input_sizes_test,
#     lf_acc_threshs_test,
#     instance_acc_threshs_test,
#     non_abstain_threshs_test,
#     random_states_test,
#     datasets_test,
#     func_dictionary_test
# ))


# In[ ]:


# test_main_with_timeout(input_params_test, time_limit_minutes=20)


# In[ ]:


# dataset_dict = {
#     "plots": gen_plots_funcs,
#     "amazon": gen_amazon_funcs,
#     "dbpedia": gen_dbpedia_funcs,
#     "agnews": gen_agnews_funcs,
#     "physician_professor": gen_pp_funcs,
#     "imdb": gen_imdb_funcs,
#     "fakenews": gen_fakenews_funcs,
#     "yelp": gen_yelp_funcs,
#     "photographer_journalist": gen_pj_funcs,
#     "professor_teacher": gen_professor_teacher_funcs,
#     "painter_architect": gen_painter_architecht_funcs,
#     "tweets": gen_tweets_funcs,
#     "spam": gen_spam_funcs,
# }

