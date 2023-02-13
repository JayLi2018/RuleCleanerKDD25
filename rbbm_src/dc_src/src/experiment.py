# implementation of causuality based 
# explanation
# import pandas as pd
# from typing import *
# from snorkel.labeling import (
# 	LabelingFunction, 
# 	labeling_function, 
# 	PandasLFApplier, 
# 	LFAnalysis,
# 	filter_unlabeled_dataframe
# 	)
# from snorkel.labeling.model import MajorityLabelVoter, LabelModel
# import os
# import logging
# import argparse
# import logconfig
# from dc_srcsrc/experiment.pybottom_up import sentence_filter, delete_words
# from itertools import chain
# from lfs import (
# 	LFs,
# 	twitter_lfs, 
# 	LFs_smaller, 
# 	LFs_running_example,
# 	lattice_lfs,
# 	lattice_dict)
# from classes import SPAM, HAM, ABSTAIN
# from itertools import combinations
# import glob
# from timer import ExecStats
# import numpy as np
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from copy import deepcopy
# from LabelRepair import LabelRepairer, Repair
# from LabelExplain import LabelExpaliner
import warnings
warnings.filterwarnings('ignore')
from rbbm_src.holoclean.examples.holoclean_repair import main
from rbbm_src.holoclean.hc_responsibility import rule_responsibility
from .classes import Complaint, RulePruner, DataPruner
import pandas as pd
from datetime import datetime
import random 
from rbbm_src.classes import StatsTracker


def dc_main(dc_input):

    # conn = psycopg2.connect(dbname="holo", user="holocleanuser", password="abcd1234")
    conn = dc_input.connection
    print(conn)
    conn.autocommit=True
    cur=conn.cursor()
    input_file=dc_input.input_csv_dir+dc_input.input_csv_file
    table_name=dc_input.input_csv_file.split('.')[0]
    cols=0
    num_lines=0
    try:
        with open(input_file) as f:
            first_line = f.readline()
            num_lines = sum(1 for line in f)
    except Exception as e:
        print(f'cant read file {input_file}')
        exit()
    else:
        cols=first_line.split(',')
        # find out column names:

    # drop preexisted repaired records 
    select_old_repairs_q = f"""
    SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES 
    WHERE TABLE_NAME LIKE '{table_name}_repaired_%' AND TABLE_TYPE = 'BASE TABLE'
    """
    cur.execute(select_old_repairs_q)

    for records in cur.fetchall():
        drop_q = f"drop table if exists {records[0]}"
        cur.execute(drop_q)


    # main(f'/home/opc/chenjie/holoclean/testdata/dc_finder_hospital_rules_{s}',initial_training=True)
    main(table_name=table_name, csv_dir=dc_input.input_csv_dir, 
        csv_file=dc_input.input_csv_file, dc_dir=dc_input.input_dc_dir, dc_file=dc_input.input_dc_file, gt_dir=dc_input.ground_truth_dir, 
        gt_file=dc_input.ground_truth_file, initial_training=True)


    union_sql = f"""
    SELECT 'before_clean' AS type, * from  {table_name}
    union all 
    SELECT 'after_clean' AS type, * from {table_name}_repaired
    order by _tid_
    """
    df_union_before_and_after = pd.read_sql(union_sql, conn) 

    j = 0
    for i in range(0,num_lines-1):
        if(j%100==0):
            print(j)
        j+=1
        df_row = pd.DataFrame(columns=['type']+cols)
        df_row.loc[0,'type'] = 'ground_truth'
        df_row.loc[0,'_tid_'] = i
        q = f"""
        SELECT * FROM {table_name}_clean WHERE _tid_={i} 
        """
        df_for_one_row = pd.read_sql(q,conn)
        for index, row in df_for_one_row.iterrows():
            df_row.loc[0, f"{row['_attribute_']}"] = row['_value_']
        df_union_before_and_after = pd.concat([df_union_before_and_after, df_row])


    repaired_dict = {x:[] for x in cols}

    # iterate this dataframe to find all wrong predictions and list them to choose
    grouped = df_union_before_and_after.groupby('_tid_')
    k = 0

    for name, group in grouped:
        tid = pd.to_numeric(group.iloc[0]['_tid_'], downcast="integer")
        if(k%100==0):
            print(k)
        for c in cols:
            if(group[group['type']=='after_clean'][c].to_string(index=False)!= group[group['type']=='before_clean'][c].to_string(index=False)):
                repaired_dict[c].append(tid)
        k+=1

    for d,v in repaired_dict.items():
        print(f"{d}: {len(repaired_dict[d])}")


    wrong_dict = {x:[] for x in cols}
    correct_dict = {x:[] for x in cols}

    # iterate this dataframe to find all wrong predictions and list them to choose
    grouped = df_union_before_and_after.groupby('_tid_')
    k = 0
    for name, group in grouped:
        tid = pd.to_numeric(group.iloc[0]['_tid_'], downcast="integer")
        if(k%100==0):
            print(k)
        for c in cols:
            if((group[group['type']=='after_clean'][c].to_string(index=False)!= group[group['type']=='ground_truth'][c].to_string(index=False))):
                wrong_dict[c].append(tid)
            if((group[group['type']=='after_clean'][c].to_string(index=False)!= group[group['type']=='before_clean'][c].to_string(index=False)) and \
                group[group['type']=='after_clean'][c].to_string(index=False)== group[group['type']=='ground_truth'][c].to_string(index=False)):
                correct_dict[c].append(tid)
             # \
             #    and (group[group['type']=='after_clean'][c].to_string(index=False)!= group[group['type']=='before_clean'][c].to_string(index=False))):
        k+=1
    wrong_repairs=[]
    correct_repairs=[]
    for k in wrong_dict:
        wrong_df = df_union_before_and_after[df_union_before_and_after['_tid_'].isin(wrong_dict[k])]
        wrong_df['wrong_attr'] = k
        wrong_repairs.append(wrong_df)
    for k in correct_dict:
        correct_df = df_union_before_and_after[df_union_before_and_after['_tid_'].isin(correct_dict[k])]
        correct_df['corrected_attr'] = k
        correct_repairs.append(correct_df)
    wrong_repairs_df = pd.concat(wrong_repairs)
    correct_repairs_df = pd.concat(correct_repairs)
    # wrong_repairs_df.drop_duplicates(['wrong_attr','_tid_'])
    wrong_repairs_df.sort_values(by=['type','_tid_', 'wrong_attr']).to_csv('wrongs_after_fix.csv', index=False)
    correct_repairs_df.sort_values(by=['type','_tid_', 'corrected_attr']).to_csv('correct_after_fix.csv', index=False)

    dfs = [x for _, x in wrong_repairs_df.groupby(['wrong_attr','_tid_'])]
    print(f"wrong dfs len: {len(dfs)}")
    now = datetime.now() # current date and time
    date_time = now.strftime("%m%d%Y%H%M%S")

    if(dc_input.prune_only):
        for d in dfs:
            print(d[:1]['wrong_attr'])
            print(d[:1]['_tid_'])
            attr_name, tid = list(d[:1]['wrong_attr'])[0], list(d[:1]['_tid_'])[0]
            complaint=Complaint(complain_type='DC', attr_name=attr_name, tid=int(tid))
            rule_responsibility(complaint, dc_input, f'prune_only_{dc_input.input_dc_file}_{date_time}.txt')
    else:
        if(dc_input.user_provide):
            for d in dfs:
                # wattr = d.iloc[0]['wrong_attr']
                # print(wattr)
                # if((d[d['type']=='after_clean'].iloc[0][wattr])==(d[d['type']=='before_clean'].iloc[0][wattr])):
                print("--------------------------------------------------------------------------------------------")  
                print(d)
                print('\n')
                # print(f"number of wrong repairs: {len(dfs)}")
            choices = input('please input wrong_attr name and _tid_ to identify the errorous repaired result (<wrong_attr>:<_tid_>): (eg: Country:7)')
            # conn.close()
            # print("need to construct a complaint now")
            attr_name, tid = choices.split(':')
            complaint=Complaint(complain_type='DC', attr_name=attr_name, tid=int(tid))
            rule_responsibility(complaint, dc_input, f'run_{dc_input.input_dc_file}_{date_time}.txt')
        else:
            # print(f"complaint: {random.sample(dfs, 1)[0]}")
            # complaint_dict=random.sample(dfs, 1)[0].to_dict('records')[0]
            complaint_dict=dfs[dc_input.random_number_for_complaint % len(dfs)].to_dict('records')[0]
            f=open(f'run_{date_time}.txt', 'a')
            print("complaint:", file=f)
            print(complaint_dict, file=f)
            f.close()
            complaint=Complaint(complain_type='DC', attr_name=complaint_dict['wrong_attr'], tid=int(complaint_dict['_tid_']))
            print('the chosen complaint is:')
            print(complaint_dict)
            ## given an attribute and a tid (row id), find responsible rule(s)
            rule_responsibility(complaint, dc_input, f'run_{dc_input.input_dc_file}_{date_time}.txt')

            dc_input.sample_contingency=False
            dc_input.stats=StatsTracker()
            rule_responsibility(complaint, dc_input, f'run_{dc_input.input_dc_file}_{date_time}_no_sample.txt')
     