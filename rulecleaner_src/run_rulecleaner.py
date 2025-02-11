#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
sys.path.append(os.path.join(os.getcwd(), ".."))


# In[2]:


from rulecleaner_src.lfs_tree import keyword_labelling_func_builder
from rulecleaner_src.TreeRules import SPAM, HAM, ABSTAIN, PredicateNode
from rulecleaner_src.LFRepair import populate_violations, fix_rules_with_solver_input
from rulecleaner_src.utils import run_snorkel_with_funcs, select_user_input, clean_text


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
import random
from collections import deque, defaultdict
import numpy as np
import pickle
from IPython.display import Image, display 
import datetime
import itertools


# In[3]:


# from rulecleaner_src.example_tree_rules import (
# gen_amazon_funcs,
# gen_professor_teacher_funcs,
# gen_painter_architecht_funcs,
# gen_imdb_funcs,
# gen_pj_funcs,
# gen_pp_funcs,
# gen_yelp_funcs,
# gen_plots_funcs,
# gen_fakenews_funcs,
# gen_dbpedia_funcs,
# gen_agnews_funcs,
# gen_tweets_funcs,
# gen_spam_funcs
# )

from rulecleaner_src.example_tree_rules import (
gen_agnews_funcs_4_class, gen_chemprot_func
)


# In[4]:


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

# dataset_dict = {'agnews4': gen_agnews_funcs_4_class}
dataset_dict = {'chemprot': gen_chemprot_func}


# In[5]:


from rulecleaner_src.main import main


# In[6]:


# user_input_sizes = [40, 80]
# random_states = [1,2,3,4,5,6,7,8, 9, 10]
# lf_acc_threshs = [0.7]
# instance_acc_threshs = [0.8]
# non_abstain_threshs = [0.8]
# datasets = list(dataset_dict)
# func_dictionary = [dataset_dict]
# instance_acc_on_valids=[False]
# use_non_abstains=[False]
# pfile_name_prefix = ('test_folder/test_run',)


user_input_sizes = [20, 40, 80, 120, 150]
random_states = [1, 2, 3,4,5,6,7,8]
lf_acc_threshs = [0.7]
instance_acc_threshs = [0.8]
non_abstain_threshs = [0.8]
datasets = list(dataset_dict)
func_dictionary = [dataset_dict]
instance_acc_on_valids=[False]
use_non_abstains=[False]
pfile_name_prefix = ('test_folder_multi_class_chemprot/test_run',)
num_class=[10]


# In[7]:


input_params = list(itertools.product(
    user_input_sizes,
    lf_acc_threshs,
    instance_acc_threshs,
    non_abstain_threshs,
    datasets,
    random_states,
    func_dictionary,
    instance_acc_on_valids,
    use_non_abstains,
    pfile_name_prefix,
    ['witan'],
    num_class
))


# In[8]:


input_params


# In[ ]:


for ip in input_params:
    main(*ip)


# In[ ]:




