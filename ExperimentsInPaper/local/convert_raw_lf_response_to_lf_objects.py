
import sys

import pickle
import glob
import re
from rulecleaner_src.lfs_tree import keyword_labelling_func_builder
from rulecleaner_src.TreeRules import SPAM, HAM, ABSTAIN, PredicateNode
from rulecleaner_src.LFRepair import populate_violations, fix_rules_with_solver_input
from rulecleaner_src.utils import run_snorkel_with_funcs, select_user_input, clean_text
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
from IPython.display import Image, display 
import datetime
import itertools
from rulecleaner_src.main import main
import random
from rulecleaner_src.lfs_tree import keyword_labelling_func_builder, regex_func_builder



def filter_invalid_regex(regex_str):
    # print(f'testing : {regex_str}')
    try:
        re.compile(regex_str)
        # print("Regex is valid!")
        return regex_str
    except re.error as e:
        print(f"Regex error: {e} for pattern {regex_str}")
        return None


def parse_labeling_functions(text: str):
    """
    Parses multiple keyword-based and regex-based labeling functions from a text file.
    Returns a list of extracted function details.
    """
    functions = []

    # Match all function definitions (both keyword & regex)
    function_matches = re.findall(r'\bdef (keyword|regex)_(\w+)(\d+)\(x\):', text, re.IGNORECASE)
    
    # print(f"DEBUG: Found {len(function_matches)} functions.")  # Print function count
    
    for lf_type, category, label in function_matches:
        expected_label = int(label)

        # Extract the full function block (handling multi-line functions)
        function_block_match = re.search(
            rf'def {lf_type}_{category}{label}\(x\):([\s\S]+?)(?=\n\s*def |\Z)', text, re.IGNORECASE
        )

        if not function_block_match:
            print(f"ERROR: Could not extract function body for {lf_type}_{category}{label}")
            continue

        function_body = function_block_match.group(1)
        # print(f"DEBUG: Extracted function {lf_type}_{category}{label}")

        # Extract keywords for keyword-based functions
        if lf_type == "keyword":
            keywords_match = re.search(r'keywords\s*=\s*\[(.*?)\]', function_body, re.DOTALL)
            if keywords_match:
                keywords = [kw.strip(" '\"") for kw in keywords_match.group(1).split(',')]
                functions.append({
                    "type": "keyword",
                    "category": category.lower(),
                    "keywords": keywords,
                    "expected_label": expected_label
                })
            else:
                print(f"ERROR: Could not extract keywords for {lf_type}_{category}{label}")

        # Extract regex patterns for regex-based functions
        elif lf_type == "regex":
            regex_patterns = []

            # Extract `re.search(r'pattern', x, re.IGNORECASE)` occurrences
            regex_match = re.findall(r"re\.search\(\s*r?['\"](.+?)['\"],\s*x", function_body, re.DOTALL)
            regex_patterns.extend(regex_match)

            # Extract single regex pattern assigned to a variable
            single_pattern_match = re.findall(r"pattern\s*=\s*r?['\"](.+?)['\"]", function_body)
            regex_patterns.extend(single_pattern_match)

            # Extract multi-line regex list (`patterns = [...]`)
            pattern_list_match = re.search(r"\s*patterns\s*=\s*\[(.*?)\]", function_body, re.DOTALL)
            if pattern_list_match:
                patterns_raw = pattern_list_match.group(1)

                # Extract individual patterns inside the list
                patterns_cleaned = re.findall(r"r?['\"](.+?)['\"]", patterns_raw, re.DOTALL)
                patterns_cleaned = [p.replace("\\'", "'") for p in patterns_cleaned]  # Fix escaped single quotes
                regex_patterns.extend(patterns_cleaned)

            # Debugging output
            # print(f"DEBUG: Extracted patterns for {lf_type}_{category}{label}: {regex_patterns}")

            # Ensure all valid regex patterns are captured
            if regex_patterns:
                functions.append({
                    "type": "regex",
                    "category": category.lower(),
                    "patterns": regex_patterns,
                    "expected_label": expected_label
                })
            else:
                print(f"ERROR: Could not extract regex patterns for {lf_type}_{category}{label}")

    # print(f"DEBUG: Successfully parsed {len(functions)} functions.")
    return functions, len(function_matches)


def sample_from_list(data_list, sample_size, seed=42):
    random.seed(seed)  # Set seed for reproducibility
    return random.sample(data_list, sample_size)



# user_input_sizes = [80, 120, 150]
# random_states = [1,2]
# lf_acc_threshs = [0.7]
# instance_acc_threshs = [0.8]
# non_abstain_threshs = [0.8]
# datasets = list(lf_construct_infos)
# func_dictionary = [lf_construct_infos]
# instance_acc_on_valids=[False]
# use_non_abstains=[False]
# pfile_name_prefix = ('test_folder/test_run',)
# lf_source = 'gpt_generated'

# input_params2 = list(itertools.product(
#     user_input_sizes,
#     lf_acc_threshs,
#     instance_acc_threshs,
#     non_abstain_threshs,
#     datasets,
#     random_states,
#     func_dictionary,
#     instance_acc_on_valids,
#     use_non_abstains,
#     pfile_name_prefix,
#     lf_source
# ))


 


# for ip in input_params2:
#     main(*ip)



if __name__ == '__main__':
    
    raw_responses = {}

    for fname in glob.glob('lf_raw_response_datase*_new.pkl'):
        print(fname)
        match = re.match(r'lf_raw_response_datase_(\w+)_new', fname)
        dataset_name = match.group(1)
        with open(fname, 'rb') as f:
            raw_responses[dataset_name] = pickle.load(f)


    with open('lf_raw_response_datase_spam_new.pkl', 'rb') as f:
        spam_raw = pickle.load(f)
    

    used_datasets = [
        'plots',
        'spam',
        'tweets',
        'painter_architect',
        'professor_teacher',
        'yelp',
        'fakenews',
        'imdb',
        'physician_professor',
        'agnews',
        'amazon'
    ]

    conn = psycopg2.connect(
        dbname="label",
        user="postgres",
    )

    mapping_dicts = {}

    for ud in used_datasets:
        print(ud)
        info_df = pd.read_sql(f'select count(*), class, label from {ud} group by class, label', conn)
        print(info_df)
        result = dict(zip(info_df['label'], info_df['class']))
        mapping_dicts[ud]= result
        print('\n')


    allowed_labels = {}


    for k,v in mapping_dicts.items():
        allowed_labels[k] = set(v.values())

    lf_construct_infos = {}


    for k, v in raw_responses.items():
        if(k in used_datasets):
            lf_construct_infos[k] = []
            for r in raw_responses[k]:
                infos, num_matches = parse_labeling_functions(r)
                # for f in infos:
                #     if(f['type']=='regex' and len(f['patterns'])>1):
                #         print(r)
                #         print(infos)
                if(len(infos)!=num_matches):
                    print(f"didnt parse all functions succesfully, the text has {num_matches}, but it only parsed {len(infos)}")
                    # print(f"original text")
                    print(r)
                    # print("infos")
                    print("returned result")
                    print(infos)
                    print('*******'*20)
                    continue

        
                lf_construct_infos[k].extend(infos)


    

    df_lf_metadata = pd.read_csv('dataset_lf_meta_data.csv')
    df_lf_metadata = df_lf_metadata.iloc[:, 1:]  # Drop the first column

    lf_metadata_dict = df_lf_metadata.to_dict(orient='records')



    dataset_dict = {item['Dataset']: {k: v for k, v in item.items() if k != 'Dataset'} for item in lf_metadata_dict}



    for k,v in lf_construct_infos.items():
        lf_construct_infos[k] = [x for x in lf_construct_infos[k] if x['expected_label'] in allowed_labels[k]]



    for k,v in lf_construct_infos.items():
        lf_construct_infos[k] = sample_from_list(lf_construct_infos[k], dataset_dict[k]['LFCount'], seed=123)

                    
    user_input_sizes = [20, 40, 80, 120, 150]
    random_states = [5,6,7,8,9,10,11,12]
    lf_acc_threshs = [0.7]
    instance_acc_threshs = [0.8]
    non_abstain_threshs = [0.8]
    datasets = list(lf_construct_infos)
    # datasets = ['spam']
    func_dictionary = [lf_construct_infos]
    instance_acc_on_valids=[False]
    use_non_abstains=[False]
    pfile_name_prefix = ('test_folder/test_run',)
    lf_source = ['gpt_generated']


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
        lf_source
    ))

    for k in func_dictionary[0]:
        print(f'filtering rules for {k}')
        for i in range(len(func_dictionary[0][k])):
            if(func_dictionary[0][k][i]['type']=='regex'):
                for j in range(len(func_dictionary[0][k][i]['patterns'])):
                    func_dictionary[0][k][i]['patterns'][j] = filter_invalid_regex(func_dictionary[0][k][i]['patterns'][j])
                func_dictionary[0][k][i]['patterns'] = [x for x in func_dictionary[0][k][i]['patterns'] if x is not None]


    for ip in input_params:
        main(*ip)