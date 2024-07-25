# from lfs_tree import *
import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)
import pydot
import networkx as nx
from string import Template
import pandas as pd
import glob
import pydot
import pickle
from IPython.display import Image, display

def view_pydot(pdot):
    plt = Image(pdot.create_png())
    display(plt)

def show_trees(directory):
    print(directory)
    # treefiles = glob.glob(f'{directory}*tree_*')
    # num_trees_per_strat = len(treefiles)/2
    # for i in range(0, int(num_trees_per_strat)):
    for f in glob.glob(f'{directory}*tree_*'):
        file = open(f)
        dot_string = file.read()
        print(f)
        graph = pydot.graph_from_dot_data(dot_string)[0]
        view_pydot(graph)
        print('\n')

def show_stats(directory):
    f = glob.glob(f'{directory}*experiment_stats')[0]
    print(f)
#                                 experiment_stats
#     file = open('../intro_example/experiment_stats')
    df = pd.read_csv(f)
    print(df)
    return df

def show_user_inputs(directory):
    f = glob.glob(f'{directory}sampled*')[0]
    df = pd.read_csv(f)
    print(list(df))
    return df[['text', 'expected_label', 'model_pred', 'id']].sort_values(by=['text'])

def view_repair_bookkeeping_results(directory):
    print(glob.glob(f'{directory}*book_keeping_dict*'))
    f = glob.glob(f'{directory}*book_keeping_dict*')[0]
#               20230921002723fix_book_keeping_dict.pkl
    with open(f, 'rb') as file:
        # Load the object from the file
        loaded_object = pickle.load(file)
#         print("Object loaded successfully:")
#         print(loaded_object)
    return loaded_object


def show_results(df_user, initial_results, after_fix_results):
    user_input_ids_set=set(df_user['cid'].to_list())
    initial_results = initial_results[initial_results['cid'].isin(user_input_ids_set)][['text',\
                                                                                       'vectors','model_pred_prob_tuple', \
                                                                                       'model_pred_diff', 'expected_label',\
                                                                                       'model_pred', 'cid']]
    after_fix_user_input= after_fix_results[after_fix_results['cid'].isin(user_input_ids_set)]
    after_fix_user_input = after_fix_user_input[['cid','vectors','model_pred','model_pred_prob_tuple']]
    before_fix_user_input = initial_results[initial_results['cid'].isin(user_input_ids_set)]
    after_fix_user_input = after_fix_user_input.rename(columns={'vectors': 'new_vectors', 'model_pred': 'new_model_pred',
                                                               'model_pred_prob_tuple': 'new_model_pred_prob_tuple'})
    result_summary  = pd.merge(after_fix_user_input, before_fix_user_input, on=['cid'])
    return result_summary