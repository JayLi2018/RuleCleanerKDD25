from itertools import combinations
import glob
import numpy as np
from copy import deepcopy
from math import floor
from itertools import product
from collections import deque,OrderedDict
from datetime import datetime
import psycopg2 
import time
from snorkel.labeling import (
	LabelingFunction, 
	labeling_function, 
	PandasLFApplier, 
	LFAnalysis,
	filter_unlabeled_dataframe
	)
from snorkel.labeling.model import MajorityLabelVoter, LabelModel
from typing import *
import pandas as pd
import logging
import random
import os
import argparse
import nltk
from nltk.corpus import stopwords
nltk.download('words')

from nltk.corpus import words as nltk_words

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import copy
import pickle
import argparse
from itertools import chain
from rbbm_src.labelling_func_src.src.TreeRules import (
	textblob_sentiment,
	Node,
	Predicate,
	KeywordPredicate,
	DCAttrPredicate,
	DCConstPredicate,
	SentimentPredicate,
	PredicateNode,
	LabelNode,
	TreeRule,
	SPAM,
	HAM,
	ABSTAIN,
	CLEAN,
	DIRTY,
	textblob_sentiment
)
from rbbm_src.labelling_func_src.src.example_tree_rules import (
	gen_example_funcs,gen_amazon_funcs,
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
from rbbm_src.labelling_func_src.src.KeyWordRuleMiner import KeyWordRuleMiner 
from rbbm_src.classes import StatsTracker, FixMonitor, RepairConfig, lf_input
from rbbm_src.labelling_func_src.src.classes import lf_input_internal, clean_text
from rbbm_src import logconfig
from rbbm_src.labelling_func_src.src.bottom_up import sentence_filter, delete_words
from rbbm_src.labelling_func_src.src.lfs import (
	LFs)
import pdb
from rbbm_src.labelling_func_src.src.classes import lf_input_internal, clean_text


nltk.download('stopwords') 
stop_words = set(stopwords.words('english')) 
nltk_words_set =  set(nltk_words.words())

logger = logging.getLogger(__name__)


def populate_violations(tree_rule, complaint):
	# given a tree rule and a complaint, populate the complaint
	# to the leaf nodes

	leaf_node = tree_rule.evaluate(complaint,'node')
	leaf_node.pairs[complaint['expected_label']].append(complaint)
	# print(leaf_nodes)
	# print(tree_rule)
	# print('\n')
	return leaf_node

def redistribute_after_fix(tree_rule, node, the_fix):
	# there are some possible "side effects" after repair for a pair of violations
	# which is solving one pair can simutaneously fix some other pairs so we need 
	# to redistribute the pairs in newly added nodes if possible
	sign=None
	cur_number=tree_rule.max_node_id+1

	# the_fix_words, llabel, rlabel = the_fix

	# if(reverse):
	# 	llabel, rlabel = rlabel, llabel
	new_predicate_node = PredicateNode(number=cur_number,pred=KeywordPredicate(keywords=[the_fix]))
	new_predicate_node.is_added=True
	cur_number+=1
	new_predicate_node.left= LabelNode(number=cur_number, pairs={HAM:[], SPAM:[]}, used_predicates=set([the_fix]))
	new_predicate_node.left.is_added=True
	cur_number+=1
	new_predicate_node.right=LabelNode(number=cur_number, pairs={HAM:[], SPAM:[]}, used_predicates=set([the_fix]))
	new_predicate_node.right.is_added=True
	new_predicate_node.left.parent= new_predicate_node
	new_predicate_node.right.parent= new_predicate_node
	tree_rule.max_node_id=max(cur_number, tree_rule.max_node_id)

	# print(node)
	if(node.parent.left is node):
		node.parent.left=new_predicate_node
	else:
		node.parent.right=new_predicate_node

	new_predicate_node.parent = node.parent

	for k in [SPAM, HAM]:
		for p in node.pairs[k]:
			if(new_predicate_node.pred.evaluate(p)):
				new_predicate_node.right.pairs[p['expected_label']].append(p)
				new_predicate_node.right.used_predicates.add(the_fix)
			else:
				new_predicate_node.left.pairs[p['expected_label']].append(p)
				new_predicate_node.left.used_predicates.add(the_fix)

	if(len(new_predicate_node.left.pairs[DIRTY])>len(new_predicate_node.left.pairs[CLEAN])):
		new_predicate_node.left.label=DIRTY
	elif(len(new_predicate_node.left.pairs[DIRTY])<len(new_predicate_node.left.pairs[CLEAN])):
		new_predicate_node.left.label=CLEAN
	if(len(new_predicate_node.right.pairs[DIRTY])>len(new_predicate_node.right.pairs[CLEAN])):
		new_predicate_node.right.label=DIRTY
	elif(len(new_predicate_node.right.pairs[DIRTY])<len(new_predicate_node.right.pairs[CLEAN])):
		new_predicate_node.right.label=CLEAN

	new_predicate_node.pairs={SPAM:{}, HAM:{}}

	return new_predicate_node

def find_available_repair(ham_sentence, spam_sentence, used_predicates, all_possible=False):
	"""
	given a leafnode, we want to find an attribute that can be used
	to give a pair of tuples the desired label.

	there are 2 possible resources:
	1: attribute equals/not equal attribute (NOTE: limited to the same attribute for now)
	2: attribute equals/not equals constant: 
		1. equals is just constant itself
		2. not equals is find any other value exist in the domain
	"""
	# loop through every attribute to find one
	res = []

	# find the difference between ham_pair and spam_pair
	ham_sentence, spam_sentence = ham_sentence.text, spam_sentence.text

	ham_available_words=list(OrderedDict.fromkeys([h for h in ham_sentence.split() if h not in spam_sentence.split()]))
	spam_available_words=list(OrderedDict.fromkeys([h for h in spam_sentence.split() if h not in ham_sentence.split()]))

	# print(f'ham_available_words: {ham_available_words}')
	# print(f'spam_available_words: {spam_available_words}')

	# exit()


	# start with attribute level and then constants
	# cand=None
	for w in ham_available_words:
		# tuple cand has x elements: 
		# w: the word to differentate the 2 sentences
		# llabel: the left node label
		# rlabel: the right node label 
		# check if the predicate is already present
		# in the current constraint
		if(w.lower() not in stop_words and w.lower() in nltk_words_set):
			if(w not in used_predicates):
				if(not all_possible):
					return w
				else:
					res.append(w)

	for w in spam_available_words:
		# tuple cand has x elements: 
		# w: the word to differentate the 2 sentences
		# llabel: the left node label
		# rlabel: the right node label 
		# check if the predicate is already present
		# in the current constraint
		if(w.lower() not in stop_words and w.lower() in nltk_words_set):
			if(w not in used_predicates):
				if(not all_possible):
					return w
				else:
					res.append(w)
	return res

def locate_node(tree, number):
	# print(tree)
	# print(f"locating node {number}")
	queue = deque([tree.root])
	while(queue):
		cur_node = queue.popleft()
		if(cur_node.number==number):
			return cur_node
		if(cur_node.left):
			queue.append(cur_node.left)
		if(cur_node.right):
			queue.append(cur_node.right)
	print('cant find the node!')
	exit()

def check_tree_purity(tree_rule, start_number=1):
	# print("checking purity...")
	# print("**********************************")
	# print(tree_rule)
	# exit()
	root = locate_node(tree_rule, start_number)
	queue = deque([root])
	leaf_nodes = []
	while(queue):
		# print(queue)
		cur_node = queue.popleft()
		if(isinstance(cur_node,LabelNode)):
			leaf_nodes.append(cur_node)
		if(cur_node.left):
			queue.append(cur_node.left)
		if(cur_node.right):
			queue.append(cur_node.right)
		# print(cur_node.number)

	for n in leaf_nodes:
		# print(f'leaf node label : {n.label}')
		for k in [HAM,SPAM]:
			for p in n.pairs[k]:
				# print(p)
				if(p['expected_label']!=n.label):
					return False
	# print("**********************************")

	return True

def calculate_gini(node, the_fix):

	# print("pairs:")
	# print(node.pairs)
	sign=None
	# the_fix_words, llabel, rlabel = the_fix

	candidate_new_pred_node = PredicateNode(number=1,pred=KeywordPredicate(keywords=[the_fix]))
	
	right_leaf_spam_cnt=0
	right_leaf_ham_cnt=0
	left_leaf_spam_cnt=0
	left_leaf_ham_cnt=0

	for k in [SPAM, HAM]:
		for p in node.pairs[k]:
			if(candidate_new_pred_node.pred.evaluate(p)):
				if(k==SPAM):
					right_leaf_spam_cnt+=1
				else:
					right_leaf_ham_cnt+=1
			else:
				if(k==SPAM):
					left_leaf_spam_cnt+=1
				else:
					left_leaf_ham_cnt+=1

	# reverse_condition = False
	if((left_leaf_spam_cnt+left_leaf_ham_cnt)==0):
		# logger.debug("node:")
		# logger.debug(node)
		# pdb.set_trace()
		for k in [SPAM, HAM]:
			for p in node.pairs[k]:
				if(candidate_new_pred_node.pred.evaluate(p)):
					if(k==SPAM):
						right_leaf_spam_cnt+=1
					else:
						right_leaf_ham_cnt+=1
				else:
					if(k==SPAM):
						left_leaf_spam_cnt+=1
					else:
						left_leaf_ham_cnt+=1
	left_spam_rate=(left_leaf_spam_cnt)/(left_leaf_spam_cnt+left_leaf_ham_cnt)
	right_spam_rate=(right_leaf_spam_cnt)/(right_leaf_ham_cnt+right_leaf_spam_cnt)

	# if(left_spam_rate>right_spam_rate):
	# 	if(llabel!=SPAM)

	# if(left_spam_rate > right_spam_rate and llabel!=SPAM):
	# 	# pdb.set_trace()
	# 	reverse_condition=True

	left_total_cnt = left_leaf_spam_cnt+left_leaf_ham_cnt
	right_total_cnt = right_leaf_spam_cnt+right_leaf_ham_cnt

	total_cnt=left_total_cnt+right_total_cnt

	gini_impurity = (left_total_cnt/total_cnt)*(1-((left_leaf_spam_cnt/left_total_cnt)**2+(left_leaf_ham_cnt/left_total_cnt)**2))+ \
	(right_total_cnt/total_cnt)*(1-((right_leaf_spam_cnt/right_total_cnt)**2+(right_leaf_ham_cnt/right_total_cnt)**2))

	# print(f"gini_impurity for {the_fix} using {the_fix}: {gini_impurity}, reverse:{reverse_condition}\n")
	
	return gini_impurity

def fix_violations(treerule, repair_config, leaf_nodes):
	# print(f"leaf_nodes")
	if(repair_config.strategy=='naive'):
		# initialize the queue to work with
		queue = deque([])
		for ln in leaf_nodes:
			queue.append(ln)
		# print(queue)
		# print(len(queue))

		while(queue):
			# print(f'queue size={len(queue)}')
			node = queue.popleft()
			# print(f'node.pairs: HAM={len(node.pairs[HAM])}, SPAM={len(node.pairs[SPAM])},')
			new_parent_node=None
			# need to find a pair of violations that get the different label
			# in order to differentiate them
			if(node.label==ABSTAIN):
				continue
			if(node.pairs[SPAM] and node.pairs[HAM]):
				found=False
				the_fix=None
				for si in range(len(node.pairs[SPAM])):
					if(found):
						break
					for hi in range(len(node.pairs[HAM])):		
						the_fix = find_available_repair(node.pairs[SPAM][si],
						 node.pairs[HAM][hi], node.used_predicates)
						if(the_fix):
							found=True
							break
				# print(f"the fix number={i}")
				# print(the_fix)
				new_parent_node=redistribute_after_fix(treerule, node, the_fix)

			# handle the left and right child after redistribution
			else:
				if(node.pairs[SPAM]):
					if(node.label!=SPAM):
						node.label=SPAM
						node.is_reversed=True
						treerule.reversed_cnt+=1
						treerule.setsize(treerule.size+2)
				elif(node.pairs[HAM]):
					if(node.label!=HAM):
						node.label=HAM
						node.is_reversed=True
						treerule.reversed_cnt+=1
						treerule.setsize(treerule.size+2)
				else:
					continue
				# if(check_tree_purity(treerule)):
				# 	return treerule

			if(new_parent_node):
				still_inpure=False
				for k in [SPAM,HAM]:
					if(still_inpure):
						break
					for p in new_parent_node.left.pairs[k]:
						if(p['expected_label']!=new_parent_node.left.label):
							queue.append(new_parent_node.left)
							still_inpure=True
							break
				still_inpure=False          
				for k in [SPAM,HAM]:
					if(still_inpure):
						break
					for p in new_parent_node.right.pairs[k]:
							if(p['expected_label']!=new_parent_node.right.label):
								queue.append(new_parent_node.right)
								still_inpure=True
								break
				# print(queue)
				treerule.setsize(treerule.size+2)
		return treerule

	elif(repair_config.strategy=='information_gain'):
		# new implementation
		# 1. ignore the label of nodes at first
		# calculate the gini index of the split
		# and choose the best one as the solution
		# then based on the resulted majority expected labels
		# to assign the label for the children, if left get dirty
		# we flip the condition to preserve the ideal dc structure'
		# 2. during step 1, keep track of the used predicates to avoid
		# redundant repetitions
		queue = deque([])
		for ln in leaf_nodes:
			queue.append(ln)
		# print(queue)
		while(queue):
			node = queue.popleft()
			new_parent_node=None
			# need to find a pair of violations that get the different label
			# in order to differentiate them
			min_gini=1
			best_fix = None
			# reverse_condition=False

			if(node.label==ABSTAIN):
				continue
			if(node.pairs[SPAM] and node.pairs[HAM]):
				# need to examine all possible pair combinations
				considered_fixes = set()
				for pair in list(product(node.pairs[SPAM], node.pairs[HAM])):
					the_fixes = find_available_repair(pair[0],
					 pair[1], node.used_predicates,
					 all_possible=True)
					for f in the_fixes:
						if(f in considered_fixes):
							continue
						gini =calculate_gini(node, f)
						considered_fixes.add(f)
						if(gini<min_gini):
							min_gini=gini
							best_fix=f
							# reverse_condition=reverse_cond
				if(best_fix):
					# if(reverse_condition):
						# pdb.set_trace()
					new_parent_node=redistribute_after_fix(treerule, node, best_fix)
			# handle the left and right child after redistribution
			else:
				if(node.pairs[SPAM]):
					if(node.label!=SPAM):
						node.label=SPAM
						node.is_reversed=True
						treerule.reversed_cnt+=1
						treerule.setsize(treerule.size+2)
				elif(node.pairs[HAM]):
					if(node.label!=HAM):
						node.label=HAM
						node.is_reversed=True
						treerule.reversed_cnt+=1
						treerule.setsize(treerule.size+2)
				else:
					continue
				# if(check_tree_purity(treerule)):

			if(new_parent_node):
				still_inpure=False
				for k in [HAM,SPAM]:
					if(still_inpure):
						break
					for p in new_parent_node.left.pairs[k]:
						if(p['expected_label']!=new_parent_node.left.label):
							queue.append(new_parent_node.left)
							still_inpure=True
							break
				still_inpure=False          
				for k in [SPAM,HAM]:
					if(still_inpure):
						break
					for p in new_parent_node.right.pairs[k]:
							if(p['expected_label']!=new_parent_node.right.label):
								queue.append(new_parent_node.right)
								still_inpure=True
								break
				treerule.setsize(treerule.size+2)
				# print(f"after fix, treerule size is {treerule.size}")

		return treerule

	elif(repair_config.strategy=='optimal'):
		# 1. create a queue with tree nodes
		# 2. need to deepcopy the tree in order to enumerate all possible trees
		# logger.debug("leaf_nodes:")
		# logger.debug(leaf_nodes)
		queue = deque([])
		for ln in leaf_nodes:
			queue.append(ln.number)
		# print(f"number of leaf_nodes: {len(queue)}")
		# print("queue")
		# print(queue)
		cur_fixed_tree = treerule
		while(queue):
			sub_root_number = queue.popleft()
			subqueue=deque([(cur_fixed_tree, sub_root_number, sub_root_number)])
			# triples are needed here, since: we need to keep track of the 
			# updated(if so) subtree root in order to check purity from that node
			# print(f"subqueue: {subqueue}")
			sub_node_pure=False
			while(subqueue and not sub_node_pure):
				logger.debug(f"len of subqueue: {len(subqueue)}")
				prev_tree, leaf_node_number, subtree_root_number = subqueue.popleft()
				node = locate_node(prev_tree, leaf_node_number)
				if(node.label==ABSTAIN):
					continue
				if(node.pairs[HAM] and node.pairs[SPAM]):
					# need to examine all possible pair combinations
					considered_fixes = set()
					# print(list(product(node.pairs[CLEAN], node.pairs[DIRTY])))
					for pair in list(product(node.pairs[HAM], node.pairs[SPAM])):
						if(sub_node_pure):
							break
						fixes = find_available_repair(pair[0],pair[1], node.used_predicates,all_possible=True)
						valid_fixes = [x for x in fixes if x]
						if(not valid_fixes):
							continue
						else:
							for f in valid_fixes:
								new_parent_node=None
								if(f in considered_fixes):
									continue
								considered_fixes.add(f)
								new_tree = copy.deepcopy(prev_tree)
								node = locate_node(new_tree, node.number)
								new_parent_node = redistribute_after_fix(new_tree, node, f)
								if(leaf_node_number==sub_root_number):
									# first time replacing subtree root, 
									# the node number will change so we need 
									# to replace it
									subtree_root_number=new_parent_node.number
									# print(f"subtree_root_number is being updated to {new_parent_node.number}")
								new_tree.setsize(new_tree.size+2)
								if(check_tree_purity(new_tree, subtree_root_number)):
									# print("done with this leaf node, the fixed tree is updated to")
									cur_fixed_tree = new_tree
									# print(cur_fixed_tree)
									sub_node_pure=True
									break
								# else:
								#     print("not pure yet, need to enqueue")
								# handle the left and right child after redistribution
								still_inpure=False
								for k in [HAM,SPAM]:
									if(still_inpure):
										break
									for p in new_parent_node.left.pairs[k]:
										if(p['expected_label']!=new_parent_node.left.label):
											# print("enqueued")
											# print("current_queue: ")
											new_tree = copy.deepcopy(new_tree)
											parent_node = locate_node(new_tree, new_parent_node.number)
											subqueue.append((new_tree, parent_node.left.number, subtree_root_number))
											# print(subqueue)
											still_inpure=True
											break
								still_inpure=False          
								for k in [HAM,SPAM]:
									if(still_inpure):
										break
									for p in new_parent_node.right.pairs[k]:
										if(p['expected_label']!=new_parent_node.right.label):
											# print("enqueued")
											# print("current_queue: ")
											new_tree = copy.deepcopy(new_tree)
											parent_node = locate_node(new_tree, new_parent_node.number)
											# new_parent_node=redistribute_after_fix(new_tree, new_node, f)
											subqueue.append((new_tree, parent_node.right.number, subtree_root_number))
											# print(subqueue)
											still_inpure=True
											break
								# print('\n')
				else:
					# print("just need to reverse node condition")
					if(node.pairs[SPAM]):
						if(node.label!=SPAM):
							node.label=SPAM
							node.is_reversed=True
							treerule.reversed_cnt+=1
							prev_tree.setsize(prev_tree.size+2)
					elif(node.pairs[HAM]):
						if(node.label!=HAM):
							node.label=HAM
							node.is_reversed=True
							treerule.reversed_cnt+=1
							prev_tree.setsize(prev_tree.size+2)
					# if(check_tree_purity(prev_tree,subtree_root_number)):                
					# 	# print("done with this leaf node, the fixed tree is updated to")
					# 	cur_fixed_tree = prev_tree
					# 	sub_node_pure=True
					# 	# print(cur_fixed_tree)
					# 	break
				# print(f"current queue size: {len(queue)}")
		print("fixed all, return the fixed tree")
		print(cur_fixed_tree)
		return cur_fixed_tree 
	else:
		print("not a valid repair option")
		exit()

def calculate_retrained_results(complaints, new_wrongs_df, file_dir):
	the_complaints = complaints[complaints['expected_label']!=complaints['model_pred']]
	the_confirmatons = complaints[complaints['expected_label']==complaints['model_pred']]
	still_wrongs = pd.merge(the_complaints, new_wrongs_df, left_on='cid', right_on='cid', how='inner')
	# still_wrongs.to_csv(file_dir+'_still_wrongs.csv', index=False)
	not_correct_anymores = pd.merge(the_confirmatons, new_wrongs_df, left_on='cid', right_on='cid', how='inner')
	# print("complaints")
	# print(complaints)
	# print("new_wrongs")
	# print(new_wrongs_df)
	# new_wrongs_df.to_csv('new_wrongs.csv', index=False)
	# print('still wrongs')
	# print(still_wrongs)

	complaint_fix_rate=confirm_preserve_rate=1
	if(len(the_complaints)>0):
		complaint_fix_rate=1-len(still_wrongs)/len(the_complaints)
	if(len(the_confirmatons)>0):
		confirm_preserve_rate=1-len(not_correct_anymores)/len(the_confirmatons)

	return complaint_fix_rate, confirm_preserve_rate

def fix_rules(repair_config, fix_book_keeping_dict, conn, return_after_percent, deletion_factor, 
	current_start_id_pos, sorted_rule_ids, deletion_type, deletion_absolute_threshold):
	all_fixed_rules = []
	cur_fixed_rules = []
	all_rules_cnt=len(fix_book_keeping_dict)

	# domain_value_dict = construct_domain_dict(conn, table_name=table_name)
	# fix_book_keeping_dict = {k.id:{'rule':k} for k in original_rules}
	# print(domain_value_dict)
	fixed_run_cnt=0

	if(current_start_id_pos+floor(all_rules_cnt*return_after_percent)>=all_rules_cnt):
		stop_at_id_pos=all_rules_cnt-1
	else:
		stop_at_id_pos=current_start_id_pos+floor(all_rules_cnt*return_after_percent)-1
	# print("fix_book_keeping_dict")
	# print(fix_book_keeping_dict)
	# print(f"current_start_id:{sorted_rule_ids[current_start_id_pos]}")
	# print(f"stop_at_id: {sorted_rule_ids[stop_at_id_pos]}")

	while(current_start_id_pos<=stop_at_id_pos):
		treerule=fix_book_keeping_dict[sorted_rule_ids[current_start_id_pos]]['rule']
		# pdb.set_trace()
		# for treerule in rules:
		leaf_nodes = []
		for i, c in repair_config.complaints.iterrows():
			# print("the complaint is")
			# print(c.text)
			# print(type(c.text))
			leaf_node_with_complaints = populate_violations(treerule, c)
			if(leaf_node_with_complaints not in leaf_nodes):
				# if node is already in leaf nodes, dont
				# need to add it again
				leaf_nodes.append(leaf_node_with_complaints)
		if(leaf_nodes):
			# its possible for certain rule we dont have any violations
			# if(sorted_rule_ids[current_start_id_pos]==6):
			# 	import pudb; pudb.set_trace()
			fixed_treerule = fix_violations(treerule, repair_config, leaf_nodes)
			fix_book_keeping_dict[sorted_rule_ids[current_start_id_pos]]['rule']=fixed_treerule
			# print(fixed_treerule)
			fix_book_keeping_dict[treerule.id]['after_fix_size']=fixed_treerule.size
			fixed_treerule_text = fixed_treerule.serialize()
			fix_book_keeping_dict[treerule.id]['fixed_treerule_text']=fixed_treerule_text
		else:
			fix_book_keeping_dict[sorted_rule_ids[current_start_id_pos]]['rule']=treerule
			fix_book_keeping_dict[treerule.id]['after_fix_size']=treerule.size
			fixed_treerule_text = treerule.serialize()
			fix_book_keeping_dict[treerule.id]['fixed_treerule_text']=fixed_treerule_text
			fixed_treerule=treerule
		# pdb.set_trace()
		if(deletion_type=='ratio'):
			if(fixed_treerule.size/fix_book_keeping_dict[treerule.id]['pre_fix_size']*deletion_factor>=1):
				fix_book_keeping_dict[treerule.id]['deleted']=True
			else:
				fix_book_keeping_dict[treerule.id]['deleted']=False
		elif(deletion_type=='absolute'):
			if(fixed_treerule.size-fix_book_keeping_dict[treerule.id]['pre_fix_size']>deletion_absolute_threshold):
				fix_book_keeping_dict[treerule.id]['deleted']=True
			else:
				fix_book_keeping_dict[treerule.id]['deleted']=False
		else:
			logger.critical("not a valid deletion type")
			exit()
		current_start_id_pos+=1

	return stop_at_id_pos



# Function to apply stemming to a sentence
def stem_sentence(sentence, stemmer):
    words = word_tokenize(sentence)
    stemmed_words = [stemmer.stem(word) for word in words]
    return stemmed_words



def run_snorkel(lf_input, LFs=None):
	conn = lf_input.connection
	# logger.critical(LFs)
	sentences_df=pd.read_sql(f'SELECT * FROM {lf_input.dataset_name}', conn)
	if(lf_input.run_gpt_rules):
		stemmer = PorterStemmer()
		sentences_df['stems'] = sentences_df['content'].apply(lambda x: stem_sentence(x, stemmer))

		logger.debug("created stems for sentences")
	# logger.critical(sentences_df.head())
	sentences_df = sentences_df.rename(columns={"class": "expected_label", "content": "old_text"})
	sentences_df['text'] = sentences_df['old_text'].apply(lambda s: clean_text(s))
	sentences_df = sentences_df[~sentences_df['text'].isna()]

	lf_internal_args = lf_input_internal(funcs=LFs)

	# Snorkel built-in labelling function applier
	applier = PandasLFApplier(lfs=lf_internal_args.funcs)

	# Apply the labelling functions to get vectors
	initial_vectors = applier.apply(df=sentences_df, progress_bar=False)
	# logger.critical(initial_vectors)
	# df.to_csv('snorkel.csv')

	if(lf_input.training_model_type=='majority'):
		# pdb.set_trace()
		model = MajorityLabelVoter()
	else:
		model = LabelModel(cardinality=2, verbose=True, device='cpu')
		model.fit(L_train=initial_vectors, n_epochs=500, log_freq=100, seed=123)
		# snorkel needs to get an estimator using fit function first
		# training model with all labelling functions
	
	# pdb.set_trace()
	preds_probs= model.predict_proba(L=initial_vectors)

	df_sentences_filtered, probs_test_filtered, filtered_vectors = filter_unlabeled_dataframe(
			X=sentences_df, y=preds_probs, L=initial_vectors
	)	
	
	# pdb.set_trace()
	df_sentences_filtered = df_sentences_filtered.reset_index(drop=True)
	prob_diffs = [abs(t[0]-t[1]) for t in probs_test_filtered]
	prob_diffs_tuples = [(t[0],t[1]) for t in probs_test_filtered]
	df_sentences_filtered['model_pred_diff'] = pd.Series(prob_diffs)
	df_sentences_filtered['model_pred_prob_tuple'] = pd.Series(prob_diffs_tuples)
	df_sentences_filtered['model_pred'] = pd.Series(model.predict(L=filtered_vectors))
	cached_vectors = dict(zip(LFs, np.transpose(filtered_vectors)))
	lf_internal_args.func_vectors = cached_vectors
	df_sentences_filtered['vectors'] = pd.Series([",".join(map(str, t)) for t in filtered_vectors])
	lf_internal_args.filtered_sentences_df = df_sentences_filtered
	df_sentences_filtered.to_csv('result.csv')
	# the wrong labels we get
	wrong_preds = df_sentences_filtered[(df_sentences_filtered['expected_label']!=df_sentences_filtered['model_pred'])]
	# df_sentences_filtered.to_csv('predictions_shakira.csv', index=False)
	wrong_preds['signal_strength'] = wrong_preds['vectors'].apply(lambda s: sum([1 for i in s.split(",") if int(i) == SPAM or int(i)==HAM]))
	wrong_preds = wrong_preds.sort_values(['signal_strength'], ascending=False)
	# logger.critical(wrong_preds)
	accuracy=(len(df_sentences_filtered)-len(wrong_preds))/len(df_sentences_filtered)

	logger.critical(f"""
		out of {len(sentences_df)} sentences, {len(df_sentences_filtered)} actually got at least one signal to \n
		make prediction. Out of all the valid predictions, we have {len(wrong_preds)} wrong predictions, \n
		accuracy = {(len(df_sentences_filtered)-len(wrong_preds))/len(df_sentences_filtered)} 
		""")

	return accuracy, df_sentences_filtered, wrong_preds

def calculate_post_fix_tree_size(fix_book_keeping_dict, post_fix_num_funcs):
	before_total_size = after_total_size = 0
	for k,v in fix_book_keeping_dict.items():
		if(not v['deleted']):
			before_total_size+=v['pre_fix_size']
			after_total_size+=v['after_fix_size']
	avg_tree_size_increase=(after_total_size-before_total_size)/post_fix_num_funcs

	return avg_tree_size_increase, after_total_size-before_total_size


def lf_main(lf_input):

	conn = lf_input.connection
	user_input_size=lf_input.user_input_size
	complaint_ratio=lf_input.complaint_ratio
	strat=lf_input.strat
	lf_source=lf_input.lf_source
	number_of_funcs=lf_input.number_of_funcs
	experiment_name=lf_input.experiment_name
	repeatable=lf_input.repeatable
	rseed=lf_input.rseed
	run_intro=lf_input.run_intro
	run_amazon = lf_input.run_amazon
	run_gpt_rules=lf_input.run_gpt_rules
	run_painter = lf_input.run_painter
	run_professor= lf_input.run_professor
	run_imdb = lf_input.run_imdb
	run_photographer = lf_input.run_photographer
	run_physician= lf_input.run_physician
	run_spam=lf_input.run_spam
	run_tweets=lf_input.run_tweets
	run_dbpedia=lf_input.run_dbpedia
	run_fakenews= lf_input.run_fakenews
	run_plots = lf_input.run_plots
	run_yelp=lf_input.run_yelp
	run_agnews=lf_input.run_agnews
	gpt_dataset=lf_input.gpt_dataset
	retrain_after_percent=lf_input.retrain_every_percent
	deletion_factor=lf_input.deletion_factor
	retrain_accuracy_thresh=lf_input.retrain_accuracy_thresh
	load_funcs_from_pickle=lf_input.load_funcs_from_pickle
	pickle_file_name=lf_input.pickle_file_name
	seed_file=lf_input.seed_file
	pre_deletion_threshold=lf_input.pre_deletion_threshold
	deletion_absolute_threshold=lf_input.deletion_absolute_threshold
	deletion_type=lf_input.deletion_type
	gpt_pickled_rules_dir=lf_input.gpt_pickled_rules_dir

	# customized_complaints_file=lf_input.customized_complaints_file
	######
	logger.debug(f"lf_input:{lf_input}")

	#########
	current_fixed_percent=0
	fix_rate=0
	rbbm_runtime=0
	bbox_runtime=0
	current_start_id_pos=prev_stop_tree_id_pos=0
	new_global_accuracy=0
	confirm_preserve_rate=0
	new_signaled_cnt=0
	post_fix_num_funcs=0
	new_all_sentences_df=None
	num_of_funcs_processed_by_algo=0

	try:
		log_map = { 'debug': logging.DEBUG,
		'info': logging.INFO,
		'warning': logging.WARNING,
		'error': logging.ERROR,
		'critical': logging.CRITICAL
		}
		print(logconfig.root)
		logconfig.root.setLevel(log_map[lf_input.log_level])
		print(lf_input.log_level)
		print(logconfig.root)
	except KeyError as e:
		print('no such log level')

	timestamp = datetime.now()
	timestamp_str = timestamp.strftime('%Y%m%d%H%M%S')
	result_dir = f'./{lf_input.experiment_name}_{timestamp_str}'
	if not os.path.exists(result_dir):
		os.makedirs(result_dir)

	
	if(lf_source=='intro' or run_intro):
		tree_rules=gen_example_funcs()
	elif(run_amazon is True):
		logger.debug("amazon!")
		tree_rules=gen_amazon_funcs()
	elif(run_painter is True):
		logger.debug("run_painter!")
		tree_rules=gen_painter_architecht_funcs()

	elif(run_professor is True):
		logger.debug("run_professor!")
		tree_rules=gen_professor_teacher_funcs()

	elif(run_imdb is True):
		logger.debug("run_imdb!")
		tree_rules=gen_imdb_funcs()

	elif(run_photographer is True):
		logger.debug("rum photographer!")
		tree_rules=gen_pj_funcs()

	elif(run_physician is True):
		logger.debug("rum physician!")
		tree_rules = gen_pp_funcs()
	elif(run_spam is True):
		logger.debug("rum spam!")
		tree_rules = gen_spam_funcs()
	elif(run_tweets is True):
		logger.debug("rum tweets!")
		tree_rules = gen_tweets_funcs()
	elif(run_agnews is True):
		logger.debug("rum agnews!")
		tree_rules=gen_agnews_funcs()

	elif(run_plots is True):
		logger.debug("rum plots!")
		tree_rules=gen_plots_funcs()
	
	elif(run_yelp is True):
		logger.debug("rum yelp!")
		tree_rules=gen_yelp_funcs()
	
	elif(run_fakenews is True):
		logger.debug("rum fakenews!")
		tree_rules=gen_fakenews_funcs()
	
	elif(run_dbpedia is True):
		logger.debug("rum dbpedia!")
		tree_rules=gen_dbpedia_funcs()
	
	elif(load_funcs_from_pickle=='true'):
		with open(f'{pickle_file_name}.pkl', 'rb') as file:
			tree_rules = pickle.load(file)
			logger.debug('loaded pickled funcs')
			logger.debug(f'we have {len(tree_rules)} funcs')
	elif(run_gpt_rules is True):
		with open(f'{gpt_pickled_rules_dir}{gpt_dataset}_gpt_rules.pkl', 'rb') as file:
			tree_rules = pickle.load(file)
			logger.debug('loaded pickled funcs')
			logger.debug(f'we have {len(tree_rules)} funcs')
	else:
		sentences_df=pd.read_sql(f'SELECT * FROM {lf_input.dataset_name}', conn)
		sentences_df = sentences_df.rename(columns={"class": "expected_label", "content": "old_text"})
		sentences_df['text'] = sentences_df['old_text'].apply(lambda s: clean_text(s))
		sentences_df = sentences_df[~sentences_df['text'].isna()]
		kwm = KeyWordRuleMiner(sentences_df)
		tree_rules = kwm.gen_funcs(number_of_funcs, 0.3)

	labelling_funcs=[f.gen_label_rule() for f in tree_rules]
	num_funcs=len(labelling_funcs)
	logger.debug(f'number of functions: {num_funcs}')
	bbox_start = time.time()
	global_accuracy, all_sentences_df, wrongs_df = run_snorkel(lf_input, LFs=labelling_funcs)
	bbox_end = time.time()
	bbox_runtime+=round(bbox_end-bbox_start,3)

	# all_sentences_df=all_sentences_df.sort_values(by=['text'])
	old_signaled_cnt=len(all_sentences_df)
	# wrongs_df.to_csv('initial_wrongs.csv', index=False)
	wrong_hams=wrongs_df[wrongs_df['expected_label']==HAM]
	wrong_spams=wrongs_df[wrongs_df['expected_label']==SPAM]
	# print(f"wrong_hams count: {len(wrong_hams)}")
	# print(f"wrong_spams count: {len(wrong_spams)}")

	rs = None
	# new_seed = int.from_bytes(os.urandom(4), byteorder="big")
	logger.debug(f"repeatable?: {repeatable}")
	random.seed()
	
	if(repeatable=='true'):
		rs = rseed
	else:
		rs = random.randint(1, 10000)
		logger.debug(f"seed: {rs}")
		if(not os.path.exists(seed_file)):
			with open(seed_file, 'w') as file:
				file.write(f'seed: {rs}\n')
		else:
			with open(seed_file, 'a') as file:
				file.write(f'seed: {rs}\n')

	logger.debug(f"random_seed: {rs}")
	logger.debug(f"user_input_size: {user_input_size}, strat:{strat}")
	# tree_rules = [f1, f2, f3]

	if(run_intro=='true'):
		# intro_input_cids = []
		# wrong_ids = random.sample(wrong_check_ids, 5)
		# correct_ids = random.sample(correct_check_ids, 5)
		# intro_input_cids.extend(wrong_ids)
		# intro_input_cids.extend(correct_ids)
		with open('intro_complaint_ids', 'r') as file:
		    intro_input_cids = [int(x) for x in file.readline().strip().split(',')]
		sampled_complaints=all_sentences_df[all_sentences_df['cid'].isin(intro_input_cids)]
		# print("sampled_complaints")
		# print(sampled_complaints)
		# print("intro_input_cids")
		# print(intro_input_cids)
		# exit()
		num_complaints=5
		num_confirm=5
	else:
		all_wrongs=all_sentences_df[all_sentences_df['expected_label']!=all_sentences_df['model_pred']]
		all_wrongs=all_wrongs.sort_values('cid')
		all_confirms=all_sentences_df[all_sentences_df['expected_label']==all_sentences_df['model_pred']]
		all_confirms=all_confirms.sort_values('cid')
		complaint_reached_max=False

		if(lf_input.user_input_sample_strat=='percentage'):
			logger.debug("percentage user input strat")
			# wrong_sample_size = floor(len(all_wrongs)*lf_input.user_input_percentage)
			wrong_sample_size = min(floor(len(all_wrongs)*lf_input.user_input_percentage),150)
			logger.debug(f"complaint size: {wrong_sample_size}")
		else:
			wrong_sample_size=floor(user_input_size*complaint_ratio)
		if(wrong_sample_size>=len(all_wrongs)):
			complaint_reached_max=True
			wrong_sample_size=len(all_wrongs)
		if(lf_input.user_input_strat=='active_learning'):
			all_wrongs=all_wrongs.sort_values('model_pred_diff')
			sampled_wrongs=all_wrongs.head(n=wrong_sample_size)
		else:
			sampled_wrongs=all_wrongs.sample(n=wrong_sample_size,random_state=rs)
		
		confirm_reached_max=False
		if(lf_input.user_input_sample_strat=='percentage'):
			logger.debug("percentage user input strat")
			confirm_sample_size = min(floor(len(all_confirms)*lf_input.user_input_percentage),150)
			logger.debug(f"confirm size: {confirm_sample_size}")
		else:
			confirm_sample_size=user_input_size-wrong_sample_size

		if(confirm_sample_size>=len(all_confirms)):
			confirm_reached_max=True
			confirm_sample_size=len(all_confirms)
		if(lf_input.user_input_strat=='active_learning'):
			all_confirms=all_confirms.sort_values('model_pred_diff')
			# pdb.set_trace()
			sampled_confirms=all_confirms.head(n=confirm_sample_size)
		else:
			sampled_confirms=all_confirms.sample(n=confirm_sample_size, random_state=rs)
		
		sampled_complaints=pd.concat([sampled_wrongs, sampled_confirms])
		num_complaints=len(sampled_wrongs)
		num_confirm=len(sampled_confirms)


	sampled_complaints['id'] = sampled_complaints.reset_index().index

	stimestamp = datetime.now()
	rc = RepairConfig(strategy=strat, complaints=sampled_complaints, acc_threshold=0.8, runtime=0, deletion_factor=deletion_factor)

	# print(f"fix_rate:{fix_rate}, retrain_accuracy_thresh:{retrain_accuracy_thresh}")
	fix_book_keeping_dict = {k.id:{'rule':k, 'deleted':False, 'pre_fix_size':k.size, 'after_fix_size':k.size, 'pre-deleted': False} for k in tree_rules}
		# fix_book_keeping_dict[treerule.id]['pre_fix_size']=treerule.size

	for f in tree_rules:
		applier = PandasLFApplier(lfs=[f.gen_label_rule()])
		initial_vectors = applier.apply(df=all_sentences_df, progress_bar=False)
		func_results = [x[0] for x in list(initial_vectors)]
		non_abstain_results_cnt=len([x for x in func_results if x!=ABSTAIN])
		gts = all_sentences_df['expected_label'].values.tolist()
		match_cnt = len([x for x,y in zip(func_results,gts) if (x == y and x!=ABSTAIN)])
		if(non_abstain_results_cnt==0):
			fix_book_keeping_dict[f.id]['pre-deleted']=True
			# pdb.set_trace()
		elif(match_cnt/non_abstain_results_cnt<pre_deletion_threshold):
			fix_book_keeping_dict[f.id]['pre-deleted']=True

	predeleted_book_keep_dict = {k:v for k,v in fix_book_keeping_dict.items() if v['pre-deleted']==True}
	new_fix_book_keeping_dict = {k:v for k,v in fix_book_keeping_dict.items() if v['pre-deleted']==False}

	tree_ids=[k for k in new_fix_book_keeping_dict]
	tree_ids.sort()

	if(retrain_accuracy_thresh!=1):
		retrain_bookkeeping_dict = {'fix_rate':[], 'new_global_accuracy':[], 'avg_tree_size_increase':[], 
		'total_tree_size_increase':[],'confirm_preserve_rate':[], 'number_of_funcs_fixed':[]}
	else:
		retrain_bookkeeping_dict =None

	while(fix_rate<=retrain_accuracy_thresh):
		# pdb.set_trace()
		logger.debug(f'curren_fix_rate:{fix_rate}')
		logger.debug(f'retrain_accuracy_thresh: {retrain_accuracy_thresh}')
		logger.debug(f'current_start_id_pos:{current_start_id_pos}')
		logger.debug(f'prev_stop_tree_id_pos: {prev_stop_tree_id_pos}')
		logger.debug(f'\n')
		rbbm_start = time.time()
		prev_stop_tree_id_pos = fix_rules(repair_config=rc, fix_book_keeping_dict=new_fix_book_keeping_dict, conn=conn, 
			return_after_percent=retrain_after_percent, deletion_factor=deletion_factor, current_start_id_pos=current_start_id_pos, 
			sorted_rule_ids=tree_ids, deletion_type=deletion_type, deletion_absolute_threshold=deletion_absolute_threshold)
		tree_rules=[v['rule'] for k,v in new_fix_book_keeping_dict.items() if not v['deleted']]
		num_of_funcs_processed_by_algo+=(prev_stop_tree_id_pos-current_start_id_pos+1)
		current_start_id_pos=prev_stop_tree_id_pos+1
		post_fix_num_funcs=len([value for value in new_fix_book_keeping_dict.values() if not value['deleted']])
		rbbm_end = time.time()
		rbbm_runtime+=round(rbbm_end-rbbm_start,3)
		# pdb.set_trace()
		new_labelling_funcs = [f.gen_label_rule() for f in tree_rules]
		logger.critical(new_labelling_funcs)
		logger.critical(f"new_labelling_funcs len: {len(new_labelling_funcs)}")
		# logger.critical(new_fix_book_keeping_dict)

		bbox_start = time.time()
		new_global_accuracy, new_all_sentences_df, new_wrongs_df = run_snorkel(lf_input, LFs=new_labelling_funcs)
		bbox_end = time.time()
		bbox_runtime+=round(bbox_end-bbox_start,3)
		new_signaled_cnt=len(new_all_sentences_df)
		fix_rate, confirm_preserve_rate = calculate_retrained_results(sampled_complaints, new_wrongs_df, result_dir+'/'+timestamp_str)
		if(retrain_bookkeeping_dict):
			avg_tree_size_increase,total_tree_size_increase = calculate_post_fix_tree_size(fix_book_keeping_dict,post_fix_num_funcs)
			retrain_bookkeeping_dict['fix_rate'].append(fix_rate)
			retrain_bookkeeping_dict['new_global_accuracy'].append(new_global_accuracy)
			retrain_bookkeeping_dict['avg_tree_size_increase'].append(avg_tree_size_increase)
			retrain_bookkeeping_dict['total_tree_size_increase'].append(total_tree_size_increase)
			retrain_bookkeeping_dict['confirm_preserve_rate'].append(confirm_preserve_rate)
			retrain_bookkeeping_dict['number_of_funcs_fixed'].append(num_of_funcs_processed_by_algo)
			logger.critical("cur retrain_bookkeeping_dict")
			logger.critical(retrain_bookkeeping_dict)

		if(current_start_id_pos>=len(new_fix_book_keeping_dict)):
			logger.debug("break")
			break

	# avg_tree_size_increase,total_tree_size_increase = calculate_post_fix_tree_size(fix_book_keeping_dict,post_fix_num_funcs)

	if(retrain_bookkeeping_dict):
		retrain_bookkeeping_dict['fix_rate'].append(fix_rate)
		retrain_bookkeeping_dict['new_global_accuracy'].append(new_global_accuracy)
		retrain_bookkeeping_dict['avg_tree_size_increase'].append(avg_tree_size_increase)
		retrain_bookkeeping_dict['total_tree_size_increase'].append(total_tree_size_increase)
		retrain_bookkeeping_dict['confirm_preserve_rate'].append(confirm_preserve_rate)
		retrain_bookkeeping_dict['number_of_funcs_fixed'].append(num_of_funcs_processed_by_algo)

	logger.critical("cur retrain_bookkeeping_dict")

	before_total_size=after_total_size=0
	fix_book_keeping_dict = {**predeleted_book_keep_dict, **new_fix_book_keeping_dict}
	avg_tree_size_increase, total_tree_size_increase = calculate_post_fix_tree_size(fix_book_keeping_dict,post_fix_num_funcs)
	# print(f"avg tree_size increase: {avg_tree_size_increase}")

	if(not os.path.exists(result_dir+'/'+timestamp_str+'_experiment_stats')):
		with open(result_dir+'/'+timestamp_str+'_experiment_stats', 'w') as file:
			# Write some text to the file
			file.write('user_input_strat,strat,seed,pickle_file_name,table_name,timestamp_str,deletion_type,deletion_absolute_threshold,rbbm_runtime,bbox_runtime,avg_tree_size_increase,user_input_size,complaint_ratio,num_complaints,num_confirmations,global_accuracy,fix_rate,confirm_preserve_rate,new_global_accuracy,prev_signaled_cnt,new_signaled_cnt,' +\
				'num_functions,deletion_factor,post_fix_num_funcs,num_of_funcs_processed_by_algo,complaint_reached_max,confirm_reached_max,lf_source,retrain_after_percent,retrain_accuracy_thresh,load_funcs_from_pickle,pre_deletion_threshold\n')

	all_sentences_df.to_csv(result_dir+'/'+timestamp_str+'_initial_results.csv', index=False)
	sampled_complaints.to_csv(f"{result_dir}/sampled_user_input_{timestamp_str}_{strat}_{str(user_input_size)}.csv", index=False)

	for kt,vt in fix_book_keeping_dict.items():
		with open(f"{result_dir}/{timestamp_str}_tree_{strat}_{kt}_dot_file", 'a') as file:
			comments=f"// presize: {fix_book_keeping_dict[kt]['pre_fix_size']}, after_size: {fix_book_keeping_dict[kt]['after_fix_size']}, deleted: {fix_book_keeping_dict[kt]['deleted']} factor: {deletion_factor} reverse_cnt:{fix_book_keeping_dict[kt]['rule'].reversed_cnt}"
			dot_file=fix_book_keeping_dict[kt]['rule'].gen_dot_string(comments)
			file.write(dot_file)
		# print("dot string:")
		# print(dot_file)
	new_all_sentences_df.to_csv(result_dir+'/'+timestamp_str+'_after_fix_results.csv', index=False)
	with open(result_dir+'/'+timestamp_str+'_experiment_stats', 'a') as file:
		# Write the row to the file
		file.write(f'{lf_input.user_input_strat},{strat},{rs},{pickle_file_name},{lf_input.dataset_name},{timestamp_str},{deletion_type},{deletion_absolute_threshold},{rbbm_runtime},{bbox_runtime},{avg_tree_size_increase},{user_input_size},{complaint_ratio},{num_complaints},{num_confirm},{round(global_accuracy,3)},{round(fix_rate,3)},{round(confirm_preserve_rate,3)},'+\
			f'{round(new_global_accuracy,3)},{old_signaled_cnt},{new_signaled_cnt},{num_funcs},{deletion_factor},{post_fix_num_funcs},{num_of_funcs_processed_by_algo},{complaint_reached_max},{confirm_reached_max},{lf_source},'+\
			f'{retrain_after_percent},{retrain_accuracy_thresh},{load_funcs_from_pickle},{pre_deletion_threshold}\n')

	with open(result_dir+'/'+timestamp_str+'_fix_book_keeping_dict.pkl', 'wb') as file:
	    # Use pickle.dump() to write the dictionary to the file
	    pickle.dump(fix_book_keeping_dict, file)

	if(retrain_bookkeeping_dict):
		with open(result_dir+'/'+timestamp_str+'_retrain_bookkeeping_dict.pkl', 'wb') as file:
		    # Use pickle.dump() to write the dictionary to the file
		    pickle.dump(retrain_bookkeeping_dict, file)
