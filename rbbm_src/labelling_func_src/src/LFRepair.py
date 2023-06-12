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
)
import rbbm_src.logconfig
from math import floor
from rbbm_src.labelling_func_src.src.example_tree_rules import gen_example_funcs
from rbbm_src.labelling_func_src.src.KeyWordRuleMiner import KeyWordRuleMiner 
from itertools import product
from collections import deque,OrderedDict
from rbbm_src.classes import StatsTracker, FixMonitor, RepairConfig, lf_input
from rbbm_src.labelling_func_src.src.classes import lf_input_internal, clean_text, wrong_check_ids, correct_check_ids
from rbbm_src.labelling_func_src.src.experiment import lf_main
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
from TreeRules import textblob_sentiment
import logging
import random
import os
import argparse
import nltk
from nltk.corpus import stopwords
import copy

nltk.download('stopwords') 
stop_words = set(stopwords.words('english')) 

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

def redistribute_after_fix(tree_rule, node, the_fix, reverse=False):
	# there are some possible "side effects" after repair for a pair of violations
	# which is solving one pair can simutaneously fix some other pairs so we need 
	# to redistribute the pairs in newly added nodes if possible
	sign=None
	cur_number=tree_rule.max_node_id+1

	the_fix_words, llabel, rlabel = the_fix
	if(reverse):
		llabel, rlabel = rlabel, llabel
	new_predicate_node = PredicateNode(number=cur_number,pred=KeywordPredicate(keywords=[the_fix_words]))
	new_predicate_node.is_added=True
	cur_number+=1
	new_predicate_node.left= LabelNode(number=cur_number, label=llabel, pairs={HAM:[], SPAM:[]}, used_predicates=set([]))
	new_predicate_node.left.is_added=True
	cur_number+=1
	new_predicate_node.right=LabelNode(number=cur_number, label=rlabel, pairs={HAM:[], SPAM:[]}, used_predicates=set([]))
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
	cand=None
	for w in ham_available_words:
		# tuple cand has x elements: 
		# w: the word to differentate the 2 sentences
		# llabel: the left node label
		# rlabel: the right node label 
		# check if the predicate is already present
		# in the current constraint
		if(w.lower() not in stop_words):
			cand = (w, SPAM, HAM)
			if(cand not in used_predicates):
				if(not all_possible):
					return cand
				else:
					res.append(cand)

	for w in spam_available_words:
		# tuple cand has x elements: 
		# w: the word to differentate the 2 sentences
		# llabel: the left node label
		# rlabel: the right node label 
		# check if the predicate is already present
		# in the current constraint
		if(w.lower() not in stop_words):
			cand = (w, HAM, SPAM)
			if(cand not in used_predicates):
				if(not all_possible):
					return cand
				else:
					res.append(cand)
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
	# print("node:")
	# print(node)
	# print("pairs:")
	# print(node.pairs)
	sign=None
	the_fix_words, llabel, rlabel = the_fix

	candidate_new_pred_node = PredicateNode(number=1,pred=KeywordPredicate(keywords=[the_fix_words]))
	
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

	reverse_condition = False
	left_spam_rate=(left_leaf_spam_cnt)/(left_leaf_spam_cnt+left_leaf_ham_cnt)
	right_spam_rate=(right_leaf_spam_cnt)/(right_leaf_ham_cnt+right_leaf_spam_cnt)

	if(left_spam_rate > right_spam_rate and llabel!=SPAM):
		reverse_condition=True

	left_total_cnt = left_leaf_spam_cnt+left_leaf_ham_cnt
	right_total_cnt = right_leaf_spam_cnt+right_leaf_ham_cnt

	total_cnt=left_total_cnt+right_total_cnt

	gini_impurity = (left_total_cnt/total_cnt)*(1-((left_leaf_spam_cnt/left_total_cnt)**2+(left_leaf_ham_cnt/left_total_cnt)**2))+ \
	(right_total_cnt/total_cnt)*(1-((right_leaf_spam_cnt/right_total_cnt)**2+(right_leaf_ham_cnt/right_total_cnt)**2))

	# print(f"gini_impurity for {the_fix} using {the_fix}: {gini_impurity}, reverse:{reverse_condition}\n")
	
	return gini_impurity, reverse_condition

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
				if(check_tree_purity(treerule)):
					return treerule

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
			reverse_condition=False

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
						gini, reverse_cond =calculate_gini(node, f)
						considered_fixes.add(f)
						if(gini<min_gini):
							min_gini=gini
							best_fix=f
							reverse_condition=reverse_cond
				if(best_fix):
					new_parent_node=redistribute_after_fix(treerule, node, best_fix, reverse_condition)
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
				if(check_tree_purity(treerule)):
					return treerule

				# print('its not pure?')
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

	elif(repair_config.strategy=='brute_force'):
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
					if(check_tree_purity(prev_tree,subtree_root_number)):                
						# print("done with this leaf node, the fixed tree is updated to")
						cur_fixed_tree = prev_tree
						sub_node_pure=True
						# print(cur_fixed_tree)
						break
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
	still_wrongs = pd.merge(the_complaints, new_wrongs_df, left_on='comment_id', right_on='comment_id', how='inner')
	# still_wrongs.to_csv(file_dir+'_still_wrongs.csv', index=False)
	not_correct_anymores = pd.merge(the_confirmatons, new_wrongs_df, left_on='comment_id', right_on='comment_id', how='inner')
	print("complaints")
	print(complaints)
	print("new_wrongs")
	print(new_wrongs_df)
	# new_wrongs_df.to_csv('new_wrongs.csv', index=False)
	print('still wrongs')
	print(still_wrongs)

	complaint_fix_rate=confirm_preserve_rate=1
	if(len(the_complaints)>0):
		complaint_fix_rate=1-len(still_wrongs)/len(the_complaints)
	if(len(the_confirmatons)>0):
		confirm_preserve_rate=1-len(not_correct_anymores)/len(the_confirmatons)

	return complaint_fix_rate, confirm_preserve_rate

def fix_rules(repair_config, fix_book_keeping_dict, conn, return_after_percent, deletion_factor, current_start_id_pos, sorted_rule_ids):
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
		stop_at_id_pos=current_start_id_pos+floor(all_rules_cnt*return_after_percent)
	# print("fix_book_keeping_dict")
	# print(fix_book_keeping_dict)
	print(f"current_start_id:{sorted_rule_ids[current_start_id_pos]}")
	print(f"stop_at_id: {sorted_rule_ids[stop_at_id_pos]}")

	while(current_start_id_pos<=stop_at_id_pos):
		treerule=fix_book_keeping_dict[sorted_rule_ids[current_start_id_pos]]['rule']
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
			fix_book_keeping_dict[sorted_rule_ids[current_start_id_pos]]['rule']=tree_rule
			fix_book_keeping_dict[treerule.id]['after_fix_size']=treerule.size
			fixed_treerule_text = treerule.serialize()
			fix_book_keeping_dict[treerule.id]['fixed_treerule_text']=fixed_treerule_text
			fixed_treerule=tree_rule

		if(fixed_treerule.size/fix_book_keeping_dict[treerule.id]['pre_fix_size']*deletion_factor>=1):
			fix_book_keeping_dict[treerule.id]['deleted']=True
		else:
			fix_book_keeping_dict[treerule.id]['deleted']=False
		current_start_id_pos+=1

	return stop_at_id_pos


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Running experiments of RBBM')

	parser.add_argument('-d','--dbname', metavar="\b", type=str, default='label',
	  help='database name which stores the dataset, (default: %(default)s)')

	parser.add_argument('-P','--port', metavar="\b", type=int, default=5432,
	  help='database port, (default: %(default)s)')

	parser.add_argument('-p', '--password', metavar="\b", type=int, default=5432,
	  help='database password, (default: %(default)s)')

	parser.add_argument('-u', '--user', metavar="\b", type=str, default='postgres',
	  help='database user, (default: %(default)s)')

	parser.add_argument('-r','--repair_method', metavar="\b", type=str, default='information_gain',
	  help='method used to repair the rules (naive, information_gain, optimal) (default: %(default)s)')

	parser.add_argument('-U','--userinput_size', metavar="\b", type=int, default=40,
	  help='user input size (default: %(default)s)')

	parser.add_argument('-t','--complaint_ratio', metavar="\b", type=float, default=0.5,
	  help='out of the user input, what percentage of it is complaint? (the rest are confirmations) (default: %(default)s)')

	parser.add_argument('-f','--lf_source', metavar="\b", type=str, default='intro',
	  help='the source of labelling function (intro / system generate) (default: %(default)s)')

	parser.add_argument('-n','--number_of_funcs', metavar="\b", type=int, default=20,
	  help='if if_source is selected as system generate, how many do you want(default: %(default)s)')

	parser.add_argument('-e', '--experiment_name', metavar="\b", type=str, default='test_blah',
	  help='the name of the experiment, the results will be stored in the directory named with experiment_name_systime (default: %(default)s)')

	parser.add_argument('-R', '--repeatable', metavar="\b", type=bool, default=True,
	  help='repeatable? (default: %(default)s)')

	parser.add_argument('-s', '--seed', metavar="\b", type=int, default=123,
	  help='if repeatable, specify a seed number here (default: %(default)s)')

	parser.add_argument('-i', '--run_intro',  metavar="\b", type=bool, default=False,
	  help='do you want to run the intro example with pre selected user input? (default: %(default)s)')
	
	parser.add_argument('-D', '--deletion_factor',  metavar="\b", type=float, default=0.5,
	  help='this is a factor controlling how aggressive the algorithm chooses to delete the rule from the rulset (default: %(default)s)')
	# if a rule gets deleted is decided by comparing (new_size/old_size)*deletion_factor and 1, if (new_size/old_size)*deletion_factor<=1 
	# we keep the rule, other wise we delete
	parser.add_argument('-E', '--retrain_every_percent',  metavar="\b", type=float, default=1,
	  help='retrain over every (default: %(default)s*100), the default order is sorted by treesize ascendingly')

	parser.add_argument('-A', '--retrain_accuracy_thresh',  metavar="\b", type=float, default=0.5,
	  help='when retrain over every retrain_every_percent, the algorithm stops when the fix rate is over this threshold (default: %(default)s)')

	# parser.add_argument('-C', '--customized_complaints_file', metavar="\b", type=text, default='test',
	#   help='input file name which contains the cids of the complaints (mainly used for running example)(default: %(default)s)')

	args = parser.parse_args()

	#########
	conn = psycopg2.connect(dbname=args.dbname, user=args.user, password=args.password)
	sample_size=args.userinput_size
	complaint_ratio=args.complaint_ratio
	strat=args.repair_method
	lf_source=args.lf_source
	number_of_funcs=args.number_of_funcs
	experiment_name=args.experiment_name
	repeatable=args.repeatable
	rseed=args.seed
	run_intro=args.run_intro
	retrain_after_percent=args.retrain_every_percent
	deletion_factor=args.deletion_factor
	retrain_accuracy_thresh=args.retrain_accuracy_thresh
	# customized_complaints_file=args.customized_complaints_file
	######
	print(args)


	timestamp = datetime.now()
	timestamp_str = timestamp.strftime('%Y%m%d%H%M%S')
	result_dir = f'./{args.experiment_name}'
	if not os.path.exists(result_dir):
		os.makedirs(result_dir)

	if(lf_source=='intro' or run_intro):
		tree_rules=gen_example_funcs()
	else:
		sentences_df=pd.read_sql(f'SELECT * FROM youtube', conn)
		sentences_df = sentences_df.rename(columns={"class": "expected_label", "content": "old_text"})
		sentences_df['text'] = sentences_df['old_text'].apply(lambda s: clean_text(s))
		sentences_df = sentences_df[~sentences_df['text'].isna()]
		kwm = KeyWordRuleMiner(sentences_df)
		tree_rules = kwm.gen_funcs(number_of_funcs, 0.3)


	labelling_funcs=[f.gen_label_rule() for f in tree_rules]
	li =lf_input(
		connection=conn,
		contingency_size_threshold=1,
		contingency_sample_times=1,
		clustering_responsibility=False,
		sample_contingency=False,
		log_level='debug',
		user_provide=True,
		training_model_type='snorkel',
		word_threshold=3,
		greedy=True,
		cardinality_thresh=2,
		using_lattice=True,
		eval_mode='single_func',
		# lattice: bool
		invoke_type='terminal', # is it from 'terminal' or 'notebook'
		arg_str=None, # only used if invoke_type='notebook'
		# lattice_dict:dict
		# lfs:List[lfunc]
		# sentences_df:pd.core.frame.DataFrame
		topk=3, # topk value for number of lfs when doing responsibility generation
		random_number_for_complaint=5,
		dataset_name='youtube',
		stats=StatsTracker(),
		prune_only=True,
		return_complaint_and_results=True
		)
	global_accuracy, all_sentences_df, wrongs_df = lf_main(li, LFs=labelling_funcs)
	# all_sentences_df=all_sentences_df.sort_values(by=['text'])
	old_signaled_cnt=len(all_sentences_df)
	# wrongs_df.to_csv('initial_wrongs.csv', index=False)
	wrong_hams=wrongs_df[wrongs_df['expected_label']==HAM]
	wrong_spams=wrongs_df[wrongs_df['expected_label']==SPAM]
	print(f"wrong_hams count: {len(wrong_hams)}")
	print(f"wrong_spams count: {len(wrong_spams)}")

	rs = None
	new_seed = int.from_bytes(os.urandom(4), byteorder="big")
	random.seed(new_seed)
	if(repeatable):
		rs = rseed
	else:
		rs = random.randint(1, 1000)

	print(f"random_seed: {rs}")
	print(f"size: {sample_size}, strat:{strat}")
	# tree_rules = [f1, f2, f3]

	if(run_intro):
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
		all_confirms=all_sentences_df[all_sentences_df['expected_label']==all_sentences_df['model_pred']]
		wrong_sample_size=floor(sample_size*complaint_ratio)
		sampled_wrongs=all_wrongs.sample(n=wrong_sample_size,random_state=rs)
		sampled_confirms=all_confirms.sample(n=sample_size-wrong_sample_size, random_state=rs)
		sampled_complaints=pd.concat([sampled_wrongs, sampled_confirms])
		num_complaints=len(sampled_wrongs)
		num_confirm=len(sampled_confirms)

	sampled_complaints['id'] = sampled_complaints.reset_index().index

	stimestamp = datetime.now()
	rc = RepairConfig(strategy=strat, complaints=sampled_complaints, monitor=FixMonitor(rule_set_size=20), acc_threshold=0.8, runtime=0)
	current_fixed_percent=0
	fixed_rate=0
	runtime=0
	current_start_id_pos=0
	new_global_accuracy=0
	confirm_preserve_rate=0
	new_signaled_cnt=0
	num_funcs=0
	post_fix_num_funcs=0
	tree_ids=[k.id for k in tree_rules]
	tree_ids.sort()
	new_all_sentences_df=None
	num_of_funcs_processed_by_algo=0
	print(f"fixed_rate:{fixed_rate}, retrain_accuracy_thresh:{retrain_accuracy_thresh}")
	fix_book_keeping_dict = {k.id:{'rule':k, 'deleted':False, 'pre_fix_size':k.size, 'after_fix_size':k.size} for k in tree_rules}
		# fix_book_keeping_dict[treerule.id]['pre_fix_size']=treerule.size

	while(fixed_rate<retrain_accuracy_thresh):
		start = time.time()
		prev_stop_tree_id_pos = fix_rules(repair_config=rc, fix_book_keeping_dict=fix_book_keeping_dict, conn=conn, 
			return_after_percent=retrain_after_percent, deletion_factor=deletion_factor, current_start_id_pos=current_start_id_pos, sorted_rule_ids=tree_ids)
		tree_rules=[v['rule'] for k,v in fix_book_keeping_dict.items() if not v['deleted']]
		num_of_funcs_processed_by_algo+=(prev_stop_tree_id_pos-current_start_id_pos)
		current_start_id_pos=prev_stop_tree_id_pos+1
		post_fix_num_funcs=len([value for value in fix_book_keeping_dict.values() if not value['deleted']])
		print(f"post_fix_num_funcs: {post_fix_num_funcs}")
		num_funcs = len(fix_book_keeping_dict)
		end = time.time()
		runtime+=round(end-start,3)
		# print(bkeepdict)
		print(f"runtime: {runtime}")
		# retrain snorkel using modified labelling funcs
		print("retraining using the fixed rules")
		print(tree_rules)
		# new_all_sentences_df.to_csv('new_all_sentences.csv', index=False)
		# new_wrongs_df.to_csv('new_wrongs.csv', index=False)
		new_labelling_funcs = [f.gen_label_rule() for f in tree_rules]
		new_global_accuracy, new_all_sentences_df, new_wrongs_df = lf_main(li, LFs=new_labelling_funcs)
		new_signaled_cnt=len(new_all_sentences_df)
		fixed_rate, confirm_preserve_rate = calculate_retrained_results(sampled_complaints, new_wrongs_df, result_dir+'/'+timestamp_str)
		if(current_start_id_pos>=len(tree_rules)):
			break

	before_total_size=after_total_size=0
	for k,v in fix_book_keeping_dict.items():
		if(not v['deleted']):
			before_total_size+=v['pre_fix_size']
			after_total_size+=v['after_fix_size']

	avg_tree_size_increase=(after_total_size-before_total_size)/post_fix_num_funcs
	print(f"avg tree_size increase: {avg_tree_size_increase}")

	if(not os.path.exists(result_dir+'/'+timestamp_str+'_experiment_stats')):
		with open(result_dir+'/'+timestamp_str+'_experiment_stats', 'w') as file:
			# Write some text to the file
			file.write('strat,runtime,avg_tree_size_increase,num_complaints,confirmation_cnt,global_accuracy,fix_rate,confirm_preserve_rate,new_global_accuracy,prev_signaled_cnt,new_signaled_cnt,' +\
				'num_functions,deletion_factor,post_fix_num_funcs,num_of_funcs_processed_by_algo\n')

	all_sentences_df.to_csv(result_dir+'/'+timestamp_str+'_initial_results.csv', index=False)
	sampled_complaints.to_csv(f"{result_dir}/sampled_complaints_{timestamp_str}_{strat}_{str(sample_size)}.csv", index=False)
	for kt,vt in fix_book_keeping_dict.items():
		with open(f"{result_dir}/{timestamp_str}_tree_{strat}_{kt}_dot_file", 'a') as file:
			comments=f"// presize: {fix_book_keeping_dict[kt]['pre_fix_size']}, after_size: {fix_book_keeping_dict[kt]['after_fix_size']}, deleted: {fix_book_keeping_dict[kt]['deleted']} factor: {deletion_factor} reverse_cnt:{fix_book_keeping_dict[kt]['rule'].reversed_cnt}"
			dot_file=fix_book_keeping_dict[kt]['rule'].gen_dot_string(comments)
			file.write(dot_file)
		print("dot string:")
		print(dot_file)
	new_all_sentences_df.to_csv(result_dir+'/'+timestamp_str+'_after_fix_results.csv', index=False)
	with open(result_dir+'/'+timestamp_str+'_experiment_stats', 'a') as file:
		# Write the row to the file
		file.write(f'{strat},{runtime},{avg_tree_size_increase},{num_complaints},{num_confirm},{round(global_accuracy,3)},{round(fixed_rate,3)},{round(confirm_preserve_rate,3)},'+\
			f'{round(new_global_accuracy,3)},{old_signaled_cnt},{new_signaled_cnt},{num_funcs},{deletion_factor},{post_fix_num_funcs},{num_of_funcs_processed_by_algo}\n')




# parameters needed 

# repair method (str): naive, information gain, optimal
# userinput size (integer):  
# complaint ratio
# lf source? (str: running example, sys_gen)
# number of lfs (if sysgen)
# db conn params
# dataset name




# 1. repeate the experiments to see how stable the runtime + tree size increase changes
# 2. finish implementing retraing with new rules to get the accuracy (global and user input)
# 3. add the parameter / functionality to control deleting factor
# 4. add the functionality of early stopping and retraing (based on percentage of the current fix on the target node)
# 5. LF sources: revisit witan repository to see if theres a good / new dataset with better lfs available