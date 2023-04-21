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
from rbbm_src.labelling_func_src.src.lfs_tree import keyword_labelling_func_builder, regex_func_builder, f_sent, f_tag, f_length
from itertools import product
from collections import deque
from rbbm_src.classes import StatsTracker, FixMonitor, RepairConfig, lf_input
from rbbm_src.labelling_func_src.src.classes import lf_input_internal, clean_text
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
	cur_number=tree_rule.size+1
	# if(len(the_fix)==4):
	#     _, _, _, sign = the_fix
	# elif(len(the_fix)==5):
	#     _, _, _, _, sign = the_fix
	# new_pred, modified_fix = convert_tuple_fix_to_pred(the_fix, reverse)
	# # print("modified_fix")
	# # print(modified_fix)
	# if(len(the_fix)==4):
	# elif(len(the_fix)==5):
	#     new_predicate_node = PredicateNode(number=cur_number, pred=DCAttrPredicate(pred=new_pred, operator=sign))
	the_fix_words, llabel, rlabel = the_fix
	if(reverse):
		llabel, rlabel = rlabel, llabel
	new_predicate_node = PredicateNode(number=cur_number,pred=KeywordPredicate(keywords=[the_fix_words]))
	cur_number+=1
	new_predicate_node.left= LabelNode(number=cur_number, label=llabel, pairs={HAM:[], SPAM:[]}, used_predicates=set([]))
	cur_number+=1
	new_predicate_node.right=LabelNode(number=cur_number, label=rlabel, pairs={HAM:[], SPAM:[]}, used_predicates=set([]))
	new_predicate_node.left.parent= new_predicate_node
	new_predicate_node.right.parent= new_predicate_node

	# print(node)
	if(node.parent.left is node):
		node.parent.left=new_predicate_node
	else:
		node.parent.right=new_predicate_node

	new_predicate_node.parent = node.parent

	# if(len(modified_fix)==4):
	#     role, attr, const, sign = modified_fix
	for k in [SPAM, HAM]:
		for p in node.pairs[k]:
			if(new_predicate_node.pred.evaluate(p)):
				new_predicate_node.right.pairs[p['expected_label']].append(p)
				new_predicate_node.right.used_predicates.add(the_fix)
			else:
				new_predicate_node.left.pairs[p['expected_label']].append(p)
				new_predicate_node.left.used_predicates.add(the_fix)

	# elif(len(modified_fix)==5):
	#     role1, attr1, role2, attr2, sign = modified_fix
	#     for k in [CLEAN, DIRTY]:
	#         for p in node.pairs[k]:
	#             # print(p)
	#             # print(f"p['{role1}']['{attr1}']{sign}p['{role2}']['{attr2}']")
	#             if(eval(f"p['{role1}']['{attr1}']{sign}p['{role2}']['{attr2}']")):
	#                 new_predicate_node.right.pairs[p['expected_label']].append(p)
	#                 new_predicate_node.right.used_predicates.add(modified_fix)
	#             else:
	#                 new_predicate_node.left.pairs[p['expected_label']].append(p)
	#                 new_predicate_node.left.used_predicates.add(modified_fix)

	# print(f"after fix {the_fix}, the left child is: {new_predicate_node.left.pairs}")
	# print(f"after fix {the_fix}, the right child is: {new_predicate_node.right.pairs}")

	new_predicate_node.pairs={CLEAN:{}, DIRTY:{}}

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
	# print('ham_sentence')
	# print(ham_sentence)

	# print('spam_sentence')
	# print(spam_sentence)

	ham_available_words=set([h for h in ham_sentence.split() if h not in spam_sentence.split()])
	spam_available_words=set([s for s in spam_sentence.split() if s not in ham_sentence.split()])

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
	print(f"leaf_nodes")
	# for ln in leaf_nodes:
	# 	print('************')
	# 	print(f'ln.label: {ln.label}')
	# 	print(f'number: {ln.number}')
	# 	print(f'left: {ln.left} ')
	# 	print(f'right: {ln.right} ')
	# 	print(f'parent: {ln.parent} ')
	# 	print(f'label: {ln.label} ')
	# 	print(f'pairs: {ln.pairs} ')
	# 	print(f'used_predicates: {ln.used_predicates}')
	# 	print('***********')
	# 	print('\n')
	# print(f'len of leaf_nodes: {len(leaf_nodes)}')
	# exit()
	if(repair_config.strategy=='naive'):
		# initialize the queue to work with
		queue = deque([])
		for ln in leaf_nodes:
			queue.append(ln)
		# print(queue)
		# print(len(queue))
		i=0
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
				the_fix = find_available_repair(node.pairs[SPAM][0],
				 node.pairs[HAM][0], node.used_predicates)
				# print(f"the fix number={i}")
				# print(the_fix)
				new_parent_node=redistribute_after_fix(treerule, node, the_fix)
				i+=1
				# print("new_parent_node")
				# print(new_parent_node)
			# handle the left and right child after redistribution
			else:
				if(node.pairs[SPAM]):
					if(node.label!=SPAM):
						node.label=SPAM
				elif(node.pairs[HAM]):
					if(node.label!=HAM):
						node.label=HAM
				if(check_tree_purity(treerule)):
					# print('its pure already!')
					# print("----------------------")
					# print('\n')
					# print(treerule)
					# print('---------------------')
					# print('\n')
					return treerule

			if(new_parent_node):
				# print(f'the new new_parent_node')
				# print(new_parent_node)
				# exit()
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
				# print("adding new predicate, size+2")
				# print('\n')
				# print(f"after fix, treerule size is {treerule.size}")
			# print(f"queue size: {len(queue)}")
		return treerule

	elif(repair_config.strategy=='information gain'):
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
						# print(f'gini score: {gini}')
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
				elif(node.pairs[HAM]):
					if(node.label!=HAM):
						node.label=HAM
				if(check_tree_purity(treerule)):
					print('its pure already!')
					print("----------------------")
					print('\n')
					print(treerule)
					print('---------------------')
					print('\n')
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
				for k in [CLEAN,DIRTY]:
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

	# elif(repair_config.strategy=='optimal'):
	#     # 1. create a queue with tree nodes
	#     # 2. need to deepcopy the tree in order to enumerate all possible trees
	#     for ln in leaf_nodes:
	#         queue = deque([])
	#         queue.append(ln.number)
	#     # print(f"number of leaf_nodes: {len(queue)}")
	#     # print("queue")
	#     # print(queue)
	#     cur_fixed_tree = treerule
	#     while(queue):
	#         sub_root_number = queue.popleft()
	#         subqueue=deque([(cur_fixed_tree, sub_root_number, sub_root_number)])
	#         # triples are needed here, since: we need to keep track of the 
	#         # updated(if so) subtree root in order to check purity from that node
	#         # print(f"subqueue: {subqueue}")
	#         sub_node_pure=False
	#         while(subqueue and not sub_node_pure):
	#             prev_tree, leaf_node_number, subtree_root_number = subqueue.popleft()
	#             # print(f"prev tree: {prev_tree}")
	#             node = locate_node(prev_tree, leaf_node_number)
	#             # print(f"node that needs to be fixed")
	#             # print(f"nodel.label: {node.label}")
	#             # print(f"nodel.clean: {node.pairs[CLEAN]}")
	#             # print(f"nodel.dirty: {node.pairs[DIRTY]}")
	#             if(node.pairs[CLEAN] and node.pairs[DIRTY]):
	#                 # print("we need to fix it!")
	#                 # need to examine all possible pair combinations
	#                 considered_fixes = set()
	#                 # print(list(product(node.pairs[CLEAN], node.pairs[DIRTY])))
	#                 found_fix = False
	#                 for pair in list(product(node.pairs[CLEAN], node.pairs[DIRTY])):
	#                     if(found_fix):
	#                         break
	#                     the_fixes = find_available_repair(pair[0],
	#                      pair[1], domain_value_dict, node.used_predicates,
	#                      all_possible=True)
	#                     # print("the fixes")
	#                     # print(the_fixes)
	#                     # print("the_fixes")
	#                     # print(the_fixes)
	#                     # print(f"fixes: len = {len(the_fixes)}")
	#                     # print("iterating fixes")
	#                     # print(f"len(fixes): {len(the_fixes)}")
	#                     for f in the_fixes:
	#                         # print(f"the fix: {f}")
	#                         new_parent_node=None
	#                         if(f in considered_fixes):
	#                             continue
	#                         considered_fixes.add(f)
	#                         new_tree = copy.deepcopy(prev_tree)
	#                         node = locate_node(new_tree, node.number)
	#                         new_parent_node = redistribute_after_fix(new_tree, node, f)
	#                         if(leaf_node_number==sub_root_number):
	#                             # first time replacing subtree root, 
	#                             # the node number will change so we need 
	#                             # to replace it
	#                             subtree_root_number=new_parent_node.number
	#                             # print(f"subtree_root_number is being updated to {new_parent_node.number}")
	#                         new_tree.setsize(new_tree.size+2)
	#                         if(check_tree_purity(new_tree, subtree_root_number)):
	#                             # print("done with this leaf node, the fixed tree is updated to")
	#                             cur_fixed_tree = new_tree
	#                             # print(cur_fixed_tree)
	#                             found_fix=True
	#                             sub_node_pure=True
	#                             break
	#                         # else:
	#                         #     print("not pure yet, need to enqueue")
	#                         # handle the left and right child after redistribution
	#                         still_inpure=False
	#                         for k in [CLEAN,DIRTY]:
	#                             if(still_inpure):
	#                                 break
	#                             for p in new_parent_node.left.pairs[k]:
	#                                 if(p['expected_label']!=new_parent_node.left.label):
	#                                     # print("enqueued")
	#                                     # print("current_queue: ")
	#                                     new_tree = copy.deepcopy(new_tree)
	#                                     parent_node = locate_node(new_tree, new_parent_node.number)
	#                                     subqueue.append((new_tree, parent_node.left.number, subtree_root_number))
	#                                     # print(subqueue)
	#                                     still_inpure=True
	#                                     break
	#                         still_inpure=False          
	#                         for k in [CLEAN,DIRTY]:
	#                             if(still_inpure):
	#                                 break
	#                             for p in new_parent_node.right.pairs[k]:
	#                                 if(p['expected_label']!=new_parent_node.right.label):
	#                                     # print("enqueued")
	#                                     # print("current_queue: ")
	#                                     new_tree = copy.deepcopy(new_tree)
	#                                     parent_node = locate_node(new_tree, new_parent_node.number)
	#                                     # new_parent_node=redistribute_after_fix(new_tree, new_node, f)
	#                                     subqueue.append((new_tree, parent_node.right.number, subtree_root_number))
	#                                     # print(subqueue)
	#                                     still_inpure=True
	#                                     break
	#                         # print('\n')
	#             else:
	#                 # print("just need to reverse node condition")
	#                 reverse_node_parent_condition(node)
	#                 if(check_tree_purity(prev_tree, subtree_root_number)):
	#                     # print("done with this leaf node, the fixed tree is updated to")
	#                     found_fix=True
	#                     cur_fixed_tree = prev_tree
	#                     sub_node_pure=True
	#                     # print(cur_fixed_tree)
	#                     break
				# print(f"current queue size: {len(queue)}")
		# print("fixed all, return the fixed tree")
		# print(cur_fixed_tree)
		# return cur_fixed_tree 
		# list_of_repaired_trees = sorted(list_of_repaired_trees, key=lambda x: x[0].size, reverse=True)
		# return list_of_repaired_trees[0] 

	else:
		print("not a valid repair option")
		exit()

def calculate_retrained_results(complaints, new_wrongs_df):
	the_complaints = complaints[complaints['expected_label']!=complaints['model_pred']]
	the_confirmatons = complaints[complaints['expected_label']==complaints['model_pred']]
	still_wrongs = pd.merge(the_complaints, new_wrongs_df, left_on='comment_id', right_on='comment_id', how='inner')
	not_correct_anymores = pd.merge(the_confirmatons, new_wrongs_df, left_on='comment_id', right_on='comment_id', how='inner')
	print("complaints")
	print(complaints)
	print("new_wrongs")
	print(new_wrongs_df)
	print('still wrongs')
	print(still_wrongs)

	complaint_fix_rate=confirm_preserve_rate=1
	if(len(the_complaints)>0):
		complaint_fix_rate=1-len(still_wrongs)/len(the_complaints)
	if(len(the_confirmatons)>0):
		confirm_preserve_rate=1-len(not_correct_anymores)/len(the_confirmatons)

	return complaint_fix_rate, confirm_preserve_rate

def fix_rules(repair_config, original_rules, conn):
	rules = original_rules
	all_fixed_rules = []
	cur_fixed_rules = []
	# domain_value_dict = construct_domain_dict(conn, table_name=table_name)
	fix_book_keeping_dict = {k.id:{} for k in original_rules}
	# print(domain_value_dict)
	for treerule in rules:
		# print("before fixing the rule, the rule is")
		# print(r)
		# treerule = parse_dc_to_tree_rule(r)
		# print(treerule)
		fix_book_keeping_dict[treerule.id]['pre_fix_size']=treerule.size
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
		# print("node with pairs")
		# for ln in leaf_nodes:
			# print(f"node id: {ln.number}")
			# print(ln.pairs)
			# print('\n')
		# print(leaf_nodes)
		if(leaf_nodes):
			# its possible for certain rule we dont have any violations
			# print("the TreeRule we wanted to fix")
			# print(treerule)
			# print("the leaf nodes")
			# print(leaf_nodes)
			fixed_treerule = fix_violations(treerule, repair_config, leaf_nodes)
			# print(fixed_treerule)
			fix_book_keeping_dict[treerule.id]['after_fix_size']=fixed_treerule.size
			fixed_treerule_text = treerule.serialize()
			fix_book_keeping_dict[treerule.id]['fixed_treerule_text']=fixed_treerule_text
		else:
			fix_book_keeping_dict[treerule.id]['after_fix_size']=treerule.size
			fixed_treerule_text = treerule.serialize()
			fix_book_keeping_dict[treerule.id]['fixed_treerule_text']=fixed_treerule_text

	return fix_book_keeping_dict


if __name__ == '__main__':

	# create a file to store the results:
	for i in range(0, 5):
		timestamp = datetime.now()
		timestamp_str = timestamp.strftime('%Y-%m-%d-%H-%M-%S')
		with open('more_lf_'+timestamp_str, 'w') as file:
			# Write some text to the file
			file.write('strat,runtime,avg_tree_size_increase,num_complaints,confirmation_cnt,global_accuracy,fix_rate,confirm_preserve_rate,new_global_accuracy,prev_signaled_cnt,new_signaled_cnt\n')
		# load dataset 
		# sentences_df=pd.read_csv('/home/opc/chenjie/RBBM/rbbm_src/labelling_func_src/src/data/youtube.csv')
		# test_df = sentences_df.head(2)
		# test_df = test_df.rename(columns={"CLASS": "label", "CONTENT": "text"})
		# # print(list(sentences_df))
		# test_df['text'] = test_df['text'].apply(lambda s: ham_text(s))

		# f1.name='blah1'
		# f2.name='blah2'
		# print(f1.name)
		# print(f2.name)
		# applier = PandasLFApplier(lfs=[f.gen_label_rule() for f in tree_rules])

		# Apply the labelling functions to get vectors
		# initial_vectors = applier.apply(df=test_df, progress_bar=False)

		# print(test_df.text.tolist())

		# print(initial_vectors)
		# start = time.time()
		# bkeepdict = fix_rules(repair_config=rc, original_rules=test_rules, conn=conn, table_name=table_name)

		conn = psycopg2.connect(dbname='label', user='postgres', password='123')
		for sample_size in [1,2,4,8,16,32,64,128]:
		# for sample_size in [1]:
			for strat in ['information gain', 'naive']:
				print(f"size: {sample_size*2}, strat:{strat}")
				f1 = keyword_labelling_func_builder(['songs', 'song'], HAM)
				f2 = keyword_labelling_func_builder(['check'], SPAM)
				f3 = keyword_labelling_func_builder(['love'], HAM)
				f4 = keyword_labelling_func_builder(['shakira'], SPAM)
				f5 = keyword_labelling_func_builder(['checking'], SPAM)
				f6 = regex_func_builder(['http'],SPAM)
				
				tree_rules = [f1, f2, f3, f4, f5, f6, f_sent, f_tag, f_length]
				labelling_funcs=[f.gen_label_rule() for f in tree_rules]

				li =lf_input(
					connection=conn,
					contingency_size_threshold=1,
					contingency_sample_times=1,
					clustering_responsibility=False,
					sample_contingency=False,
					log_level='DEBUG',
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
				old_signaled_cnt=len(all_sentences_df)
				all_sentences_df.to_csv('initial_results.csv', index=False)
				wrongs_df.to_csv('initial_wrongs.csv', index=False)
				wrong_hams=wrongs_df[wrongs_df['expected_label']==HAM]
				wrong_spams=wrongs_df[wrongs_df['expected_label']==SPAM]
				# for index, row in wrong_hams.iterrows():
				# 	logger.critical("--------------------------------------------------------------------------------------------")  
				# 	logger.critical(f"setence#: {index}  sentence: {row['text']} \n correct_label : {row['expected_label']}  pred_label: {row['model_pred']} vectors: {row['vectors']}\n")

				# for index, row in wrong_spams.iterrows():
				# 	logger.critical("--------------------------------------------------------------------------------------------")  
				# 	logger.critical(f"setence#: {index}  sentence: {row['text']} \n correct_label : {row['expected_label']}  pred_label: {row['model_pred']} vectors: {row['vectors']}\n")
				print(f"wrong_hams count: {len(wrong_hams)}")
				print(f"wrong_spams count: {len(wrong_spams)}")
				random_seed = random.randint(0, 1000)
				rng = pd.np.random.default_rng(random_seed)
				sampled_complaints = pd.concat([all_sentences_df[all_sentences_df['expected_label']==HAM].sample(n=sample_size,random_state=rng), \
					all_sentences_df[all_sentences_df['expected_label']==SPAM].sample(n=sample_size, random_state=rng)])
				print(f"sampled_complaints")
				num_complaints=len(sampled_complaints[sampled_complaints['expected_label']!=sampled_complaints['model_pred']])
				num_confirm=len(sampled_complaints[sampled_complaints['expected_label']==sampled_complaints['model_pred']])
				print(sampled_complaints)
				stimestamp = datetime.now()
				# Convert the timestamp to a string
				sample_time_stamp = stimestamp.strftime('%Y-%m-%d-%H-%M-%S')
				# sampled_complaints.to_csv(f'sampled_complaints_{sample_size}_{sample_time_stamp}.csv', index=False)
				# choices = input('please input sentence # of sentence, multiple sentences should be seperated using space')
				# sentences_of_interest = wrong_preds[wrong_preds.index.isin(choice_indices)]
				rc = RepairConfig(strategy=strat, complaints=sampled_complaints, monitor=FixMonitor(rule_set_size=20), acc_threshold=0.8, runtime=0)
				# rc = RepairConfig(strategy='naive', complaints=sampled_complaints, monitor=FixMonitor(rule_set_size=20), acc_threshold=0.8, runtime=0)

				start = time.time()
				bkeepdict = fix_rules(repair_config=rc, original_rules=tree_rules, conn=conn)
				num_funcs = len(bkeepdict)
				before_total_size=after_total_size=0
				for k,v in bkeepdict.items():
					before_total_size+=v['pre_fix_size']
					after_total_size+=v['after_fix_size']

				end = time.time()
				runtime=round(end-start,3)
				avg_tree_size_increase=(after_total_size-before_total_size)/num_funcs
				# print(bkeepdict)
				print(f"runtime: {runtime}")
				print(f"avg tree_size increase: {avg_tree_size_increase}")
				# retrain snorkel using modified labelling funcs

				print("retraining using the fixed rules")
				print(tree_rules)
				new_labelling_funcs = [f.gen_label_rule() for f in tree_rules]
				new_global_accuracy, new_all_sentences_df, new_wrongs_df = lf_main(li, LFs=new_labelling_funcs)
				new_signaled_cnt=len(new_all_sentences_df)
				new_wrongs_df.to_csv('new_wrongs.csv', index=False)
				fixed_rate, confirm_preserve_rate = calculate_retrained_results(sampled_complaints, new_wrongs_df)
				with open('lf_'+timestamp_str, 'a') as file:
					# Write the row to the file
					file.write(f'{strat},{runtime},{avg_tree_size_increase},{num_complaints},{num_confirm},{round(global_accuracy,3)},{round(fixed_rate,3)},{round(confirm_preserve_rate,3)},{round(new_global_accuracy,3)},{old_signaled_cnt},{new_signaled_cnt}\n')
					# file.write('strat,runtime,avg_tree_size_increase,num_complaints,confirmation_cnt,global_accuracy,fix_rate,confirm_preserve_rate,new_global_accuracy,prev_signaled_cnt,new_signaled_cnt\n')

		new_rules = []

		wrongs_df.to_csv('wrongs.csv', index=False)





# 1. repeate the experiments to see how stable the runtime + tree size increase changes
# 2. finish implementing retraing with new rules to get the accuracy (global and user input)
# 3. add the parameter / functionality to control deleting factor
# 4. add the functionality of early stopping and retraing (based on percentage of the current fix on the target node)
# 5. LF sources: revisit witan repository to see if theres a good / new dataset with better lfs available