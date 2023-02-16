# implementation of causuality based 
# explanation
import heapq
import pandas as pd
from typing import *
from snorkel.labeling import (
	LabelingFunction, 
	labeling_function, 
	PandasLFApplier, 
	LFAnalysis,
	filter_unlabeled_dataframe
	)
from snorkel.labeling.model import MajorityLabelVoter, LabelModel
import os
import logging
import argparse
# import logconfig
from rbbm_src.labelling_func_src.src.bottom_up import sentence_filter, delete_words
from itertools import chain
from rbbm_src.labelling_func_src.src.lfs import LFs,twitter_lfs, LFs_smaller, LFs_running_example
from rbbm_src.labelling_func_src.src.classes import SPAM, HAM, ABSTAIN
from itertools import combinations
import glob
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from copy import deepcopy
from textblob import TextBlob


logger = logging.getLogger(__name__)

# flows
# for functions:
#  1. every function in an counterfactual cause is an actual cause with contingency |counterfactual cause| - 1



"""
# algorithm:
expected label = l_{T'}

responsibilities R = {F_1: -1 , F_2: -1 ... F_i=-1}

cached_models = {}


initialize the candidate singular functions as candidate solutions S = { {F_1}, {F_2} \ldots, {F_i} }
	
	while S
		for s in S
			if M_{F-s} (T0) = l_{T’}
				remove s from S (S[i] = None)
			else:
				cached_models[s] = M_{F-s}
				save model results as it can be used in the next iteration


None monotone : 

{f1, f2} --> flips the label

f1 -> f2. f2 -> f1 contingency

{f1, f2, f3} --- flips the label but {f1, f3} and {f2, f3} do not

f3 -> {f1, f2}

"""
# logger = logging.getLogger(__name__)

# parser = argparse.ArgumentParser(description='Running experiments')

# parser.add_argument('-M','--training_model_type', metavar="\b", type=str, default='majority',
# help='the model used to get the label: majority/snorkel (default: %(default)s)')

# parser.add_argument('-l', '--log_level', metavar='\b', type=str, default='debug',
# help='loglevel: debug/info/warning/error/critical (default: %(default)s)')

# parser.add_argument('-u', '--user_provide', metavar='\b', type=str, default='yes',
# help='user select from all wrong labels? (default: %(default)s)')

# parser.add_argument('-T', '--approach', metavar='\b', type=str, default='responsibility',
# help='approach: responsibility based or anchor based (responsibility/anchor) (default: %(default)s)')

# parser.add_argument('-W', '--word_threshold', metavar='\b', type=int, default=0,
# help='word threshold when evaluating inflences of words(default: %(default)s)')

# parser.add_argument('-g', '--greedy', metavar='\b', type=str, default='False',
# help='early stop if no increase in terms of words influence (default: %(default)s)')

# parser.add_argument('-c', '--cardinality_thresh', metavar='\b', type=int, default=4,
# help='cardinality threshold if non greedy (i.e. exhaustive), ONLY userful when greedy==False (default: %(default)s)')

# # parser.add_argument('-F', '--first_model_type', metavar='\b', type=str, default='topk',
# # help='first term model type in doing words influence expectation difference formula (topk/full) (default: %(default)s)')

# # parser.add_argument('-S', '--second_model_type', metavar='\b', type=str, default='topk',
# # help='second term model type in doing words influence expectation difference formula (topk/full) (default: %(default)s)')

# parser.add_argument('-E', '--eval_mode', metavar='\b', type=str, default='single_func',
# help='method used to evaluate the model (default: %(default)s)')

def subj(x):
    b = TextBlob(x.text)
    return b.sentiment.polarity

class LabelExpaliner:

	def __init__(self):
		# self.timer = Timer()
		self.retrain_cnt=0
		# self.timer.stats.startTimer('timername')

	# def func_responsibility(self, funcs, func_vectors, sentences, expected_label, sentence_of_interest, model_type, topk, 
	# 	lattice=False, lattice_layers=None, lattice_dict=None):

	def func_responsibility(self, lf_input, lf_internal_args):
		# NOTE: when lattice is given, the order of funcs matter!
		# generate the original model's prediction first 
		sdf = pd.DataFrame(data={'text':[lf_internal_args.sentence_of_interest]})
		logger.critical(f'sentence of Interest: {lf_internal_args.sentence_of_interest}')
		applier = PandasLFApplier(lfs=lf_internal_args.funcs)
		logger.critical(sdf)
		soi_labels = applier.apply(df=sdf, progress_bar=False)
		# current labels by each function
		func_labels = soi_labels.tolist()[0]
		func_w_labels_on_soi = dict(zip(lf_internal_args.funcs, func_labels))
		logger.critical(func_labels)
		logger.critical(lf_internal_args.funcs)
		logger.critical(func_w_labels_on_soi)

		# "combine" labels given a model type and get one single prediction
		
		lf_input.stats.startTimer('retrain')
		ori_label, _ =self.retrain(funcs=lf_internal_args.funcs, func_vectors=lf_internal_args.func_vectors, model_type=lf_input.training_model_type, 
			soi_df=sdf, sentences_df=lf_internal_args.filtered_sentences_df)
		self.retrain_cnt+=1
		lf_input.stats.stopTimer('retrain')

		logger.critical(f"original label : {ori_label}\n")
		logger.critical(f"expected_label: {lf_internal_args.expected_label}\n")

		# cached intermediate model results
		model_results = {}
		responsibilities = {f:[-1] for f in lf_internal_args.funcs}
		func_contingencies = {}
		contingency_cand_dict = {}
		# look_up_cnt=0 
		# new_model_cnt=0
		# heap_list = []

		topk = min(len(lf_internal_args.funcs), lf_input.topk)
		logger.critical(f"topk={topk}")

		for i in range(0, len(lf_internal_args.funcs)):
			func_contingencies[lf_internal_args.funcs[i]]=[]
			contingency_cand_dict[lf_internal_args.funcs[i]] = [fc for fc in lf_internal_args.funcs if fc.name!=lf_internal_args.funcs[i].name]

		# iterate over each function, using a heap structure to
		# track the smallest function with its responsibility
		# if the highest possible responsibility of the current iteration
		# is smaller than the smallest element in the heap root, then
		# we can early stop

		if(not lf_input.using_lattice):
			for j in range(0, len(lf_internal_args.funcs)-1):
				# logger.critical(f"current heap: {heap_list}")
				for f in lf_internal_args.funcs:
					if(responsibilities[f][0]==-1):
						self.gen_func_responsibility(lf_input, contingency_cand_dict, f, j, lf_input.training_model_type, model_results, 
							responsibilities, func_contingencies, lf_internal_args.funcs, lf_internal_args.func_vectors, 
							lf_internal_args.filtered_sentences_df, sdf,
							lf_internal_args.expected_label)
				# if(len(heap_list)>=topk):
				# 	break
		else:
			remaining_funcs=deepcopy(lf_internal_args.funcs)
			while(remaining_funcs):
				logger.critical(f"remaining_funcsL: {remaining_funcs}")
				f = remaining_funcs[0]
				for j in range(0, len(lf_internal_args.funcs)-1):
					logger.critical(f"on func {f}, trying contingency size {j}")
					# logger.critical(responsibilities)
					# logger.critical(funcs)
					if(responsibilities[f][0]==-1):
						if(self.gen_func_responsibility(lf_input, contingency_cand_dict, f, j, lf_input.training_model_type, model_results, 
							responsibilities, func_contingencies, lf_internal_args.funcs, lf_internal_args.func_vectors, 
							lf_internal_args.filtered_sentences_df, sdf,
							lf_internal_args.expected_label)):
							break
				if(responsibilities[f][0]==-1):
					logger.critical(f"{f} has no responsibility, skipping {lf_internal_args.lattice_dict[f]}")
				# remove everything contained by f from fun candidates
					funcs_left=[fc for fc in remaining_funcs if fc not in lf_internal_args.lattice_dict[f] and fc is not f]
				else:
					funcs_left=[fc for fc in remaining_funcs if fc is not f]
				remaining_funcs=funcs_left

				# if(len(heap_list)>=topk):
				# 	break

		logger.critical(responsibilities)
		# logger.critical(model_results)
		# logger.critical(f"look_up_cnt: {look_up_cnt}")
		# logger.critical(f"new_model_cnt: {new_model_cnt}")

		return responsibilities

	def gen_func_responsibility(self, lf_input, contingency_cand_dict, f, j, model_type, model_results, responsibilities, 
		func_contingencies, funcs, func_vectors, sentences, sdf, expected_label):
		f_contingency_cands = contingency_cand_dict[f]
		for con in combinations(f_contingency_cands,j):
			# contigency candidate, that is the set that needs to be
			# removed first before f being removed
			cause_cand = list(con)
			contingency_cand = frozenset(cause_cand)
			cause_cand.append(f)
			func_contingencies[f].append(cause_cand)
			model_funs = [mf for mf in funcs if (mf not in cause_cand)]
			cause_set = frozenset(cause_cand)

			if(model_type=='snorkel'):
				if(len(model_funs)<3):
					logger.critical("cur model func len <3, skip")
					continue
			if(cause_set in model_results):
				# look_up_cnt+=1
				flabel = model_results[cause_set]
			else:
				# new_model_cnt+=1
				lf_input.stats.startTimer('retrain')
				flabel, _ =self.retrain(funcs=model_funs, func_vectors=func_vectors, model_type=model_type, soi_df=sdf, sentences_df=sentences)
				self.retrain_cnt+=1
				lf_input.stats.stopTimer('retrain')
				model_results[cause_set]=flabel
				# logger.critical(f'model_results len : {len(model_results)}')
				# logger.critical(f"after training using {model_funs}: we get {flabel}")
			if(flabel==expected_label):
				logger.critical(f'flipped to {flabel}')
				if(len(contingency_cand)>0):
					if(responsibilities[f][0]==-1):
						if(contingency_cand not in model_results):
							model_funs = [mf for mf in funcs if (mf not in contingency_cand)]
							lf_input.stats.startTimer('retrain')
							flabel, _ =self.retrain(funcs=model_funs, func_vectors=func_vectors, model_type=model_type, 
								soi_df=sdf, sentences_df=sentences)
							self.retrain_cnt+=1
							lf_input.stats.stopTimer('retrain')
							model_results[contingency_cand] = flabel
						if(model_results[contingency_cand]!= expected_label):
							responsibilities[f][0]=1/len(cause_cand)
							responsibilities[f].append(cause_cand)
							responsibility = 1/len(cause_cand)
							logger.critical((responsibility, f))
							# heapq.heappush(heap_list, (responsibility, id(f), f))
							return True
				else:
					responsibilities[f][0]=1
					responsibility = 1
					logger.critical((responsibility, f))
					# heapq.heappush(heap_list, (responsibility, id(f),f))
					return True
		return False


	def retrain(self, funcs, func_vectors, model_type, soi_df, sentences_df, predict_all=False):
		applier = PandasLFApplier(lfs=funcs)
		# get sentence of interest vector
		# logger.critical(soi_df)
		# logger.critical(funcs)
		# logger.critical(list(func_vectors))
		soi_vector = applier.apply(df=soi_df, progress_bar=False)
		used_vecs = []
		for f in funcs:
			used_vecs.append(func_vectors[f])
			# logger.critical(func_vectors[f])
		input_vectors = np.array(used_vecs, dtype=int).transpose()
		# logger.critical(f"input_vectors: {input_vectors}")
		if(model_type=='majority'):
			model = MajorityLabelVoter()
		else:
			model = LabelModel(cardinality=2, verbose=True)
			# snorkel needs to get an estimator using fit function first
			# logger.critical(type(input_vectors))
			# logger.critical(type(input_vectors[0]))
			# logger.critical(type(input_vectors[0][0]))
			# logger.critical(input_vectors)
			model.fit(L_train=input_vectors, n_epochs=500, log_freq=100, seed=123)
		labels_on_soi = model.predict(L=soi_vector)
		if(predict_all):
			# logger.critical(input_vectors)
			# logger.critical(input_vectors.shape)
			labels_full = model.predict(L=input_vectors)
			# logger.critical(labels_full.shape)
			return labels_on_soi[0], model, labels_full

		# logger.critical(f"flabels: {labels_on_soi}\n")
		# logger.critical(f"retrian with: \n{funcs}, input vector is \n{soi_train}, result label is \n{labels_on_soi[0]}")
		return labels_on_soi[0], model

	# def generate_words_influence(self, 
	# 	sentence_of_interest,
	# 	model_funcs,
	# 	model, 
	# 	sentences_df,
	# 	expected_label,
	# 	model_type,
	# 	# norm_values,
	# 	eval_mode, #'single_func' 'new_model'
	# 	word_threshold,
	# 	greedy = True,
	# 	cardinality_thresh = 3
	# 	):
	def generate_words_influence(self, 
		lf_input,
		lf_internal_args
		):
		"""
		given a sentence of interest, generate combination of words and evaluate their influences on 
		a given model by using a general expectation difference

		sentence_of_interest: sentence that the user is interested in
		first/second_term_model: 2 models in expectation formula E

		𝐼𝑛𝑓(𝐴)=E_{D}(𝑧|𝐴)[1_{𝑙=M′}(𝑧)]−E_{D}(𝑧|𝐴)[1_{𝑙≠M′}(𝑧)]
		"""

		stop_words = set(stopwords.words('english'))
		word_tokens = word_tokenize(lf_internal_args.sentence_of_interest)
		# filter out stopwords
		filtered_words = list(set([w for w in word_tokens if not w.lower() in stop_words]))
		logger.critical(filtered_words)
		if(lf_input.eval_mode=='new_model'):
			words_influence_dict = {model:[]}
			words_influence_dict[lf_internal_args.model].append(self.gen_expectation(
				words=filtered_words, 
				expected_label=lf_internal_args.expected_label, 
				sentences_df=lf_internal_args.filtered_sentences_df, 
				model_funcs=lf_internal_args.funcs,
				model=lf_internal_args.model, 
				# norm_values=norm_values,
				word_threshold=lf_input.word_threshold,
				greedy=lf_input.greedy,
				cardinality_thresh=lf_input.cardinality_thresh
				)
				)			

		if(lf_input.eval_mode=='single_func'):
			words_influence_dict = {f:None for f in lf_internal_args.funcs}
			for f in lf_internal_args.funcs:
				if(f.words_used):
					filtered_words = list(set([w for w in filtered_words if w not in f.words_used]))
				words_influence_dict[f] = self.gen_expectation(
								words=deepcopy(filtered_words), 
								expected_label=lf_internal_args.expected_label, 
								sentences_df=lf_internal_args.filtered_sentences_df, 
								model_funcs=[f],
								model=MajorityLabelVoter(), 
								# norm_values=norm_values,
								word_threshold=lf_input.word_threshold,
								greedy=lf_input.greedy,
								cardinality_thresh=lf_input.cardinality_thresh
								)

		return words_influence_dict

	def gen_expectation(self, words, expected_label, sentences_df, 
		model_funcs, model, word_threshold,
		greedy = True,cardinality_thresh = 3,
		htest_level=0.05):
		
		logger.critical(f'greedy: {greedy}')		
		res = {}
		j=1

		applier = PandasLFApplier(lfs=model_funcs)	
		train_inputs_filtered = applier.apply(df=sentences_df, progress_bar=False)
		sentences_df['pred'] = pd.Series(model.predict(L=train_inputs_filtered))
		norm_values = sentences_df.groupby(['pred'])['pred'].count().to_dict()
		# adjust the expectation difference here by considering the 
		# overall distribution difference between different classes
		norm_sum = sum(v for (k,v) in norm_values.items())
		second_norm = sum(v for (k,v) in norm_values.items() if k!=expected_label)

		# while(words and j<len(words)):
		words_to_evaulate = [[w] for w in words]
		words_dict = {k:v for (v,k) in enumerate(words)}

		cur_card = 1
		while(words_to_evaulate):
			if(not greedy and cur_card>cardinality_thresh):
				logger.critical('not greedy and cur_card > cardinality_threshd')
				break
			for pw in words_to_evaulate:
				logger.critical(pw)
				cur_words = list(pw)
				logger.critical(cur_words)
				applier = PandasLFApplier(lfs=model_funcs)
				# applier_2 = PandasLFApplier(lfs=second_term_model_funcs)
				sentences_match_cur_words = sentences_df[sentences_df['text'].apply(lambda s: 
						sentence_filter(cur_words, s))]
				# if(cur_words==['hot'] and len(model_funcs)==1 and model_funcs[0].name=='textblob_subjectivity'):
				# 	logger.critical(train_inputs_filtered)
				# 	logger.critical(pd.Series(model.predict(L=train_inputs_filtered)))
				# 	sentences_match_cur_words.to_csv('hots.csv')
				# 	sentences_df.to_csv('all_sentences_when_hot.csv')
				len_match = len(sentences_match_cur_words)
				if(sentences_match_cur_words.empty or len(sentences_match_cur_words)==1):
					logger.critical('empty or only one sentence')
					continue
				else:
					# corpus_inputs = applier.apply(df=sentences_match_cur_words, progress_bar=False)
					# corpus_results = model.predict(L=corpus_inputs)
					# TODO: memorize labels by f so we dont repeat!
					model_pred_dict = sentences_match_cur_words.groupby(['pred']).size().to_dict()
					logger.critical(model_pred_dict)
					logger.critical(expected_label)
					logger.critical(len_match)
					try:
						first_term_value = model_pred_dict[expected_label]/len_match
					except KeyError as e:
						first_term_value = 0	
					second_term_value = 1- first_term_value

					# add a test layer: replace word set with '' to see the result
					# sentences_with_wordset_deleted = sentences_match_cur_words.copy(deep=True)
					# sentences_with_wordset_deleted['text'] = sentences_with_wordset_deleted['text'].apply(lambda s:
					# 	delete_words(cur_words, s))

					# corpus_inputs_with_wordset_deleted = applier.apply(df=sentences_with_wordset_deleted, progress_bar=False)
					# corpus_results_with_wordset_deleted = model.predict(L=corpus_inputs_with_wordset_deleted)
					# first_term_value_with_wordset_deleted = len([x for x in corpus_results_with_wordset_deleted if x==expected_label])/len_match
					# second_term_value_with_wordset_deleted = 1- first_term_value_with_wordset_deleted

				# hypothesis testing:
				comb_term = len(list(combinations(range(1,len(sentences_match_cur_words)+1),2)))
				logger.critical("model_pred_dict:")
				logger.critical(model_pred_dict)
				if(expected_label in norm_values and expected_label in model_pred_dict):
					p_expected = norm_values[expected_label]/norm_sum
					pvalue = comb_term * p_expected ** model_pred_dict[expected_label] * (1-p_expected)**(len(sentences_match_cur_words)-model_pred_dict[expected_label])
				else:
					p_expected = 0
					pvalue = 0
				if(pvalue<htest_level):
					significant=True 
				else:
					significant=False

				if(second_norm!=0):
					if(expected_label in norm_values):
						expect_diff = first_term_value * norm_sum/norm_values[expected_label] - second_term_value * norm_sum/second_norm
					else:
						expect_diff = -(second_term_value * norm_sum/second_norm)
				else:
					if(expected_label in norm_values):
						expect_diff = first_term_value * norm_sum/norm_values[expected_label]
					else:
						expect_diff =0
				words_res_dict = {'words': cur_words,
				'expect_diff': expect_diff,
				# 'expect_diff_after_deletion': first_term_value_with_wordset_deleted * norm_sum/norm_values[expected_label]
				# - second_term_value_with_wordset_deleted * norm_sum/second_norm,
				'first_term_value': first_term_value,
				'second_term_value': second_term_value,
				# 'first_term_value_with_wordset_deleted': first_term_value_with_wordset_deleted,
				# 'second_term_value_with_wordset_deleted': second_term_value_with_wordset_deleted,
				'num_sentence_match_words': len_match,
				'norm_values': norm_values, # TODO: adjust norm values to f specific
				'increase':False, # a flag to label if a word set has increase over previous steps
				'is_significant':significant,
				'pval': pvalue
				}

				logger.critical(words_res_dict)
				
				if(greedy):
					# TODO: check all subsets with BCs :)
					if(frozenset(pw[:-1]) in res):
						if((words_res_dict['first_term_value']-words_res_dict['second_term_value']) - \
							(res[frozenset(pw[:-1])]['first_term_value']-res[frozenset(pw[:-1])]['second_term_value'])>word_threshold):
							words_res_dict['increase']=True
							res[frozenset(pw)] = words_res_dict
					else:
						if(len(pw)==1):
							words_res_dict['increase']=True
							res[frozenset(pw)] = words_res_dict
				else:
					res[frozenset(pw)] = words_res_dict
					if(len(pw)>1):
						if((words_res_dict['first_term_value']-words_res_dict['second_term_value']) - \
							(res[frozenset(pw[:-1])]['first_term_value']-res[frozenset(pw[:-1])]['second_term_value'])>word_threshold):
							res[frozenset(pw)]['increase']=True
					else:
						res[frozenset(pw)]['increase']=True

				# if(words_res_dict['expect_diff'] == words_res_dict['expect_diff_after_deletion']):
				# 	ineffective_words.append(pw)
				# else:
				# 	# logger.critical("effective words found!")
				# 	# logger.critical(words_res_dict)
				# 	res.append(words_res_dict)
			new_words_to_evaulate = []
			logger.critical(res)
			for p in words_to_evaulate:
				if(frozenset(p) in res):	
					max_ind = max([words_dict[v] for v in p])
					if(max_ind<len(words_to_evaulate)-1):
							for i in range(max_ind+1, len(words)):
									new_p = list(deepcopy(p))
									new_p.append(words[i])
									new_words_to_evaulate.append(new_p)
			words_to_evaulate = new_words_to_evaulate
			logger.critical(words_to_evaulate)
			cur_card+=1

				# cur_word_sets = list(combinations(words, j))
			# j+=1

		return res

	def func_and_words_influence(self, 
		lf_input,
		lf_internal_args
		):
		# funcs, 
		# func_vectors, 
		# filtered_sentences_df, 
		# expected_label, 
		# predicted_label, 
		# sentence_of_interest,
		# topk=5, 
		# lattice_layers=None, 
		# lattice_dict=None
		# training_model_type: model_type
		# word_threshold: word_threshold
		# cardinality_thresh: cardinality_thresh
		# using_lattice: lattice
		# eval_mode: eval_mode
		# stats:


		# added a eval_mode argument: if 'single_func', do the topk funcs individually, 
		# if 'new_model', do the topk funcs together
		sdf = pd.DataFrame(data={'text':[lf_internal_args.sentence_of_interest]})
		logger.critical(f"expected_label: {lf_internal_args.expected_label}")
		responsibilities = self.func_responsibility(lf_input, lf_internal_args)
		# topk_funcs  = [k for k, v in sorted(responsibilities.items(), key=lambda item: item[1], reverse=True)][0:topk]
		topk_funcs_list = sorted([[k, v[0]] for (k,v) in responsibilities.items() if v[0]!=-1],key=lambda x: x[1], reverse=True)
		topk_funcs = [t[0] for t in topk_funcs_list]
		if(len(topk_funcs)>=lf_input.topk):
			topk_funcs = topk_funcs[:topk]

		logger.critical(f'topk_funcs: {topk_funcs}')

		lf_internal_args.topk_funcs=topk_funcs
		# applier = PandasLFApplier(lfs=funcs)	
		# train_inputs_filtered = applier.apply(df=filtered_sentences_df, progress_bar=False)
		# filtered_sentences_df['pred'] = pd.Series(model.predict(L=train_inputs_filtered))
		# global_cnts = filtered_sentences_df.groupby(['pred'])['pred'].count().to_dict()
		# logger.critical(global_cnts)

		# if(SPAM in global_cnts):
		# 	global_spam_cnt = global_cnts[SPAM]
		# else:
		# 	global_spam_cnt = 1

		# if(HAM in global_cnts):
		# 	global_ham_cnt = global_cnts[HAM]
		# else:
		# 	global_ham_cnt = 1
		words_influences = None
		if(lf_input.eval_mode=='new_model'):
			# _, model = self.retrain(topk_funcs, func_vectors, model_type, sdf, filtered_sentences_df)
			# _, model = self.retrain(lf_input,lf_internal_args)
			_, model = self.retrain(topk_funcs, lf_internal_args.func_vectors, lf_internal_args.training_model_type, sdf, 
				lf_internal_args.filtered_sentences_df)
			lf_internal_args.model=model
			words_influences = self.generate_words_influence(lf_input,lf_internal_args)
			# words_influences = self.generate_words_influence(
			# 	sentence_of_interest=sentence_of_interest,
			# 	model=model,
			# 	model_funcs=topk_funcs,
			# 	sentences_df=filtered_sentences_df,
			# 	expected_label=predicted_label,
			# 	model_type=model_type,
			# 	# norm_values={SPAM:global_spam_cnt, HAM:global_ham_cnt},
			# 	eval_mode=eval_model,
			# 	word_threshold=word_threshold,
			# 	greedy=greedy, 
			# 	cardinality_thresh=cardinality_thresh
			# 	)

		elif(lf_input.eval_mode=='single_func'):
			lf_internal_args.model=None
			words_influences = self.generate_words_influence(lf_input,lf_internal_args)

			# words_influences = self.generate_words_influence(
			# 	sentence_of_interest=sentence_of_interest,
			# 	model=None,
			# 	model_funcs=topk_funcs,
			# 	sentences_df=filtered_sentences_df,
			# 	expected_label=predicted_label,
			# 	model_type=model_type,
			# 	# norm_values={SPAM:global_spam_cnt, HAM:global_ham_cnt},
			# 	eval_mode=eval_mode,
			# 	word_threshold=word_threshold,
			# 	greedy=greedy, 
			# 	cardinality_thresh=cardinality_thresh
			# 	)
		logger.critical("function influences:")
		logger.critical(responsibilities)
		logger.critical("words_influences")
		logger.critical(words_influences)

		return responsibilities, words_influences

# def main(lf_input, lfs, sentences_df, type='notebook', lattice=None, args_str=None):

# 	lexp = LabelExpaliner()

# 	log_map = { 'debug': logging.DEBUG,
# 	'info': logging.INFO,
# 	'warning': logging.WARNING,
# 	'error': logging.ERROR,
# 	'critical': logging.CRITICAL
# 	}

# 	if(type=='notebook'):
# 		args = parser.parse_args(args_str.split())
# 	else:
# 		args=parser.parse_args()

# 	logger.critical(' '.join(f'{k}={v}' for k, v in vars(args).items()))

# 	try:
# 		logconfig.root.setLevel(log_map[lf_input.log_level])
# 	except KeyError as e:
# 		print('no such log level')
	
# 	LFs = lfs

# 	# Snorkel built-in labelling function applier
# 	applier = PandasLFApplier(lfs=LFs)

# 	# Apply the labelling functions to get vectors
# 	initial_vectors = applier.apply(df=sentences_df, progress_bar=False)
# 	# df.to_csv('snorkel.csv')

# 	if(lf_input.training_model_type=='majority'):
# 		model = MajorityLabelVoter()
# 	else:
# 		model = LabelModel(cardinality=2, verbose=True)
# 		model.fit(L_train=initial_vectors, n_epochs=500, log_freq=100, seed=123)
# 		# snorkel needs to get an estimator using fit function first
# 		# training model with all labelling functions

# 	probs_test= model.predict_proba(L=initial_vectors)
# 	df_sentences_filtered, probs_test_filtered = filter_unlabeled_dataframe(
# 			X=sentences_df, y=probs_test, L=initial_vectors
# 	)
# 	# reset df_train to those receive signals
# 	df_sentences_filtered = df_sentences_filtered.reset_index(drop=True)
# 	filtered_vectors = applier.apply(df=df_sentences_filtered, progress_bar=False)
# 	cached_vectors = dict(zip(LFs, np.transpose(filtered_vectors)))
# 	logger.critical(cached_vectors)

# 	if(lf_input.training_model_type=='snorkel'):
# 		model.fit(L_train=filtered_vectors, n_epochs=500, log_freq=100, seed=123)

# 	df_sentences_filtered['model_pred'] = pd.Series(model.predict(L=filtered_vectors))

# 	df_sentences_filtered['vectors'] = pd.Series([",".join(map(str, t)) for t in filtered_vectors])

# 	# df_test_filtered.to_csv('result.csv')
# 	# the wrong labels we get
# 	wrong_preds = df_sentences_filtered[(df_sentences_filtered['label']!=df_sentences_filtered['model_pred']) & (df_sentences_filtered['model_pred']!=ABSTAIN)]
# 	wrong_preds['signal_strength'] = wrong_preds['vectors'].apply(lambda s: sum([1 for i in s.split(",") if int(i) == SPAM or int(i)==HAM]))
# 	wrong_preds = wrong_preds.sort_values(['signal_strength'], ascending=False)
# 	logger.critical(wrong_preds)
# 	logger.critical(f"""
# 		out of {len(sentences_df)} sentences, {len(df_sentences_filtered)} actually got at least one signal to \n
# 		make prediction. Out of all the valid predictions, we have {len(wrong_preds)} wrong predictions, \n
# 		accuracy = {(len(df_sentences_filtered)-len(wrong_preds))/len(df_sentences_filtered)} 
# 		""")

# 	if(lf_input.user_provide):
# 		for index, row in wrong_preds.iterrows():
# 			logger.critical("--------------------------------------------------------------------------------------------")  
# 			logger.critical(f"setence#: {index} \n sentence: {row['text']} \n correct_label : {row['label']} \n pred_label: {row['model_pred']} \n vectors: {row['vectors']}\n")
# 		choices = input('please input sentence # of sentence, multiple sentences should be seperated using space')
# 		logger.critical(f"choices: {choices}")
# 		choice_indices = [int(x.strip()) for x in choices.split()]
# 		logger.critical(f"choice_indices: {choice_indices}")
# 		sentences_of_interest = wrong_preds[wrong_preds.index.isin(choice_indices)]
# 		logger.critical(f"sentences_of_interest: {sentences_of_interest}")
# 	else:
# 		sentences_of_interest = wrong_preds

# 	logger.critical(sentences_of_interest[['text','model_pred']])

# 	rind=0
# 	for i, r in sentences_of_interest.iterrows():
# 		# soi_df = df_test_filtered[df_test_filtered['text']==s]
# 		soi_label = r['model_pred']
# 		soi_correct_label = r['label']
# 		logger.debug(soi_label)
# 		soi = r['text']
# 		rind+=1
# 		logconfig.root.handlers = []
# 		file_handler=logging.FileHandler(f'{i}_log_new.txt', 'w+')
# 		console_handler = logging.StreamHandler()
# 		console_handler.setFormatter(logconfig.stream_formatter)
# 		logconfig.root.addHandler(console_handler)
# 		file_handler.setFormatter(logconfig.file_formatter)
# 		logconfig.root.addHandler(file_handler)
# 		# res = func_responsibility(funcs=LFs2, sentences=df_train, expected_label=t['label'], sentence_of_interest=t['text'], model_type='snorkel')
# 		if(lattice):
# 			func_responsibility, words_responsibility = lexp.func_and_words_influence(funcs=LFs, func_vectors=cached_vectors, 
# 				filtered_sentences_df=df_sentences_filtered, expected_label=soi_correct_label, predicted_label=soi_label, sentence_of_interest=soi, topk=2, 
# 				lattice=lattice)
# 		else:
# 			func_responsibility, words_responsibility = lexp.func_and_words_influence(funcs=LFs, func_vectors=cached_vectors, 
# 				filtered_sentences_df=df_sentences_filtered, expected_label=soi_correct_label, predicted_label=soi_label, sentence_of_interest=soi, model_type=args.training_model_type, topk=2, 
# 				eval_mode=args.eval_mode, word_threshold=args.word_threshold, greedy=greedy, cardinality_thresh=args.cardinality_thresh)

# 	return func_responsibility, words_responsibility, df_sentences_filtered

# if __name__ == '__main__':

# 	all_filenames = [i for i in glob.glob('data/*.csv')]
	
# 	df = pd.concat([pd.read_csv(f) for f in all_filenames])

# 	# df = pd.read_csv('data/Youtube05-Shakira.csv')

# 	df = df.rename(columns={"CLASS": "label", "CONTENT": "text"})

# 	# df = df.rename(columns={"class": "label", "content": "text"})

# 	# df_train = df.iloc[0:100].append(df.iloc[200:])
# 	# df_test = df.iloc[100:200]

# 	main(args_str="", lfs=LFs_smaller, sentences_df=df, type='terminal')