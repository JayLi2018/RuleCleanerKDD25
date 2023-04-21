# implementation of causuality based 
# explanation
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
from rbbm_src import logconfig
from rbbm_src.labelling_func_src.src.bottom_up import sentence_filter, delete_words
from itertools import chain
from rbbm_src.labelling_func_src.src.lfs import (
	LFs,
	twitter_lfs, 
	LFs_smaller, 
	LFs_running_example,
	lattice_lfs,
	lattice_dict)
from rbbm_src.labelling_func_src.src.classes import SPAM, HAM, ABSTAIN, lf_input_internal, clean_text

from itertools import combinations
import glob
import numpy as np
from copy import deepcopy
from rbbm_src.labelling_func_src.src.LabelRepair import LabelRepairer, Repair
from rbbm_src.labelling_func_src.src.LabelExplain import LabelExpaliner

logger = logging.getLogger(__name__)


def lf_main(lf_input, LFs=lattice_lfs):
	#combine all files in the list,
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

	conn = lf_input.connection
	# all_filenames = [i for i in glob.glob('/home/opc/chenjie/labelling_explanation/src.labelling_func_src/src/data/Youtube05-Shakira.csv')]
	# logger.critical(all_filenames)
	logger.critical(LFs)
	sentences_df=pd.read_sql(f'SELECT * FROM {lf_input.dataset_name}', conn)
	logger.critical(sentences_df.head())
	# sentences_df = pd.concat([pd.read_csv(f) for f in all_filenames ])
	# sentences_df = sentences_df.rename(columns={"content": "text"})
	sentences_df = sentences_df.rename(columns={"class": "expected_label", "content": "text"})
	sentences_df['text'] = sentences_df['text'].apply(lambda s: clean_text(s))

	lexp = LabelExpaliner()
	lf_internal_args = lf_input_internal(funcs=LFs, lattice_dict=lattice_dict)

	# funcs
	# func_vectors
	# filtered_sentences_df
	# expected_label
	# predicted_label
	# sentence_of_interest
	# lattice_layers

	# if(lf_input.invoke_type=='notebook'):
	# 	args=lf_input.args_str.split()
	# # 	args = parser.parse_args(lf_input.arg_str.split())
	# # else:
	# # 	args=parser.parse_args()

	# logger.critical(' '.join(f'{k}={v}' for k, v in vars(args).items()))

	# LFs = lfs

	# Snorkel built-in labelling function applier
	applier = PandasLFApplier(lfs=lf_internal_args.funcs)

	# Apply the labelling functions to get vectors
	initial_vectors = applier.apply(df=sentences_df, progress_bar=False)
	# logger.critical(initial_vectors)
	# df.to_csv('snorkel.csv')

	if(lf_input.training_model_type=='majority'):
		model = MajorityLabelVoter()
	else:
		model = LabelModel(cardinality=2, verbose=True, device='cpu')
		model.fit(L_train=initial_vectors, n_epochs=500, log_freq=100, seed=123)
		# snorkel needs to get an estimator using fit function first
		# training model with all labelling functions

	probs_test= model.predict_proba(L=initial_vectors)
	df_sentences_filtered, probs_test_filtered = filter_unlabeled_dataframe(
			X=sentences_df, y=probs_test, L=initial_vectors
	)
	# reset df_train to those receive signals
	df_sentences_filtered = df_sentences_filtered.reset_index(drop=True)
	# df_sentences_filtered=sentences_df
	filtered_vectors=initial_vectors
	filtered_vectors = applier.apply(df=df_sentences_filtered, progress_bar=False)
	cached_vectors = dict(zip(LFs, np.transpose(filtered_vectors)))
	lf_internal_args.func_vectors = cached_vectors

	# logger.critical(cached_vectors)
	if(lf_input.training_model_type=='snorkel'):
		model.fit(L_train=filtered_vectors, n_epochs=500, log_freq=100, seed=123)

	df_sentences_filtered['model_pred'] = pd.Series(model.predict(L=filtered_vectors))

	df_sentences_filtered['vectors'] = pd.Series([",".join(map(str, t)) for t in filtered_vectors])

	lf_internal_args.filtered_sentences_df = df_sentences_filtered

	# df_test_filtered.to_csv('result.csv')
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

	if(lf_input.return_complaint_and_results):
		return accuracy, df_sentences_filtered, wrong_preds

	if(lf_input.user_provide):
		for index, row in wrong_preds.head(10).iterrows():
			logger.critical("--------------------------------------------------------------------------------------------")  
			logger.critical(f"setence#: {index}  sentence: {row['text']} \n correct_label : {row['expected_label']}  pred_label: {row['model_pred']} vectors: {row['vectors']}\n")
		choices = input('please input sentence # of sentence, multiple sentences should be seperated using space')
		logger.critical(f"choices: {choices}")
		choice_indices = [int(x.strip()) for x in choices.split()]
		logger.critical(f"choice_indices: {choice_indices}")
		sentences_of_interest = wrong_preds[wrong_preds.index.isin(choice_indices)]
		logger.critical(f"sentences_of_interest: {sentences_of_interest}")
	else:
		sentences_of_interest = wrong_preds


	logger.critical(sentences_of_interest[['text','model_pred']])

	rind=0
	for i, r in sentences_of_interest.iterrows():
		# soi_df = df_test_filtered[df_test_filtered['text']==s]
		soi_label = r['model_pred']
		soi_correct_label = r['expected_label']
		lf_internal_args.expected_label=soi_correct_label
		lf_internal_args.predicted_label=soi_label
		logger.debug(soi_label)
		soi = r['text']
		lf_internal_args.sentence_of_interest=soi
		rind+=1
		logconfig.root.handlers = []
		file_handler=logging.FileHandler(f'{i}_log_new.txt', 'w+')
		console_handler = logging.StreamHandler()
		console_handler.setFormatter(logconfig.stream_formatter)
		logconfig.root.addHandler(console_handler)
		file_handler.setFormatter(logconfig.file_formatter)
		logconfig.root.addHandler(file_handler)
		lf_internal_args.lattice_dict=lattice_dict
		# res = func_responsibility(funcs=LFs2, sentences=df_train, expected_label=t['label'], sentence_of_interest=t['text'], model_type='snorkel')
		# greedy = True if args.greedy=='True' else False

		# func_responsibility, words_responsibility = lexp.func_and_words_influence(funcs=LFs, func_vectors=cached_vectors, 
		# 	filtered_sentences_df=df_sentences_filtered, expected_label=soi_correct_label, predicted_label=soi_label, sentence_of_interest=soi, model_type=args.training_model_type, topk=1000, 
		# 	eval_mode=args.eval_mode, word_threshold=args.word_threshold, greedy=greedy, cardinality_thresh=args.cardinality_thresh, lattice=lattice, 
		# 	lattice_dict=lattice_dict)
		func_responsibility, words_responsibility = lexp.func_and_words_influence(lf_input=lf_input, lf_internal_args=lf_internal_args)

		# words_res_d = {}
		# logger.critical(words_responsibility)
		# for k,v in words_responsibility.items():
		#     vlist = ((kk,vv) for kk,vv in words_responsibility[k].items() if (vv['increase']==True and vv['is_significant']==True
		#     	and vv['expect_diff']>0))
		# #     words_res_d[k] = [k for k in vv]
		#     words_res_d[k] = sorted(vlist, key=lambda x: x[1]['first_term_value']- x[1]['second_term_value'], reverse=True)

		# repair_res = {}
		# lrp = LabelRepairer(lexp)
		# repair_results = []

		# repair_plans = lrp.generate_repair_candidates(lfs=list(words_res_d))
		# repairs = lrp.construct_repairs(func_and_words_influence_dict=words_res_d, repair_candidates=repair_plans,
		# 	expected_label=soi_correct_label)
		# # for func, wlist in words_res_d.items():
		# # 	words_consider = wlist[:min(3, len(wlist))]
		# # 	logger.critical(f"on func: {func}")
		# # 	logger.critical(f"words considered:{words_consider}")
		# # 	for w in words_consider:
		# # 		logger.critical(w[1]['words'])
		# # 		repair_cands = lrp.generate_repair_candidates(lf=func, words=w[1]['words'], rtype='add', 
		# # 			expected_label=soi_correct_label)
		# logger.critical(f"we have {len(repairs)} repair candidates")
		# xi = 1
		# for r in repairs:
		# 	logger.critical(f"on {xi}th repair ... ")
		# 	# [Repair(lf=func: nothttp, action='delete'), Repair(lf=func: nothttp, action='delete')]]
		# 	logger.critical(f'the repair: {r}')
		# 	logger.critical(f"is the repair elements euqal?")
		# 	len_repair = len(r)
		# 	len_set_repair = len(set(r))
		# 	logger.critical(f"len_repair: {len_repair}")
		# 	logger.critical(f"len_set_repair: {len_set_repair}")
		# 	logger.critical(f"{True if(len_repair==len_set_repair) else False}")
		# 	if(len_repair==len_set_repair):
		# 		repair_results.append(
		# 			lrp.evaluate_repair_candidate(repair_index=xi,
		# 			model_type=args.training_model_type,
		# 			lfs = LFs, lf_vectors= cached_vectors,
		# 			sentences_with_model_pred= df_sentences_filtered,
		# 			the_repair = r,
		# 			soi = soi, expected_label=soi_correct_label)
		# 		)
		# 	xi+=1
		# logger.critical(f"retrain cnt: {lexp.retrain_cnt}")
		# logger.critical(f"time to retrain: {lexp.timer.time['retrain']}")

		# return func_responsibility, words_res_d, df_sentences_filtered, repair_results