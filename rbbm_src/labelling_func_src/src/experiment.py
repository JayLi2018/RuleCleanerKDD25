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
	lattice_dict)
from rbbm_src.labelling_func_src.src.classes import SPAM, HAM, ABSTAIN, lf_input_internal, clean_text

from itertools import combinations
import glob
import numpy as np
from copy import deepcopy
from rbbm_src.labelling_func_src.src.LabelRepair import LabelRepairer, Repair
from rbbm_src.labelling_func_src.src.LabelExplain import LabelExpaliner

logger = logging.getLogger(__name__)


def lf_main(lf_input, LFs=None):
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
	logger.critical(LFs)
	sentences_df=pd.read_sql(f'SELECT * FROM {lf_input.dataset_name}', conn)
	logger.critical(sentences_df.head())
	sentences_df = sentences_df.rename(columns={"class": "expected_label", "content": "old_text"})
	sentences_df['text'] = sentences_df['old_text'].apply(lambda s: clean_text(s))
	sentences_df = sentences_df[~sentences_df['text'].isna()]
	lexp = LabelExpaliner()
	lf_internal_args = lf_input_internal(funcs=LFs, lattice_dict=lattice_dict)


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

	return accuracy, df_sentences_filtered, wrong_preds