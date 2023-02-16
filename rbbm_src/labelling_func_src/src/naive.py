from itertools import combinations, chain
from bottom_up import (
	sentence_filter,
	reward_function,
	relevance
	)
from utils import load_spam_dataset

from snorkel.labeling import (
	LabelingFunction, 
	labeling_function, 
	PandasLFApplier, 
	LFAnalysis)
from snorkel.labeling.model import MajorityLabelVoter, LabelModel
from lfs import *
from typing import *
from classes import lfunc, SPAM, HAM, ABSTAIN
import pandas as pd
import logging
# import logconfig
import argparse


logger = logging.getLogger(__name__)


# naive/exhaustive algorithm 
# to find the best R = F \cup A
# that maximizes 
# 1_{M_F(s_0) = M(s_0)}(E_{z \in S| A}[1_{M_F(z) = M(s_0)}] - E_{z \in S|\neg A}[1_{M(z) = M(s_0)}])



# modules 
# 1. S generator (use built in combinations)
# 2. value function


class CandidateGenerator:
	def __init__(self, words, funcs, model_version='majority'):
		self.words = words
		self.funcs = funcs
		self.model_version=model_version
	
	def generate_all(self):
		w_combs = [list(ws) for ws in list(chain(*map(lambda x: combinations(self.words, x), range(1, len(self.words)+1))))]
		# words cant be empty?
		f_combs = [list(fs) for fs in list(chain(*map(lambda x: combinations(self.funcs, x), range(1, len(self.funcs)+1))))]
		# funcs cant be empty for sure
		res = []
		for wc in w_combs:
			for fc in f_combs:
				if(len(fc)<3 and self.model_version=='snorkel'):
					if(len(fc)==1):
						fc.extend([keyword_empty, keyword_space])
					else:
						fc.append(keyword_empty)		
				d={'words': wc, 'funcs':fc}
				res.append(d)

		return res


def run_naive(
	full_model,
	full_model_applier,
	sentence_of_interest: str,
	sentences: pd.DataFrame,
	label_for_s: int,
	labeling_funcs: List[lfunc],
	model_type: Optional[str]='majority',
	):
	
	words = list(set([w.lower() for w in sentence_of_interest.split()]))
	logger.debug(f"words: {words}")
	cg = CandidateGenerator(words=words, funcs=labeling_funcs, model_version=model_type)
	candidate_exps = cg.generate_all()
	logger.debug(f"number of candidates: {len(candidate_exps)}")
	res = [] # a list of results with their scores
	i=0
	for c in candidate_exps:
		if(i%50==0):
			logger.debug(f"we are on {i}th.....")
		model_f_applier=PandasLFApplier(lfs=c['funcs'])
		# 2 groups of sentences based on A

		if(model_type=='majority'):
			sentences_with_A=sentences[sentences['text'].apply(lambda s: 
				sentence_filter(c['words'], s))]

			sentences_not_with_A=sentences[~sentences['text'].apply(lambda s: 
				sentence_filter(c['words'], s))]

			model_f = MajorityLabelVoter()
			if(len(sentences_with_A)==0):
				E_s_with_A=0
			else:
				labels_with_A = model_f_applier.apply(df=sentences_with_A, progress_bar=False)
				model_results_with_A = model_f.predict(L=labels_with_A)
				num_model_results_with_A = len([x for x in model_results_with_A if x==label_for_s])
				E_s_with_A =num_model_results_with_A/len(model_results_with_A)

			if(len(sentences_not_with_A)==0):
				E_s_not_with_A=0
			else:
				labels_not_with_A = full_model_applier.apply(df=sentences_not_with_A, progress_bar=False)
				model_results_not_with_A = full_model.predict(L=labels_not_with_A)
				num_model_results_not_with_A = len([x for x in model_results_not_with_A if x==label_for_s])
				E_s_not_with_A =num_model_results_not_with_A/len(sentences_not_with_A)

		else:
			model_f = LabelModel(cardinality=2, verbose=True) 
			labels = model_f_applier.apply(df=sentences, progress_bar=False)
			# snorkel needs to get an estimator using fit function first
			model_f.fit(L_train=labels, n_epochs=500, log_freq=100, seed=123)
			# filter out unlabeled data and only predict those that receive signals
			probs_train = model_f.predict_proba(L=labels)
			# what should be done here, should include those or not
			df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(
				X=sentences, y=probs_train, L=labels
			)
			# reset df_train to those receive signals
			df = df_train_filtered.reset_index(drop=True)

			sentences_with_A=df[df['text'].apply(lambda s: 
				sentence_filter(c['words'], s))]

			sentences_not_with_A=df[~df['text'].apply(lambda s: 
				sentence_filter(c['words'], s))]


			if(len(sentences_with_A)==0):
				E_s_with_A=0
			else:
				labels_with_A = model_f_applier.apply(df=sentences_with_A, progress_bar=False)
				model_results_with_A = model_f.predict(L=labels_with_A)
				num_model_results_with_A = len([x for x in model_results_with_A if x==label_for_s])
				E_s_with_A =num_model_results_with_A/len(model_results_with_A)
			if(len(sentences_not_with_A)==0):
				E_s_not_with_A=0
			else:
				labels_not_with_A = full_model_applier.apply(df=sentences_not_with_A, progress_bar=False)
				model_results_not_with_A = full_model.predict(L=labels_not_with_A)
				num_model_results_not_with_A = len([x for x in model_results_not_with_A if x==label_for_s])
				E_s_not_with_A =num_model_results_not_with_A/len(sentences_not_with_A)

		c_res = {"explainer": c, "num_s_with_A": len(sentences_with_A), \
		"num_s_not_with_A": len(sentences_not_with_A), "num_result_A": num_model_results_with_A, \
		"num_resul_notA":num_model_results_not_with_A, "score": E_s_with_A-E_s_not_with_A}
		res.append(c_res)
		i+=1

	res = sorted(res, key=lambda d: d['score'], reverse=True) 
	res_df = pd.DataFrame(res)

	return res_df

if __name__ == '__main__':

	LFs = [
	keyword_shakira,
	keyword_my,
	keyword_subscribe,
	regex_link,
	keyword_song,
	# has_person_nlp,
	# textblob_polarity,
	# textblob_subjectivity,
	short_comment,
	]

	parser = argparse.ArgumentParser(description='Running experiments')

	parser.add_argument('-M','--model', metavar="\b", type=str, default='majority',

	help='the model used to get the label: majority/snorkel (default: %(default)s)')
	args=parser.parse_args()

	logger.debug(args.model)

	df = pd.read_csv('data/Youtube05-Shakira.csv')
	df = df.rename(columns={"CLASS": "label", "CONTENT": "text"})

	applier = PandasLFApplier(lfs=LFs)

	df_train = applier.apply(df=df, progress_bar=False)

	if(args.model=='majority'):
		model = MajorityLabelVoter()
	else:
		model = LabelModel(cardinality=2, verbose=True)
		# snorkel needs to get an estimator using fit function first
		model.fit(L_train=df_train, n_epochs=500, log_freq=100, seed=123)
		# filter out unlabeled data and only predict those that receive signals
		probs_train = model.predict_proba(L=df_train)
		df_filtered, probs_train_filtered = filter_unlabeled_dataframe(
			X=df, y=probs_train, L=df_train
		)
		# reset df to those receive signals
		df = df_filtered.reset_index(drop=True)
		df_train = applier.apply(df=df, progress_bar=False)
		model.fit(L_train=df_train, n_epochs=500, log_freq=100, seed=123)

	df['pred'] = pd.Series(model.predict(L=df_train))
	# the wrong labels we get
	wrong_preds = df[(df['label']!=df['pred']) & (df['pred']!=ABSTAIN)]

	for index, row in wrong_preds.iterrows():
	  print("--------------------------------------------------------------------------------------------")  
	  print(f"setence#: {index} \n sentence: {row['text']} \n correct_label : {row['label']} \n pred_label: {row['pred']} \n")
	choices = input('please input sentence # of sentence, multiple sentences should be seperated using space')
	logger.debug(f"choices: {choices}")
	choice_indices = [int(x.strip()) for x in choices.split()]
	logger.debug(f"choice_indices: {choice_indices}")
	sentences_of_interest = list(wrong_preds.loc[choice_indices].text.values.astype(str))
	logger.debug(f"sentences_of_interest: {sentences_of_interest}")

	for s in sentences_of_interest:
		soi_df = df[df['text']==s]
		# other_sentences = df[~df['text'].isin(s)]
		soi_label = list(soi_df['pred'].values.astype(int))[0]
		soi_correct_label = list(soi_df['label'].values.astype(int))[0]
		soi = list(soi_df['text'].values.astype(str))[0].lower()

		resdf= run_naive(
			full_model=model,
			full_model_applier=applier,
			sentence_of_interest=soi,
			sentences=df,
			model_type=args.model,
			label_for_s=soi_label,
			labeling_funcs=LFs)
		resdf.to_csv(f'{soi}_results.csv', index=False)
