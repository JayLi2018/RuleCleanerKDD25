from rbbm_src.labelling_func_src.src.LabelExplain import LabelExpaliner
from rbbm_src.labelling_func_src.src.classes import lfunc_dec, make_keyword_lf,keyword_lookup, SPAM, HAM, ABSTAIN
from snorkel.labeling import PandasLFApplier
from copy import deepcopy
import pandas as pd
import logging
import numpy as np
from rbbm_src.labelling_func_src.src.lfs import textblob_sentiment
from snorkel.preprocess import preprocessor
from dataclasses import dataclass
from itertools import chain

logger = logging.getLogger(__name__)

@dataclass(eq=True)
class Repair:
	new_lf : lfunc_dec = None
	old_lf : lfunc_dec = None
	action : str = None

	def __hash__(self):
		return hash(self.new_lf) + hash(self.old_lf) + hash(self.action)

	# def __str__(self):
	# 	return f"{self.lf}:{self.the_action}"

	# def set_lf(self, lf):
	# 	self.lf = lf

	# def set_action(self, the_action):
	# 	self.the_action = the_action


class LabelRepairer:

	"""
	LabelRepairer is responsible for 
		1.the repair candidate generation (add/remove/refine)
		2.given a repair candidate, evaulates the metrics of performing this repair candidate
			a. if the expected label is acquired
			b. the repair coverage (different repairs the coverage computations are also different)
	"""


	def __init__(self, explainer):
		self.explainer = explainer

	def generate_repair_candidates(self, lfs, max_step=1):
		"""
		Args:
			lfs: labelling functions to be repaired (based on the results from the explanation part)
			max_step: maximum number of steps can take in a repair(e.g. 2 step repair could be
			delete a lf and add a new lf)
			)
		"""
		res = []
		# permutations of rtypes ('add','refine','delete') and responsible lfs
		card_perm=min(len(lfs), max_step)
		all_possible_repairs = []
		for s in range(1, card_perm+1):
			action_perms = gen_unique_sequence_with_repet(['add','refine', 'delete'], s)
			lf_perms = gen_unique_sequence_with_repet(lfs, s)
			final_combs = [zip(r1,r2) for r1 in action_perms for r2 in lf_perms]
			all_possible_repairs_in_s = [list(f) for f in final_combs]
			all_possible_repairs.extend(all_possible_repairs_in_s)

		return all_possible_repairs


	def construct_repairs(self, func_and_words_influence_dict, repair_candidates, expected_label):
		"""
		repair_candidates is a list of lists of tuples,
		each tuple encodes the following information:
			t[0]: action
			t[1]: the function
		
		return:  a list of list of repair objects
		"""

		res_repairs = []

		# [('add', func: textblob_subjectivity), ('add', func: textblob_subjectivity)]
		for r in repair_candidates:
			repairs_for_this_cand = [[]]
			for step in r:
				cur_step_repairs = []
				for one_repair in repairs_for_this_cand:
					words = func_and_words_influence_dict[step[1]]
					words_lists = [w[1]['words'] for w in words[:min(3, len(words))]]
					# consider up to 3 words as repair candidates
					if(words_lists):
						# words_to_consider= sorted(list(chain.from_iterable(words_lists)))
						for w in words_lists:
							new_repair=deepcopy(one_repair)
							if(step[0]=='add'):
								# for w in words_to_consider:
								rep = Repair()
								rep.new_lf = make_keyword_lf(keywords=w, label=expected_label, name=f"added_keyword_{','.join(w)}")
								logger.critical(rep.new_lf)
								rep.action='add'
								new_repair.append(rep)
								cur_step_repairs.append(new_repair)
							if(step[0]=='refine'):
								for w in words_lists:
									new_repair=deepcopy(one_repair)
									# for w in words_to_consider:
									@lfunc_dec(name=f"refined_{step[1].name}_with_{','.join(w)}")
									def composite_func(x):
										if any(word in x.text.lower() for word in w):
											return expected_label
										else:
											return step[1](x)
									rep = Repair()
									rep.new_lf = composite_func
									rep.old_lf = step[1]
									rep.action='refine'
									new_repair.append(rep)
									cur_step_repairs.append(new_repair)
					if(step[0]=='delete'):
						new_repair=deepcopy(one_repair)
						rep = Repair()
						rep.action='delete'
						rep.old_lf=step[1]
						new_repair.append(rep)
						cur_step_repairs.append(new_repair)
				repairs_for_this_cand = cur_step_repairs
			res_repairs.extend(repairs_for_this_cand)

		logger.critical(res_repairs)

		return res_repairs		


	def evaluate_repair_candidate(self, repair_index, model_type, lfs, lf_vectors, 
		sentences_with_model_pred, the_repair, soi,
		expected_label):
		"""
		Args:
			model_type: majority/snorkel
			lfs: a list of lfs from original model input
			lf_vectors: a cached dictionary of predictions for each sentence by each lf
			sentences_with_model_pred(Dataframe): the dataframe used in the setting. should be the same as the 
				ones used in explanation part since we need the original labels to calculate effects of a repair
			the_repair: a list of repair class objects that contains the component(s) needed to do the repair and 
			to evaluate
			soi: sentence of interest
			expected_label: the expected label used to check if the repair actually fixed soi
		"""
		lfs_to_test = deepcopy(lfs)
		lf_vectors = deepcopy(lf_vectors)		
		sdf = pd.DataFrame(data={'text':[soi]})
		for one_step in the_repair:
			if(one_step.action=='add' or one_step.action=='refine'):
				if(one_step.action=='refine'):
					if(one_step.old_lf in lf_vectors):
						del lf_vectors[one_step.old_lf]
						lfs_to_test = [f for f in lfs_to_test if f.name!=one_step.old_lf.name]
				# only add lf that is syntax/keyword related for now
				logger.critical(one_step.new_lf)
				applier = PandasLFApplier(lfs=[one_step.new_lf])
				filtered_vectors = applier.apply(df=sentences_with_model_pred, progress_bar=False)
				lf_vectors[one_step.new_lf] = np.transpose(filtered_vectors)[0]
				lfs_to_test.append(one_step.new_lf)

			if(one_step.action=='delete'):
				if(one_step.old_lf in lf_vectors):
					# TODO: how to prevent refine and then delete?
					del lf_vectors[one_step.old_lf]
					lfs_to_test = [f for f in lfs_to_test if f.name!=one_step.old_lf.name]
		logger.critical(lfs_to_test)
		label_after_the_repair, model, model_results = self.explainer.retrain(funcs=lfs_to_test, func_vectors=lf_vectors, 
			model_type=model_type, soi_df=sdf, sentences_df=sentences_with_model_pred, predict_all=True)
		sentences_with_model_pred['pred_after_repair'] = pd.Series(model_results)
		# logger.critical(sentences_with_model_pred[sentences_with_model_pred["model_pred"]==sentences_with_model_pred['pred_after_repair']])
		agreed_cnt = len(sentences_with_model_pred[sentences_with_model_pred["model_pred"]==sentences_with_model_pred['pred_after_repair']])
		correct_cnt = len(sentences_with_model_pred[sentences_with_model_pred["pred_after_repair"]==sentences_with_model_pred['label']])
		flipping_rate = 1 - agreed_cnt/len(sentences_with_model_pred)
		accuracy = correct_cnt/len(sentences_with_model_pred)
		# sentences_with_model_pred.to_csv(f'repair_csv/repair_number_{repair_index}.csv')

		# if(the_repair.the_action=='refine'):
		# 	applier = PandasLFApplier(lfs=[the_repair.lf])


		# if(the_repair.the_action=='delete'):
		# 	pass

		return {
		"repair_index": repair_index,
		'the_repair': the_repair,
		'expected_label': expected_label, 
		'label_after_the_repair': label_after_the_repair, 
		'agreed_cnt': agreed_cnt,
		'flipping_rate': flipping_rate,
		'accuracy': accuracy,
		'correct_cnt': correct_cnt,
		'num_sentences': len(sentences_with_model_pred)}


def gen_unique_sequence_with_repet(l, card):
	
	def extendl(l, cands):
		res = []
		for c in cands:
			nl=l[:]
			nl.append(c)
			res.append(nl)
		return res
	
	res = [[e] for e in l]
	i=1
	while(i<card):
		new_res = []
		for r in res:
			new_res.extend(extendl(r,l))
		res = new_res
		i+=1

	return res
