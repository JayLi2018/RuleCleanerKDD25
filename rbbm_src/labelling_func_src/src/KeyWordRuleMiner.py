from snorkel.labeling import(
PandasLFApplier
)
import psycopg2 
import pandas as pd 
from rbbm_src.labelling_func_src.src.lfs_tree import keyword_labelling_func_builder
from rbbm_src.labelling_func_src.src.classes import SPAM, HAM, ABSTAIN, lf_input_internal, clean_text
import pickle
import nltk
from nltk.corpus import stopwords
from math import floor
from itertools import combinations

nltk.download('stopwords') 
stop_words = set(stopwords.words('english')) 


def is_word(token):
    from enchant.checker import SpellChecker
    spell_checker = SpellChecker("en_US")
    spell_checker.set_text(token)

    for error in spell_checker:
        return False

    return True


class KeyWordRuleMiner:
	"""
	given a dataframe with expected labels, 
	generate keyword functions such that tries to
	follow the labels as much as it can
	"""

	def __init__(self,df):
		self.df = df

	def verify_word(self, cand, stop_words, checked_words):
		if(cand in checked_words):
			# print(f"{word} already checked, skipped")
			return False
		for w in cand:
			if(cand in stop_words):
				# print(f"{word} is a stop word, skipped")
				return False
		for w in cand:
			if(not is_word(w)):
				# print(f"{word} is not a word, skipped")
				return False
		return True


	def gen_funcs(self, count, apply_to_sentence_percentage_thresh, label_accuracy_thresh, label_accuracy_cap, pickle_it=False, 
		pickle_file_name=None, checked_words=None, is_good=True, cardinality_thresh=1):
		res = []
		if(not checked_words):
			checked_words = set([])
		candidate_sentences = self.df['text'].values.tolist()
		cur_cnt=0
		ccnt=0
		for c in candidate_sentences:
			ccnt+=1
			words=c.lower().split()
			cur_card = 1
			while(cur_card<=cardinality_thresh):
				cur_words = [cw for cw in combinations(words,cur_card)]
				for ws in cur_words:
					checked = False
					for w in ws:
						if(self.verify_word(w, stop_words, checked_words)):
							checked_words.add(w)
						else:
							checked=True 
					if(checked):
						continue
					for t in [SPAM, HAM]:
						cand_f = keyword_labelling_func_builder(keywords=[x for x in ws], expected_label=t, is_good=is_good)
						tree_rules=[cand_f]
						labelling_funcs=[f.gen_label_rule() for f in tree_rules]
						applier = PandasLFApplier(lfs=labelling_funcs)
						initial_vectors = applier.apply(df=self.df, progress_bar=False)
						func_results = [x[0] for x in list(initial_vectors)]
						non_abstain_results_cnt=len([x for x in func_results if x!=ABSTAIN])
						# print(f'non_abstain_results_cnt: {non_abstain_results_cnt}')
						if(non_abstain_results_cnt/len(self.df)>apply_to_sentence_percentage_thresh):
							gts = self.df['expected_label'].values.tolist()
							match_cnt = len([x for x,y in zip(func_results,gts) if (x == y and x!=ABSTAIN)])
							if(label_accuracy_thresh <= match_cnt/non_abstain_results_cnt <= label_accuracy_cap):
								cur_cnt+=1
								print(f"word: {ws}, label:{t}, match_cnt: {match_cnt}, non_abstain_results_cnt: {non_abstain_results_cnt}, accuracy:{match_cnt/non_abstain_results_cnt}, current cnt={cur_cnt}")
								res.append(cand_f)
								print(f"sentence_cnt: {ccnt}, current result_cnt={cur_cnt}")
								if(len(res)>=count):
									if(pickle_it):
										with open(f'{pickle_file_name}.pkl', 'wb') as file:
										    pickle.dump(res, file)
									return res, checked_words
								break

						# else:
						# 	print(f"{word} didnt pass the test, unlucky")
					# else:
					# 	print(f"{word} does not meet apply_to_sentence_percentage_thresh, skipped")
				# print('\n')
				cur_card+=1

		print('didnt have enough words / threshold too high to have eligible funcs, returned the found funcs anyway')
		return res, checked_words
		# exit()




if __name__ == '__main__':

	# conn = psycopg2.connect(dbname='label', user='postgres', password='123')
	conn = psycopg2.connect(dbname='label', user='postgres', password='123')

	# sentences_df=pd.read_sql(f'SELECT * FROM youtube', conn)
	# sentences_df=pd.read_sql(f'SELECT * FROM enron', conn)

	for t in ['youtube']:
		for bad_ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
			for size in [30]:
				bad_cnt = floor(size * bad_ratio)
				good_cnt = size-bad_cnt
				sentences_df=pd.read_sql(f'SELECT * FROM {t}', conn)

				# sentences_df = sentences_df.rename(columns={"class": "expected_label", "content": "old_text"})
				sentences_df = sentences_df.rename(columns={"class": "expected_label",  "content": "old_text"})

				sentences_df['text'] = sentences_df['old_text'].apply(lambda s: clean_text(s))
				sentences_df = sentences_df[~sentences_df['text'].isna()]
				sentences_df = sentences_df.sort_values(by=['cid'])
				kwm = KeyWordRuleMiner(sentences_df)
				total_funcs=[]
				bad_funcs, checked_words=kwm.gen_funcs(count=bad_cnt, 
					apply_to_sentence_percentage_thresh=0.025, label_accuracy_thresh=0.45, label_accuracy_cap=0.55, pickle_it=False,
					is_good=False, cardinality_thresh=2)

				total_funcs.extend(bad_funcs)

				good_funcs, checked_words =kwm.gen_funcs(count=good_cnt,
					apply_to_sentence_percentage_thresh=0.01, label_accuracy_thresh=0.70, label_accuracy_cap=1, pickle_it=False,
					checked_words=checked_words, is_good=True, cardinality_thresh=2)

				total_funcs.extend(good_funcs)

				checked_words = None

				with open(f'pickled_funcs_{t}_{size}_bad_{bad_cnt}.pkl', 'wb') as file:
				    pickle.dump(total_funcs, file)
	# print(funcs)