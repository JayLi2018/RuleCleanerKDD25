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

	def gen_funcs(self, count, apply_to_sentence_percentage_thresh, label_accuracy_thresh, label_accuracy_cap, pickle_it=False, pickle_file_name=None, checked_words=None):
		res = []
		if(not checked_words):
			checked_words = set([])
		candidate_sentences = self.df['text'].values.tolist()
		cur_cnt=0
		ccnt=0
		for c in candidate_sentences:
			ccnt+=1
			for w in c.split():
				word=w.lower()
				if(len(res)>=count):
					if(pickle_it):
						with open(f'{pickle_file_name}.pkl', 'wb') as file:
						    pickle.dump(res, file)
						# with open(f'{pickle_file_name}.pkl', 'rb') as file:
						#     test_funcs = pickle.load(file)
						# for obj in test_funcs:
						#     print(obj)
					return res, checked_words
				if(word in stop_words):
					# print(f"{word} is a stop word, skipped")
					continue
				elif(not is_word(word)):
					# print(f"{word} is not a word, skipped")
					continue
				elif(word in checked_words):
					# print(f"{word} already checked, skipped")
					continue
				checked_words.add(word)
				for t in [SPAM, HAM]:
					cand_f = keyword_labelling_func_builder([word], t)
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
							print(f"word: {word}, label:{t}, match_cnt: {match_cnt}, non_abstain_results_cnt: {non_abstain_results_cnt}, current cnt={cur_cnt}")
							res.append(cand_f)
							print(f"sentence_cnt: {ccnt}, current result_cnt={cur_cnt}")
							break
						# else:
						# 	print(f"{word} didnt pass the test, unlucky")
					# else:
					# 	print(f"{word} does not meet apply_to_sentence_percentage_thresh, skipped")
				# print('\n')

		print('didnt have enough words / threshold too high to have eligible funcs')
		exit()




if __name__ == '__main__':

	# conn = psycopg2.connect(dbname='label', user='postgres', password='123')
	conn = psycopg2.connect(dbname='label', user='postgres', password='123', port=5433)

	# sentences_df=pd.read_sql(f'SELECT * FROM youtube', conn)
	sentences_df=pd.read_sql(f'SELECT * FROM enron', conn)

	# sentences_df = sentences_df.rename(columns={"class": "expected_label", "content": "old_text"})
	sentences_df = sentences_df.rename(columns={"label": "expected_label",  "content": "old_text"})

	sentences_df['text'] = sentences_df['old_text'].apply(lambda s: clean_text(s))
	sentences_df = sentences_df[~sentences_df['text'].isna()]

	kwm = KeyWordRuleMiner(sentences_df)

	# funcs = kwm.gen_funcs(count=30, 
	# 	apply_to_sentence_percentage_thresh=0.025, label_accuracy_thresh=0.7, pickle_it=True, pickle_file_name='picked_funcs_613')

	# funcs = kwm.gen_funcs(count=30, 
	# 	apply_to_sentence_percentage_thresh=0.025, label_accuracy_thresh=0.7, pickle_it=True, pickle_file_name='picked_funcs_613')

	total_funcs=[]

	bad_funcs, checked_words=kwm.gen_funcs(count=10, 
		apply_to_sentence_percentage_thresh=0.025, label_accuracy_thresh=0, label_accuracy_cap=0.2, pickle_it=False, pickle_file_name='picked_funcs_620')

	total_funcs.extend(bad_funcs)

	good_funcs, checked_words =kwm.gen_funcs(count=20,
		apply_to_sentence_percentage_thresh=0.03, label_accuracy_thresh=0.65, label_accuracy_cap=1, pickle_it=False, pickle_file_name='picked_funcs_620', checked_words=checked_words
		)

	total_funcs.extend(good_funcs)

	with open(f'pickled_funcs_enron.pkl', 'wb') as file:
	    pickle.dump(total_funcs, file)
	# print(funcs)