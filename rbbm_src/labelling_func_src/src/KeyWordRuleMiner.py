from snorkel.labeling import(
PandasLFApplier
)
import psycopg2 
import pandas as pd 
from rbbm_src.labelling_func_src.src.lfs_tree import keyword_labelling_func_builder
from rbbm_src.labelling_func_src.src.classes import SPAM, HAM, ABSTAIN, lf_input_internal, clean_text

class KeyWordRuleMiner:
	"""
	given a dataframe with expected labels, 
	generate keyword functions such that tries to
	follow the labels as much as it can
	"""

	def __init__(self,df):
		self.df = df

	def gen_funcs(self, count, label_accuracy_thresh):
		res = []
		checked_words = set([])
		candidate_sentences = self.df['text'].values.tolist()
		for c in candidate_sentences:
			for w in c.split(' '):
				if(len(res)>=count):
					return res
				for t in [SPAM, HAM]:
					cand_f = keyword_labelling_func_builder([w], t)
					tree_rules=[cand_f]
					labelling_funcs=[f.gen_label_rule() for f in tree_rules]
					applier = PandasLFApplier(lfs=labelling_funcs)
					initial_vectors = applier.apply(df=self.df, progress_bar=False)
					func_results = [x[0] for x in list(initial_vectors)]
					non_abstain_results_cnt=len([x for x in func_results if x!=ABSTAIN])
					if(non_abstain_results_cnt>0):
						gts = self.df['expected_label'].values.tolist()
						match_cnt = len([x for x,y in zip(func_results,gts) if (x == y and x!=ABSTAIN)])
						if(match_cnt/non_abstain_results_cnt>=label_accuracy_thresh):
							print(f'non_abstain_results_cnt: {non_abstain_results_cnt}, match_cnt:{match_cnt}')
							res.append(cand_f)
							break
		print('didnt have enough words / threshold too high to have eligible funcs')
		exit()




if __name__ == '__main__':

	conn = psycopg2.connect(dbname='label', user='postgres', password='123')
	sentences_df=pd.read_sql(f'SELECT * FROM youtube', conn)
	sentences_df = sentences_df.rename(columns={"class": "expected_label", "content": "old_text"})
	sentences_df['text'] = sentences_df['old_text'].apply(lambda s: clean_text(s))
	sentences_df = sentences_df[~sentences_df['text'].isna()]

	kwm = KeyWordRuleMiner(sentences_df)

	funcs = kwm.gen_funcs(5, 0.5)

	# print(funcs)