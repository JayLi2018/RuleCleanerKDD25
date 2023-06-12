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
from math import floor
from rbbm_src.labelling_func_src.src.lfs_tree import keyword_labelling_func_builder, regex_func_builder, senti_func_builder, pos_func_builder, length_func_builder


def gen_example_funcs():
	TreeRule.rule_counter=0
	f1 = keyword_labelling_func_builder(['songs', 'song'], HAM)
	f2 = keyword_labelling_func_builder(['check'], SPAM)
	f3 = keyword_labelling_func_builder(['love'], HAM)
	f4 = keyword_labelling_func_builder(['shakira'], SPAM)
	f5 = keyword_labelling_func_builder(['checking'], SPAM)
	f6 = regex_func_builder(['http'],SPAM)
	f_sent = senti_func_builder(0.5)
	f_tag=pos_func_builder(['PRPVB'])
	f_length=length_func_builder(5)
	f7 = keyword_labelling_func_builder(['subscribe'], SPAM)

	example_rules = [f1, f2, f3, f4, f5, f6, f_sent, f_tag, f_length, f7]

	return example_rules