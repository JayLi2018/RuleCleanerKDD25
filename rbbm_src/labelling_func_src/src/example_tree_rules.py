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

def gen_amazon_funcs():
	TreeRule.rule_counter=0
	f1 = keyword_labelling_func_builder(['star', 'stars'], HAM)
	f2 = keyword_labelling_func_builder(['product', 'fit', 'quality', 'size', 'cheap', 'wear'], SPAM)
	f3 = keyword_labelling_func_builder(['great'], HAM)

	# f4 is an estension of f3
	cur_number=1
	tree_size=1
	f4_root = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=['great']))
	cur_number+=1
	tree_size+=1
	f4_root_left_leaf = LabelNode(number=cur_number, label=ABSTAIN, pairs={SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f4_root_right_child = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=['stars','works']))
	cur_number+=1
	tree_size+=1
	f4_root_right_child_left_leaf = LabelNode(number=cur_number, label=ABSTAIN, pairs={SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f4_root_right_child_right_leaf = LabelNode(number=cur_number, label=HAM, pairs={SPAM:[], HAM:[]}, used_predicates=set([]))
	f4_root.left = f4_root_left_leaf
	f4_root_left_leaf.parent=f4_root
	f4_root.right = f4_root_right_child
	f4_root_right_child.parent=f4_root
	f4_root_right_child.left = f4_root_right_child_left_leaf
	f4_root_right_child_left_leaf.parent=f4_root_right_child
	f4_root_right_child.right = f4_root_right_child_right_leaf
	f4_root_right_child_right_leaf.parent=f4_root_right_child
	f4 = TreeRule(rtype='lf', root=f4_root, size=tree_size, max_node_id=cur_number, is_good=True)

	# f4 = keyword_labelling_func_builder(['great','stars','works'], HAM)
	f5 = keyword_labelling_func_builder(['waste'], SPAM)
	f6 = keyword_labelling_func_builder(['shoes','item','price','comfortable','plastic'], HAM)
	f7 = keyword_labelling_func_builder(['junk','bought','like','dont','just','use','buy','work','small','didnt','did','disappointed'], SPAM)

	# f8 is an estension of f7
	cur_number=1
	tree_size=1
	f8_root = PredicateNode(pred=KeywordPredicate(keywords=['junk','bought','like','dont','just','use','buy','work','small','didnt','did','disappointed']))
	cur_number+=1
	tree_size+=1
	f8_root_left_leaf = LabelNode(number=cur_number, label=ABSTAIN, pairs={SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f8_root_right_child = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=['shoes','metal','fabric','replace','battery','warranty','plug']))
	cur_number+=1
	tree_size+=1
	f8_root_right_child_left_leaf = LabelNode(number=cur_number, label=ABSTAIN, pairs={SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f8_root_right_child_right_leaf = LabelNode(number=cur_number, label=SPAM, pairs={SPAM:[], HAM:[]}, used_predicates=set([]))
	
	f8_root.left = f8_root_left_leaf
	f8_root_left_leaf.parent=f8_root

	f8_root.right = f8_root_right_child
	f8_root_right_child.parent=f8_root

	f8_root_right_child.left = f8_root_right_child_left_leaf
	f8_root_right_child_left_leaf.parent=f8_root_right_child
	f8_root_right_child.right = f8_root_right_child_right_leaf
	f8_root_right_child_right_leaf.parent=f8_root_right_child
	f8 = TreeRule(rtype='lf', root=f8_root, size=tree_size, max_node_id=cur_number, is_good=True)


	# f8 = keyword_labelling_func_builder(['junk','bought','like','dont','just','use','buy','work','small','didnt','did','disappointed',
	# 	'shoes','metal','fabric','replace','battery','warranty','plug'], SPAM)

	f9  = keyword_labelling_func_builder(['love', 'perfect', 'loved', 'nice', 'excellent', 'works', 'loves', 'awesome', 'easy'], HAM)

	# f10 is an estension of f9
	cur_number=1
	tree_size=1
	f10_root = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=['love', 'perfect', 'loved', 'nice', 'excellent', 'works', 'loves', 'awesome', 'easy']))
	cur_number+=1
	tree_size+=1
	f10_root_left_leaf = LabelNode(number=cur_number, label=ABSTAIN, pairs={SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f10_root_right_child = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=['stars', 'soft']))
	cur_number+=1
	tree_size+=1
	f10_root_right_child_left_leaf = LabelNode(number=cur_number, label=ABSTAIN,  pairs={SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f10_root_right_child_right_leaf = LabelNode(number=cur_number, label=HAM,  pairs={SPAM:[], HAM:[]}, used_predicates=set([]))
	f10_root.left = f10_root_left_leaf
	f10_root_left_leaf.parent=f10_root
	f10_root.right = f10_root_right_child
	f10_root_right_child.parent=f10_root
	f10_root_right_child.left = f10_root_right_child_left_leaf
	f10_root_right_child_left_leaf.parent=f10_root_right_child
	f10_root_right_child.right = f10_root_right_child_right_leaf
	f10_root_right_child_right_leaf.parent=f10_root_right_child
	f10 = TreeRule(rtype='lf', root=f10_root,size=tree_size, max_node_id=cur_number, is_good=True)
	# f10  = keyword_labelling_func_builder(['love', 'perfect', 'loved', 'nice', 'excellent', 'works', 'loves', 'awesome', 'easy', 'stars', 'soft'], HAM)


	# f11 is an estension of f9
	cur_number=1
	tree_size=1
	f11_root = PredicateNode(number=cur_number,pred=KeywordPredicate(keywords=['love', 'perfect', 'loved', 'nice', 'excellent', 'works', 'loves', 'awesome', 'easy']))
	cur_number+=1
	tree_size+=1
	f11_root_left_leaf = LabelNode(number=cur_number,label=ABSTAIN, pairs={SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f11_root_right_child = PredicateNode(number=cur_number,pred=KeywordPredicate(keywords=['shoes', 'bought', 'use', 'purchase', 'purchased', 'colors', 'install', 'clean', 'design',
	 'pair', 'screen', 'comfortable', 'products']))
	cur_number+=1
	tree_size+=1
	f11_root_right_child_left_leaf = LabelNode(number=cur_number,label=ABSTAIN, pairs={SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f11_root_right_child_right_leaf = LabelNode(number=cur_number,label=HAM, pairs={SPAM:[], HAM:[]}, used_predicates=set([]))
	f11_root.left = f11_root_left_leaf
	f11_root_left_leaf.parent=f11_root
	f11_root.right = f11_root_right_child
	f11_root_right_child.parent=f11_root
	f11_root_right_child.left = f11_root_right_child_left_leaf
	f11_root_right_child_left_leaf.parent=f11_root_right_child
	f11_root_right_child.right = f11_root_right_child_right_leaf
	f11_root_right_child_right_leaf.parent=f11_root_right_child
	f11 = TreeRule(rtype='lf', root=f11_root,size=tree_size, max_node_id=cur_number, is_good=True)

	# f11 = keyword_labelling_func_builder(['love', 'perfect', 'loved', 'nice', 'excellent', 'works', 'loves', 'awesome', 'easy', 'stars', 'soft',
	# 	'shoes', 'bought', 'use', 'purchase', 'purchased', 'colors', 'install', 'clean', 'design', 'pair', 'screen', 'comfortable', 'products'], HAM)

	f12  = keyword_labelling_func_builder(['returned','broke','battery','cable','fits','install','sturdy','ordered','usb','replacement','brand','installed','unit',
		'batteries','box','warranty','defective','cheaply','durable','advertised'], SPAM)
	f13 = keyword_labelling_func_builder(['cute','shirt'], HAM)
	f14 = keyword_labelling_func_builder(['fabric','return','money','poor','garbage','poorly','terrible','useless','horrible','returning','flimsy'], SPAM)
	f15 = keyword_labelling_func_builder(['pants','looks','toy','color','camera','water','phone','bag','worked','arrived','lasted'], SPAM)

	amazon_rules = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15]
	return amazon_rules
