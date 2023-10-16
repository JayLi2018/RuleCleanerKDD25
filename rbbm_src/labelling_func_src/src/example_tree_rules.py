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


def gen_professor_teacher_funcs():
	# HAM: teacher # SPAM: professor
	# bias_pt
	# \begin{lfList}
	# \item \lfClassB{\textbf{teacher:} students{\lfOr}teacher}\lfStats{(coverage: 22\%, accuracy: 81\%)}
	# \item \lfClassA{\textbf{professor:} research}\lfStats{(coverage: 31\%, accuracy: 92\%)}
	# \item \lfClassB{\textbf{teacher:} husband{\lfOr}loves{\lfOr}lives}\lfStats{(coverage: 7\%, accuracy: 85\%)}
	# \item \lfClassA{\textbf{professor:} ph{\lfOr}phd{\lfOr}postdoctoral}\lfStats{(coverage: 17\%, accuracy: 94\%)}
	# \item \lfClassB{\textbf{teacher:} com{\lfOr}years{\lfOr}music{\lfOr}children{\lfOr}time{\lfOr}classroom{\lfOr}teachers{\lfOr}teach{\lfOr}enjoys{\lfOr}help{\lfOr}schools}\lfStats{(coverage: 39\%, accuracy: 72\%)}
	# \item \lfClassNull{book{\lfOr}press{\lfOr}author{\lfOr}appeared{\lfOr}politics{\lfOr}political}
	# \item \lfClassA{\textbf{professor:} medical{\lfOr}medicine{\lfOr}clinical{\lfOr}health}\lfStats{(coverage: 11\%, accuracy: 86\%)}
	# \item \lfClassNull{people{\lfOr}life{\lfOr}books}
	# \item \lfClassA{\textbf{professor:} engineering{\lfOr}systems{\lfOr}management{\lfOr}computer}\lfStats{(coverage: 12\%, accuracy: 86\%)}
	# \item \lfClassNull{english{\lfOr}literature{\lfOr}writing{\lfOr}art}
	# \end{lfList}
	TreeRule.rule_counter=0
	f1 = keyword_labelling_func_builder(['students', 'teacher'], HAM)
	f2 = keyword_labelling_func_builder(['research'], SPAM)
	f3 = keyword_labelling_func_builder(['husband','loves','lives'], HAM)
	f4 = keyword_labelling_func_builder(['ph','phd','postdoctoral'], SPAM)
	f5 = keyword_labelling_func_builder(['com','years','music','children','time','classroom','teachers','teach','enjoys','help','schools'], HAM)
	f6 = keyword_labelling_func_builder(['medical','medicine','clinical','health'], SPAM)
	f7 = keyword_labelling_func_builder(['engineering','systems','management','computer'], SPAM)
	professor_teacher_funcs = [f1, f2, f3, f4, f5, f6, f7]

	return professor_teacher_funcs

def gen_painter_architecht_funcs():
	# HAM: painter # SPAM: architect
	# bias_pa
	# \begin{lfList}
	# \item \lfClassB{\textbf{painter:} paintings{\lfOr}painting}\lfStats{(coverage: 25\%, accuracy: 99\%)}
	# \item \lfClassA{\textbf{architect:} architecture{\lfOr}development{\lfOr}architect{\lfOr}software{\lfOr}architectural{\lfOr}solutions{\lfOr}management}\lfStats{(coverage: 34\%, accuracy: 95\%)}
	# \item \lfClassB{\textbf{painter:} art{\lfOr}gallery{\lfOr}artist}\lfStats{(coverage: 30\%, accuracy: 88\%)}
	# \item \lfClassA{\textbf{architect:} data{\lfOr}enterprise}\lfStats{(coverage: 5\%, accuracy: 98\%)}
	# \item \lfClassA{\textbf{architect:} experience{\lfOr}projects}\lfStats{(coverage: 19\%, accuracy: 86\%)}
	# \item \lfClassB{\textbf{painter:} paints{\lfOr}life{\lfOr}paint{\lfOr}color}\lfStats{(coverage: 15\%, accuracy: 89\%)}
	# \item \lfClassA{\textbf{architect:} microsoft{\lfOr}technologies{\lfOr}business{\lfOr}services{\lfOr}applications{\lfOr}web{\lfOr}application}\lfStats{(coverage: 13\%, accuracy: 91\%)}
	# \item \lfClassB{\textbf{painter:} york}\lfStats{(coverage: 8\%, accuracy: 71\%)}
	# \item \lfClassB{\textbf{painter:} colors}\lfStats{(coverage: 2\%, accuracy: 97\%)}
	# \item \lfClassB{\textbf{painter:} images{\lfOr}work{\lfOr}arts{\lfOr}born{\lfOr}school{\lfOr}landscape}\lfStats{(coverage: 43\%, accuracy: 70\%)}
	# \end{lfList}
	TreeRule.rule_counter=0
	f1 = keyword_labelling_func_builder(['paintings', 'painting'], HAM)
	f2 = keyword_labelling_func_builder(['architecture','architect','software','architectural','solutions','management'], SPAM)
	f3 = keyword_labelling_func_builder(['art','gallery','artist'], HAM)
	f4 = keyword_labelling_func_builder(['data','enterprise'], SPAM)
	f5 = keyword_labelling_func_builder(['experience','projects'], SPAM)
	f6 = keyword_labelling_func_builder(['microsoft','technologies','business','services','applications','web','application'], SPAM)
	f7 = keyword_labelling_func_builder(['paints','life','paint','color'], HAM)
	f8 = keyword_labelling_func_builder(['york'], HAM)
	f9 = keyword_labelling_func_builder(['colors'], HAM)
	f10 = keyword_labelling_func_builder(['images','work','arts','born','school','landscape'], HAM)

	painter_architecht_funcs = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]

	return painter_architecht_funcs
