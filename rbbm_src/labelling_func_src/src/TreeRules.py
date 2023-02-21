from textblob import TextBlob
from collections import deque
from typing import *
from dataclasses import dataclass
import re 

SPAM=1
HAM=0
ABSTAIN=-1
CLEAN=HAM
DIRTY=SPAM

def textblob_sentiment(x: str) -> float:
    scores = TextBlob(x)
    print(scores.sentiment.subjectivity)
    return scores.sentiment.subjectivity


@dataclass
class Node:
	"""
	node structure used in the tree rule
	"""
	number: int=0
	left: 'Node'=None
	right: 'Node'=None
	parent: 'Node'=None

class Predicate:
	def __init__(self, instance: str):
		pass

	def evaluate(self):
		pass

class KeywordPredicate(Predicate):

	def __init__(self, keyword:str):
		self.keyword=keyword

	def __repr__(self):
		return f"keyword_predicate-word-{self.keyword}"

	def __str__(self):
		return f"keyword_predicate-word-{self.keyword}"

	def evaluate(self,instance: str):
		print(f"{self.keyword} in instance? {self.keyword in instance}")
		return self.keyword in instance

class DCAttrPredicate(Predicate):
	def __init__(self, pred:str, operator:str):
		self.pred=pred
		self.operator=operator
		# print(self.pred)
		# print(re.findall(r'(t[12])\.[-\w]+,(t[12])\.[-\w]+', self.pred))
		# print(self.pred)
		self.pred_identifier= re.findall(r'(t[12])\.([-\w]+),(t[12])\.([-\w]+)', self.pred)[0] + (self.operator,)
		self.attr=self.pred_identifier[1]

	def __repr__(self):
		return f"dc-attr-pred-{self.operator}{self.pred}"
	def __str__(self):
		return f"dc-attr-pred-{self.operator}{self.pred}"
	def evaluate(self, dict_to_eval:dict):
	    # attr = re.search(r't[1|2]\.([-\w]+)', self.pred).group(1)
	    # print(f"dict_to_eval['t1']['{self.attr}']{self.operator}dict_to_eval['t2']['{self.attr}']")	    
	    return eval(f"dict_to_eval['t1']['{self.attr}']{self.operator}dict_to_eval['t2']['{self.attr}']")

class DCConstPredicate(Predicate):
	def __init__(self, pred:str, operator:str):
		self.pred=pred
		self.operator=operator
		# print(self.pred)
		self.pred_identifier = re.findall(r'(t[1|2])\.([-\w]+),[\"|\'](.*)[\"|\']',self.pred)[0]+ (self.operator,)
		self.role, self.attr, self.const, _ = self.pred_identifier

	def __repr__(self):
		return f"dc-const-pred-{self.pred}"
	
	def __str__(self):
		return f"dc-const-pred-{self.pred}"
	
	def evaluate(self, dict_to_eval:dict):
		# triples where each triple is in the format of (t1, attr, constant) (or t2)
		return eval(f"dict_to_eval['{self.role}']['{self.attr}']{self.operator}'{self.const}'")

class SentimentPredicate(Predicate):
	def __init__(self, thresh:float=0.5, sent_func:Callable=textblob_sentiment):
		self.thresh=thresh
		self.sent_func=sent_func

	def __repr__(self):
		return f"sentiment_predicate-thresh-{self.thresh}"

	def __str__(self):
		return f"sentiment_predicate-thresh-{self.thresh}"

	def evaluate(self, instance: str):
		return self.sent_func(instance)>self.thresh


@dataclass
class PredicateNode(Node):
	pred: Predicate=None


@dataclass
class LabelNode(Node):
	label: int=ABSTAIN
	pairs: dict=None
	used_predicates: set([])=None

# @dataclass
# DenialPredicate(Predicate):
# 	accessedtuples: List(int)

class TreeRule:
# 	"""
# 	tree like structure of rules
# 	some properties of the tree structure
# 	1. type:
# 		'lf': labelling function rule (snorkel flavor)
# 		'dc': denial constraints (holoclean DC flavor)
# 	2. __str__ / __repr__ string representation of the rules 
# 	"""
	def __init__(self, rtype: str, root: 'Node', size: int):
		self.rtype = rtype
		self.root = root
		self.size = size

	def setsize(self, new_size:int):
		self.size=new_size

	def __str__(self):
		str_list = []
		level=0
		queue = deque([(self.root, level)])
		while(queue):
			# print(f"level: {level}, queue: {queue}")
			cur_node, level = queue.popleft()
			# print(cur_node)
			if(isinstance(cur_node, PredicateNode)):
				str_list.append(level*'	'+str(cur_node.pred))
			else:
				str_list.append(level*'	'+str(cur_node.label))
			level+=1
			if(cur_node.left):
				queue.append((cur_node.left, level))
			if(cur_node.right):
				queue.append((cur_node.right, level))
		return '\n'.join(str_list)

	def __repr__(self):
		return self.__str__()

	def evaluate(self, instance: Union[str, dict]):
		"""
		return the result given the sentence/ a pair of tuples
		"""
		cur_node=self.root
		used_predicates = []
		while(cur_node):
			if(isinstance(cur_node, LabelNode)):
				for u in used_predicates:
					cur_node.used_predicates.add(u)
				return cur_node
			if(cur_node.pred.evaluate(instance)):
				used_predicates.append(cur_node.pred.pred_identifier)
				cur_node.right.parent=cur_node
				cur_node=cur_node.right
			else:
				cur_node.left.parent=cur_node
				cur_node=cur_node.left

	def serialize(self):
		# return the rule as the acceptyed format for DC/LF Models 
		# LF for Snorkel, Holoclean text format for DC
		res=[]
		if(self.rtype=='dc'):
			
			queue = deque([self.root])

			while(queue):
				cur_node = queue.popleft()
				if(isinstance(cur_node, LabelNode)):
					if(cur_node.label==DIRTY):
						# print("cur_node is dirty. converting the path from root to this dirty label")
						res.append(self.construct_dc_from_dirty(cur_node))
				if(cur_node.left):
					queue.append(cur_node.left)
				if(cur_node.right):
					queue.append(cur_node.right)
		return res

	def construct_dc_from_dirty(self, dirty_node):
		res = ['t1&t2']
		negate_cond=False
		cur_node = dirty_node

		while(cur_node.parent):
			# print(f"cur_node: {cur_node}, cur_node.parent: {cur_node.parent.parent}")
			if(cur_node.parent.left is cur_node):
				if(cur_node.parent.pred.operator=='!='):
					ori_op='IQ'
					replace_op='EQ'
				else:
					ori_op='EQ'
					replace_op='IQ'
				res.append(cur_node.parent.pred.pred.replace(ori_op, replace_op))
			else:
				res.append(cur_node.parent.pred.pred)
			cur_node = cur_node.parent
		# print(f"res: {res}")
		return '&'.join(res)


if __name__ == '__main__':

	x = 'subscribe my channel, I have some good songs'
	# # print(eval("'songs' in x"))
	# # code if song in sentence and sentiment > 0.5 - > HAM 
	# # else abstain
	r1n1 = PredicateNode(pred=KeywordPredicate(keyword='song'))
	r1n2 = LabelNode(label=ABSTAIN)
	r1n3 = PredicateNode(pred=SentimentPredicate(thresh=0.5, sent_func=textblob_sentiment))
	r1n4 = LabelNode(label=ABSTAIN)
	# r1n4.left=r1n6
	r1n5 = LabelNode(label=HAM)
	r1n1.left=r1n2
	r1n1.right=r1n3
	r1n3.left=r1n4
	r1n3.right=r1n5
	print(r1n1)
	print(r1n2)
	keyword_song_with_sentiment = TreeRule(rtype='lf', root=r1n1)
	print(keyword_song_with_sentiment)
	print(keyword_song_with_sentiment.evaluate(x))