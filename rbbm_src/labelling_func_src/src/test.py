from lfs import all_LFs
from bottom_up import keyword_search
from itertools import chain, combinations

l = [[x, []] for x in all_LFs]

s = "hey this"


# for f in l:
# 	f[1].extend(keyword_search(s, f[0]))


# print(l)

# def all_subsets(ss):
# 	"""
# 	given a sentence generate all the combination of words as sets( a list of tuples)
# 	"""
# 	wlist = list(set([w.lower() for w in ss.split()]))
# 	return chain(*map(lambda x: combinations(wlist, x), range(1, len(wlist)+1)))

	
# ts = list(all_subsets(s))
# exp_diff_list = []
# for t in ts:
# 	exp_diff_list.append([t, len(t)])


# print(exp_diff_list)

x = {'1': [0.3333333333333333, frozenset({'keyword_shakira', 'keyword_my'})], 
'2': [0.5, frozenset({'textblob_polarity'})], 
'3': [0.3333333333333333, frozenset({'keyword_shakira', 'check'})], 
'4': [0.3333333333333333, frozenset({'keyword_shakira', 'check'})], 
'5': [0.3333333333333333, frozenset({'keyword_my', 'check'})], 
'6':  [0.25, frozenset({'keyword_shakira', 'keyword_subscribe', 'keyword_my'})], 
'7': [0.5, frozenset({'keyword_shakira'})], 
'8': [-1]}

result  = [k for k, v in sorted(x.items(), key=lambda item: item[1], reverse=True)][0:5]

print(result)