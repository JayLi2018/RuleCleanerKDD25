# this file has 2 functions:
# 1.add_noise:
	# create a noisy dataset by randomly replacing some ground truth
	# with some different values in that column domain
# 2.format_to_holoclean_gt:
	# format the original file to ground truth

import pandas as pd
import random
import math
import string

def add_noise(df, sample_rate=1, noise_percentage=0.01, random_replace_rate=0.2):
	"""
	df: dataframe
	sample_rate: sample (if needed) dataframe as the real input df to work with
	noise_percentage: percentage of the cells you want to "pollute"
	random_replace_rate: instead of using domain value, we replace with some arbitrary nonsense
	"""
	if(sample_rate<1):
		df = df.sample(n=sample_rate, random_state=1)
	row, col = df.shape 
	num_cells = row * col
	num_noise = num_cells * noise_percentage
	domain_values = {}
	# iloc[row, col]
	for c in list(df):
	    domain_values[c]=list(df[c].unique())


	i=1
	while(i<=math.floor(num_noise*(1-random_replace_rate))):
		rrow = random.randint(0, row-1)
		rcol = random.randint(0, col-1)
		col_name = list(df.iloc[rrow:rrow+1,rcol:rcol+1])[0]
		df.iloc[rrow,rcol] = domain_values[col_name][random.randint(0, len(domain_values[col_name])-1)]
		i+=1
		# print(f"i={i}, col_name={col_name}, {len(df.iloc[rrow,rcol])}, rand_str:{domain_values[col_name][random.randint(0, len(domain_values[col_name])-1)]}")

	j=1
	while(j<=math.floor(num_noise*random_replace_rate)):
		rrow = random.randint(0, row-1)
		rcol = random.randint(0, col-1)
		col_name = list(df.iloc[rrow:rrow+1,rcol:rcol+1])[0]
		randomstr = ''.join([random.choice(string.ascii_lowercase) for x in \
			range(len(df.iloc[rrow,rcol]))])
		df.iloc[rrow,rcol] = randomstr
		j+=1
		# print(f"j={j}, col_name={col_name}, {len(df.iloc[rrow,rcol])}, rand_str:{randomstr}")

	return df

def gen_gt_df(df):
	cols=list(df)
	dict_list = []
	for index, row in df.iterrows():
	    for c in cols:
	        d = {'tid':index, 'attribute':c, 'correct_val':row[c]}
	        dict_list.append(d)
	    index+=1
	gt_df=pd.DataFrame(dict_list)
	return gt_df

if __name__ == '__main__':
	df = pd.read_csv('testdata/Adult500.csv')
	noisy_df = add_noise(df=df)
	noisy_df.to_csv('Adult500_noisy.csv', index=False)
	gt_df=gen_gt_df(df)
	gt_df.to_csv('Adult500_clean.csv', index=False)

