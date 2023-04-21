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

def add_noise(df, sample=True, sample_rate=1, sample_size=500, noise_percentage=0.01, random_replace_rate=0.2):
	"""
	df: dataframe
	sample_rate: sample (if needed) dataframe as the real input df to work with
	noise_percentage: percentage of the cells you want to "pollute"
	random_replace_rate: instead of using domain value, we replace with some arbitrary nonsense
	"""
	# Note: sample rate overrides sample_size
	if(sample):
		if(sample_rate<1):
			df = df.sample(frac=sample_rate, random_state=1)
		else:
			df = df.sample(n=sample_size)
	gt_df = df.copy(deep=True)
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
		print(f"repacing: row:{rrow},col:{rcol}")
		df.iloc[rrow,rcol] = domain_values[col_name][random.randint(0, len(domain_values[col_name])-1)]
		i+=1
		# print(f"i={i}, col_name={col_name}, {len(df.iloc[rrow,rcol])}, rand_str:{domain_values[col_name][random.randint(0, len(domain_values[col_name])-1)]}")

	j=1
	while(j<=math.floor(num_noise*random_replace_rate)):
		rrow = random.randint(0, row-1)
		rcol = random.randint(0, col-1)
		col_name = list(df.iloc[rrow:rrow+1,rcol:rcol+1])[0]
		str_to_replace = str(df.iloc[rrow,rcol])
		randomstr = ''.join([random.choice(string.ascii_lowercase) for x in \
			range(len(str_to_replace))])
		df.iloc[rrow,rcol] = randomstr
		j+=1
		print(f"rand_str:{randomstr}")

	return df, gt_df

def gen_gt_df(df):
	cols=list(df)
	dict_list = []
	tid=0
	for index, row in df.iterrows():
	    for c in cols:
	        d = {'tid':tid, 'attribute':c, 'correct_val':row[c]}
	        dict_list.append(d)
	    tid+=1
	gt_df=pd.DataFrame(dict_list)
	return gt_df

def gen_gt_given_tids(tids_to_keep, gt_df_full):
	cols=list(gt_df_full)
	gt_df_full.insert(0, '_tid_', range(0, len(gt_df_full)))
	gt_df_given_tids=df[df['_tid_'].isin(tids_to_keep)]

	dict_list = []
	for index, row in df.iterrows():
	    for c in cols:
	        d = {'tid':row['_tid_'], 'attribute':c, 'correct_val':row[c]}
	        dict_list.append(d)
	gt_df=pd.DataFrame(dict_list)
	return gt_df

if __name__ == '__main__':
	# df = pd.read_csv('Adult_full.csv')
	# df = df[['age','workclass','education','marital-status','occupation','relationship','race','sex','hours-per-week','native-country','income']]
	# noisy_df, gt_df = add_noise(df=df, sample=True, sample_rate=1, sample_size=500, noise_percentage=0.01, random_replace_rate=0.1)
	# noisy_df.to_csv('Adult500_noisy.csv', index=False)
	# gt_df_formated=gen_gt_df(gt_df)
	# gt_df_formated.to_csv('Adult500_clean.csv', index=False)

	# for size in [500, 1000, 2000, 4000, 8000, 16000, 32000]:
	# 	df = pd.read_csv('Adult_full.csv')
	# 	# df = df[['age','workclass','education','marital-status','occupation','relationship','race','sex','hours-per-week','native-country','income']]
	# 	noisy_df, gt_df = add_noise(df=df, sample=True, sample_rate=1, sample_size=size, noise_percentage=0.01, random_replace_rate=0.1)
	# 	noisy_df.to_csv(f'adult_samples/adult_{size}.csv', index=False)
	gt_df = pd.read_csv('/home/opc/chenjie/RBBM/experiments/dc/hospital_1k.csv')
	gt_df_formated=gen_gt_df(gt_df)
	gt_df_formated.to_csv(f'/home/opc/chenjie/RBBM/experiments/dc/gt_hospital_1k.csv', index=False)
