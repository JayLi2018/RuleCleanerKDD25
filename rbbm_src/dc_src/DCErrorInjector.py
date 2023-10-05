import pandas as pd
import random
from math import floor

def inject_error(df,  exclude_cols=[], percent_of_errors=-1, number_of_errors=-1, col_to_inject=[]):

	if(number_of_errors!=-1):
		number_of_errors = number_of_errors
	elif(percent_of_errors!=-1):
		x, y = df.shape
		number_of_errors = floor(x*y*percent_of_errors)
	if(col_to_inject):
		cols = col_to_inject
	else:
		cols = [x for x in list(df) if x not in exclude_cols] 

	domain_dict = {}
	# print(f"cols: {cols}")
	for c in cols:
		domain_dict[c] = list(df[c].unique())

	# print(f"domain_dict:{domain_dict}")
	df['_tid_'] = range(len(df))
	df['is_dirty'] = 'False'

	row_ids = list(df['_tid_'])
	errored_row_ids = set()
	cur_error_cnt=0
	while(cur_error_cnt<number_of_errors):
		# print(f"cols: {cols}")
		# choose a column
		# choose a row
		# replace the correct value with the wrong value, set is_dirty=True
		c_to_inject = random.choices(cols)[0]
		# print(f'col_to_inject: {c_to_inject}')
		candidate_values = domain_dict[c_to_inject]
		r_to_inject = random.choices(row_ids)[0]
		while(r_to_inject in errored_row_ids):
			r_to_inject = random.choices(row_ids)[0]
		errored_row_ids.add(r_to_inject)
		val = random.choices(candidate_values)[0]
		# print(str(df.loc[df['_tid_']==r_to_inject, c_to_inject]))
		# print(val)
		while(str(df.loc[df['_tid_']==r_to_inject, c_to_inject])==val):
			val = random.choices(candidate_values)
		df.loc[df['_tid_']==r_to_inject, c_to_inject]=val
		df.loc[df['_tid_']==r_to_inject, 'is_dirty']='True'
		cur_error_cnt+=1

	return df


if __name__ == '__main__':
	# Sample data for NBA players and teams
	data = {
	    'First Name': ['LeBron', 'Stephen', 'Kevin', 'Kawhi', 'Giannis', 'Anthony', 'James', 'Luka', 'Damian', 'Chris',
	                   'Devin', 'Jayson', 'Joel', 'Nikola', 'Zion', 'Russell', 'Ben', 'Paul', 'Bradley'],
	    'Last Name': ['James', 'Curry', 'Durant', 'Leonard', 'Antetokounmpo', 'Davis', 'Harden', 'Dončić', 'Lillard',
	                  'Paul', 'Booker', 'Tatum', 'Embiid', 'Jokić', 'Williamson', 'Westbrook', 'Simmons', 'George', 'Beal'],
	    'Team': ['Los Angeles Lakers', 'Golden State Warriors', 'Brooklyn Nets', 'LA Clippers', 'Milwaukee Bucks',
	             'Los Angeles Lakers', 'Brooklyn Nets', 'Dallas Mavericks', 'Portland Trail Blazers', 'Phoenix Suns',
	             'Phoenix Suns', 'Boston Celtics', 'Philadelphia 76ers', 'Denver Nuggets', 'New Orleans Pelicans',
	             'Washington Wizards', 'Philadelphia 76ers', 'LA Clippers', 'Washington Wizards']
	}

	# Create the Pandas DataFrame
	df = pd.DataFrame(data)

	print(inject_error(df=df, percent_of_errors=0.2))