# input: directory storing the pickled_file
import glob
import pickle
import re
import pprint


def evaluate_sample(dir, pattern='pickle_run_'):
	for f in glob.glob(f"{dir}/{pattern}*no_sample.txt"):
		print(f)
		print("no_sample_dict")
		with (open(f, "rb")) as no_sample_f:
			resp_d_no_sample=pickle.load(no_sample_f)
			print(resp_d_no_sample)
		r_time=re.findall(r'[0-9]+', f)[1]
		# print(r_time)
		with (open(f"{dir}/{pattern}{r_time}.txt", 'rb')) as sample_f:
			resp_d_sample=pickle.load(sample_f)
			print("sample_dict")
			print(resp_d_sample)
		correct_cnt=0
		for k in resp_d_no_sample:
			if(resp_d_no_sample[k][0]==resp_d_sample[k][0]):
				correct_cnt+=1
		print(f"accuracy:{correct_cnt/len(resp_d_no_sample)}\n") 

if __name__ == '__main__':
	for pattern in ['pickle_run_dc_sample_10_','pickle_run_dc_sample_20_','pickle_run_dc_sample_30_']:
		evaluate_sample('/home/opc/author/labelling_explanation', pattern)