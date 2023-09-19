# entry point / driver code
import logging
import argparse
import rbbm_src.logconfig
from rbbm_src.classes import (
	StatsTracker,
	lf_input,
	dc_input)
from rbbm_src.labelling_func_src.src.experiment import lf_main 
from rbbm_src.dc_src.src.experiment import dc_main 
import psycopg2
import logging
import colorful



def main():
	"""
	entry point function for the system, it accepts an input object
	which is either  lf_input or dc_input
	"""

	# file_handler=logging.FileHandler('log_general.txt', 'a')
	# file_handler.setFormatter(file_formatter)
	logger = logging.getLogger(__name__)

	parser = argparse.ArgumentParser(description='Running experiments of LabelExplaination')
	# common arguments needed for either DC or LF

	parser.add_argument('-l', '--log_level', metavar='\b', type=str, default='critical',
	help='loglevel: debug/info/warning/error/critical (default: %(default)s)')

	parser.add_argument('-p', '--user_provide', metavar='\b', type=str, default='True',
	help='user select from all wrong labels(for LF) or all wrong repairs(for DC)? (default: %(default)s)')

	parser.add_argument('-U', '--use_case', metavar='\b', type=str, default='dc',
		help='use case of the run, is it for dc or lf? (default: %(default)s)')

	parser.add_argument('-R', '--sample_contingency', metavar='\b', type=str, default='True',
	help='when evaluating responsibility, sample contingency? (default: %(default)s)')

	parser.add_argument('-Z', '--contingency_sample_times', metavar='\b', type=str, default='4',
	help='if choosing uniform sampling contingency, how many times do you want to sample? (default: %(default)s)')

	parser.add_argument('-K', '--optimize_using_clustering', metavar='\b', type=str, default='False',
	help='when calculating responsibilities, use clustering method? (default: %(default)s)')

	parser.add_argument('-N', '--random_number_for_complaint', metavar='\b', type=str, default='999',
	help='random number chosen to select a complaint: given a list of complaint L with size N, we choose L[N mod len(L)](default: %(default)s)')

	parser.add_argument('-d', '--dbname', metavar= '\b', type=str, default='holo',
	help='dbname used during application (default: %(default)s)')

	parser.add_argument('-u', '--dbuser', metavar= '\b', type=str, default='postgres',
	help='dbuser used during application (default: %(default)s)')

	parser.add_argument('-P', '--dbpaswd', metavar= '\b', type=str, default='abcd1234',
	help='dbname used during application (default: %(default)s)')

	# arguments needed for LFs labelling function

	parser.add_argument('-M','--training_model_type', metavar="\b", type=str, default='snorkel',
	help='the model used to get the label: majority/snorkel (default: %(default)s)')

	parser.add_argument('-W', '--word_threshold', metavar='\b', type=int, default=0,
	help='word threshold when evaluating inflences of words(default: %(default)s)')

	# parser.add_argument('-g', '--greedy', metavar='\b', type=str, default='False',
	# help='early stop if no increase in terms of words influence (default: %(default)s)')

	parser.add_argument('-A', '--cardinality_thresh', metavar='\b', type=int, default=4,
	help='cardinality threshold if non greedy (i.e. exhaustive), ONLY userful when greedy==False (default: %(default)s)')

	parser.add_argument('-L', '--using_lattice', metavar='\b', type=str, default='False',
	help='using lattice when fiding rule influences? (default: %(default)s)')

	parser.add_argument('-E', '--eval_mode', metavar='\b', type=str, default='single_func',
	help='method used to evaluate the model (default: %(default)s)')

	parser.add_argument('-D', '--dataset_name', metavar='\b', type=str, default='enron',
	help='dataset used in the use case of labelling functions (default: %(default)s)' )

	# ----------------------------------------------------------------------------------------------------

	# stuff needed for DCs
	# conn = psycopg2.connect(dbname="holo", user="holocleanuser", password="abcd1234")

	parser.add_argument('-c', '--dc_dir', metavar='\b', type=str, default='/home/opc/chenjie/RBBM/experiments/dc/',
	help='holoclean needs a input text file which contains the denial constraints, this will be the dir it finds the file (default: %(default)s)')

	# parser.add_argument('-C', '--dc_file', metavar='\b', type=str, default='dc_finder_adult_rules.txt',
	# help='holoclean needs a input csv file as the starting point, this will be the dir it finds the file')

	parser.add_argument('-C', '--dc_file', metavar='\b', type=str, default='dc_sample_30',
	help='holoclean needs a input text file which contains the denial constraints, this will be the file inside dc_dir (default: %(default)s)')

	parser.add_argument('-s', '--input_csv_dir', metavar='\b', type=str, default='/home/opc/chenjie/RBBM/experiments/dc/',
	help='holoclean needs a input csv file as the starting point, this will be the dir it finds the file (default: %(default)s)')

	parser.add_argument('-S', '--input_csv_file', metavar='\b', type=str, default='adult500.csv',
	help='holoclean needs a input csv file as the starting point, this will the file inside input_csv_dir (default: %(default)s)')

	parser.add_argument('-t', '--ground_truth_dir', metavar='\b', type=str, default='/home/opc/chenjie/RBBM/experiments/dc/',
	help='holoclean needs ground truth file to evaluate, this will be the dir it finds the file (default: %(default)s)')

	parser.add_argument('-T', '--ground_truth_file', metavar='\b', type=str, default='adult500_clean.csv',
	help='holoclean needs ground truth file to evaluate, this will be the file inside ground_truth_dir (default: %(default)s)')

	parser.add_argument('-H', '--contingency_size_threshold', metavar='\b', type=str, default='2',
	help='if enumerate and test contingency, up to what size do you want to try (default: %(default)s)')

	parser.add_argument('-O', '--prune_only', metavar='\b', type=str, default='False',
	help='stop after pruning to see some properties (default: %(default)s)')


	args=parser.parse_args()

	logger.critical(args)
	if(args.sample_contingency=='True'):
		sample_contingency=True
	else:
		sample_contingency=False
	if(args.user_provide=='True'):
		user_provide=True
	else:
		user_provide=False
	if(args.prune_only=='True'):
		prune_only=True
	else:
		prune_only=False
	if(args.optimize_using_clustering):
		clustering_responsibility=True
	else:
		clustering_responsibility=False

	conn = psycopg2.connect(dbname=args.dbname, user=args.dbuser, password=args.dbpaswd)

	if(args.use_case=='dc'):
		# conn = psycopg2.connect(dbname=args.dbname, user=args.dbuser, password=args.dbpaswd)
		input_arg_obj = dc_input(connection=conn,
			contingency_size_threshold=int(args.contingency_size_threshold),
			contingency_sample_times=int(args.contingency_sample_times),
			sample_contingency=sample_contingency,
			user_provide=user_provide,
			input_dc_dir=args.dc_dir,
			input_dc_file=args.dc_file,
			input_csv_dir=args.input_csv_dir,
			input_csv_file=args.input_csv_file,
			ground_truth_dir=args.ground_truth_dir,
			ground_truth_file=args.ground_truth_file,
			random_number_for_complaint=int(args.random_number_for_complaint),
			stats=StatsTracker(),
			prune_only=prune_only,
			clustering_responsibility=clustering_responsibility)
	else:
		if(args.using_lattice=='True'):
			using_lattice=True
		else:
			using_lattice=False
			
		input_arg_obj=lf_input(connection=conn,
			contingency_size_threshold=int(args.contingency_size_threshold),
			contingency_sample_times=int(args.contingency_sample_times),
			sample_contingency=sample_contingency,
			log_level=args.log_level,
		    user_provide=user_provide,
		    training_model_type=args.training_model_type,
		    word_threshold=args.word_threshold,
		    greedy=True,
		    cardinality_thresh=args.cardinality_thresh,
		    using_lattice=using_lattice,
		    eval_mode=args.eval_mode,
		    invoke_type='terminal',
		    arg_str='',
		    topk=100,
		    random_number_for_complaint=int(args.random_number_for_complaint),
		    # lattice_dict=args.lattice_dict,
		    # lfs=args.lfs,
		    # sentences_df=args.sentences_df,
		    dataset_name=args.dataset_name,
		    stats=StatsTracker(),
		    prune_only=prune_only,
		    clustering_responsibility=clustering_responsibility)

	if(isinstance(input_arg_obj, lf_input)):
		lf_main(input_arg_obj)
	if(isinstance(input_arg_obj, dc_input)):
		dc_main(input_arg_obj)


if __name__ == '__main__':
	main()
