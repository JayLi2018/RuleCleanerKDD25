# entry point / driver code
import logging
import argparse
import rbbm_src.logconfig
from rbbm_src.classes import (
	StatsTracker,
	lf_input,
	dc_input)
from rbbm_src.labelling_func_src.src.experiment import lf_main 
from rbbm_src.dc_src.DCRepair import dc_main 
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

	parser = argparse.ArgumentParser(description='Running experiments of LFRepair/DCRepair')
	# common arguments needed for either DC or LF

	parser.add_argument('-l', '--log_level', metavar='\b', type=str, default='debug',
	help='loglevel: debug/info/warning/error/critical (default: %(default)s)')

	parser.add_argument('-s', '--user_input_size', metavar='\b', type=str, default=20,
	help='user input size total (the complaint size is decided by user_input_size*complaint_ratio) and confirm size is decided by user_input_size-complaint_size(default: %(default)s)')

	parser.add_argument('-r', '--complaint_ratio', metavar='\b', type=float, default=0.5,
	help='user input complaint ratio (default: %(default)s)')

	parser.add_argument('-G','--strategy', metavar="\b", type=str, default='information gain',
	  help='method used to repair the rules (naive, information_gain, optimal) (default: %(default)s)')

	parser.add_argument('-D', '--deletion_factor',  metavar="\b", type=float, default=0.5,
	  help='this is a factor controlling how aggressive the algorithm chooses to delete the rule from the rulset (default: %(default)s)')

	parser.add_argument('-E', '--retrain_every_percent',  metavar="\b", type=float, default=1,
	  help='retrain over every (default: %(default)s*100), the default order is sorted by treesize ascendingly')

	parser.add_argument('-A', '--retrain_accuracy_thresh',  metavar="\b", type=float, default=0.5,
	  help='when retrain over every retrain_every_percent, the algorithm stops when the fix rate is over this threshold (default: %(default)s)')

	# parser.add_argument('-p', '--user_provide', metavar='\b', type=str, default='True',
	# help='user select from all wrong labels(for LF) or all wrong repairs(for DC)? (default: %(default)s)')

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

	# parser.add_argument('-A', '--cardinality_thresh', metavar='\b', type=int, default=4,
	# help='cardinality threshold if non greedy (i.e. exhaustive), ONLY userful when greedy==False (default: %(default)s)')

	parser.add_argument('-L', '--using_lattice', metavar='\b', type=str, default='False',
	help='using lattice when fiding rule influences? (default: %(default)s)')

	# parser.add_argument('-E', '--eval_mode', metavar='\b', type=str, default='single_func',
	# help='method used to evaluate the model (default: %(default)s)')

	parser.add_argument('-n', '--dataset_name', metavar='\b', type=str, default='enron',
	help='dataset used in the use case of labelling functions (default: %(default)s)' )

	# ----------------------------------------------------------------------------------------------------

	# stuff needed for DCs

	parser.add_argument('-C', '--dc_file', metavar='\b', type=str, default='/home/opc/chenjie/RBBM/rbbm_src/muse/data/mas/tax_rules.txt',
	help='holoclean needs a input text file which contains the denial constraints, this will be the file inside dc_dir (default: %(default)s)')

	parser.add_argument('-S', '--semantic_version', metavar='\b', type=str, default='ind',
	help='muse semantic version (ind/stage/end/step) (default: %(default)s)')

	parser.add_argument('-t', '--table_name', metavar='\b', type=str, default='tax',
	help='the table name from database cr that you want to work with (default: %(default)s)')

	parser.add_argument('-T', '--pre_filter_thresh', metavar='\b', type=float, default=1,
	help='prefilter those DCs that have number of tuples involved in violations above this thresh (default: %(default)s)')

	parser.add_argument('-F', '--desired_dcs_file', metavar='\b', type=str, default='/home/opc/chenjie/RBBM/rbbm_src/dc_src/user_desired_dcs.txt',
	help='the ground truth DCs that so called user think is correct (default: %(default)s)')

	parser.add_argument('-I', '--user_specify_pairs', metavar='\b', type=str, default='True',
	help='user specify pairs of violations to repair? (default: %(default)s)')

	args = parser.parse_args()

	logger.debug(args)

	if(args.use_case=='dc'):
		if(args.user_specify_pairs=='True'):
			user_specify_pairs=True
		else:
			user_specify_pairs=False
			
		# conn = psycopg2.connect(dbname=args.dbname, user=args.dbuser, password=args.dbpaswd)
		input_arg_obj = dc_input(
			dc_file=args.dc_file,
			stats=StatsTracker(),
			log_level=args.log_level,
			table_name=args.table_name,
			pre_filter_thresh=args.pre_filter_thresh,
			semantic_version=args.semantic_version,
			user_input_size=args.user_input_size,
			complaint_ratio=args.complaint_ratio,
			desired_dcs_file=args.desired_dcs_file,
			strategy=args.strategy,
			deletion_factor=args.deletion_factor,
			acc_threshold=args.retrain_accuracy_thresh,
			user_specify_pairs=user_specify_pairs,
			retrain_every_percent=args.retrain_every_percent
			)
	else:

		conn = psycopg2.connect(dbname=args.dbname, user=args.dbuser, password=args.dbpaswd)
		
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
