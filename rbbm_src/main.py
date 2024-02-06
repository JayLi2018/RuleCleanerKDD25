# entry point / driver code
import logging
import argparse
import rbbm_src.logconfig
from rbbm_src.classes import (
	StatsTracker,
	lf_input,
	dc_input)
from rbbm_src.labelling_func_src.src.LFRepair import lf_main 
import psycopg2
import logging
import colorful



def main():
	"""
	entry point function for the system, it accepts an input object
	"""

	# file_handler=logging.FileHandler('log_general.txt', 'a')
	# file_handler.setFormatter(file_formatter)
	logger = logging.getLogger(__name__)

	parser = argparse.ArgumentParser(description='Running experiments of LFRepair')
	# common arguments needed for LF

	parser.add_argument('-U', '--use_case', metavar='\b', type=str, default='lf',
		help='use case of the run, is it for dc or lf? (default: %(default)s)')

	parser.add_argument('-e', '--experiment_name',  type=str, default='test_blah',
	  help='the name of the experiment, the results will be stored in the directory named with experiment_name_systime (default: %(default)s)')

	parser.add_argument('-R', '--repeatable',  type=str, default='true',
	  help='repeatable? (default: %(default)s)')

	parser.add_argument('-x', '--seed',  type=int, default=123,
	  help='if repeatable, specify a seed number here (default: %(default)s)')

	parser.add_argument('-X', '--seed_file',  type=str, default='seeds.txt',
	  help='if repeatable, specify a seed number here (default: %(default)s)')

	parser.add_argument('-E', '--retrain_every_percent',   type=float, default=1,
	  help='retrain over every (default: %(default)s*100), the default order is sorted by treesize ascendingly')

	parser.add_argument('-A', '--retrain_accuracy_thresh',   type=float, default=1,
	  help='when retrain over every retrain_every_percent, the algorithm stops when the fix rate is over this threshold (default: %(default)s)')

	parser.add_argument('-T', '--pre_filter_thresh', metavar='\b', type=float, default=0,
	help='prefilter those rules that have number of tuples involved in violations above this thresh (default: %(default)s)')

	parser.add_argument('-l', '--log_level', metavar='\b', type=str, default='debug',
	help='loglevel: debug/info/warning/error/critical (default: %(default)s)')

	parser.add_argument('-s', '--user_input_size', metavar='\b', type=int, default=20,
	help='user input size total (the complaint size is decided by user_input_size*complaint_ratio) and confirm size is decided by user_input_size-complaint_size(default: %(default)s)')

	parser.add_argument('-r', '--complaint_ratio', metavar='\b', type=float, default=0.5,
	help='user input complaint ratio (default: %(default)s)')

	parser.add_argument('-G','--strategy',  type=str, default='information_gain',
	  help='method used to repair the rules (naive, information_gain, optimal) (default: %(default)s)')

	parser.add_argument('-D', '--deletion_factor',   type=float, default=0.5,
	  help='this is a factor controlling how aggressive the algorithm chooses to delete the rule from the rulset (default: %(default)s)')

	parser.add_argument('-W', '--deletion_absolute_threshold',   type=int, default=10,
	  help='this is threshold for absolute tree size increase (default: %(default)s)')

	parser.add_argument('-b', '--deletion_type',   type=str, default='ratio',
	  help='deletion type (ratio/absolute) (default: %(default)s)')

	# Arguments for LF use case only:
	parser.add_argument('-d','--dbname',  type=str, default='label',
	  help='database name which stores the dataset, (default: %(default)s)')

	parser.add_argument('-P','--port',  type=int, default=5433,
	  help='database port, (default: %(default)s)')

	parser.add_argument('-p', '--password',  type=int, default=5432,
	  help='database password, (default: %(default)s)')

	parser.add_argument('-u', '--user',  type=str, default='postgres',
	  help='database user, (default: %(default)s)')

	parser.add_argument('-f','--lf_source',  type=str, default='undefined',
	  help='the source of labelling function (intro / system generate) (default: %(default)s)')
	
	parser.add_argument('-O','--number_of_funcs',  type=int, default=20,
	  help='if if_source is selected as system generate, how many do you want(default: %(default)s)')

	parser.add_argument('-i', '--run_intro',  action="store_true")

	parser.add_argument('-z', '--run_amazon',  action="store_true")
	
	parser.add_argument('-w', '--run_painter',  action="store_true")
	
	parser.add_argument('-o', '--run_professor',  action="store_true")

	parser.add_argument('-k', '--load_funcs_from_pickle',   type=str, default='false',
	  help='(flag indicating if we want to load functions from a pickle file default: %(default)s)')

	parser.add_argument('-K', '--pickle_file_name',   type=str, default='placeholder_name',
	  help='(if load_funcs_from_pickle, then heres the pickle file name : %(default)s)')

	parser.add_argument('-M','--training_model_type',  type=str, default='snorkel',
	help='the model used to get the label: majority/snorkel (default: %(default)s)')

	parser.add_argument('-n', '--dataset_name', metavar='\b', type=str, default='youtube',
	help='dataset used in the use case of labelling functions (default: %(default)s)' )

	parser.add_argument('-t', '--table_name', metavar='\b', type=str, default='tax',
	help='the table name from database cr that you want to work with (default: %(default)s)')

	parser.add_argument("--run-gpt-rules", action="store_true")
	parser.add_argument("--gpt-dataset", type=str, metavar='\b', default='youtube', help="youtube/amazon/pt/pa")
	parser.add_argument("--gpt-pickled-rules-dir", metavar='\b', type=str, default='/home/opc/chenjie/RBBM/chatgpt_rbbm/chatgpt_rules/')

	
	args = parser.parse_args()

	logger.critical(args)
	
	conn = psycopg2.connect(dbname=args.dbname, user=args.user, password=args.password, port=args.port)

	input_arg_obj =lf_input(
	strat=args.strategy,
	complaint_ratio=args.complaint_ratio,
	user_input_size=args.user_input_size,
	connection=conn,
	log_level=args.log_level,
	training_model_type=args.training_model_type,
	number_of_funcs=args.number_of_funcs,
	experiment_name=args.experiment_name,
	repeatable=args.repeatable,
	rseed=args.seed,
	run_intro=args.run_intro,
	run_amazon=args.run_amazon,
	run_painter=args.run_painter,
	run_professor=args.run_professor,
	retrain_every_percent=args.retrain_every_percent,
	deletion_factor=args.deletion_factor,
	retrain_accuracy_thresh=args.retrain_accuracy_thresh,
	load_funcs_from_pickle=args.load_funcs_from_pickle,
	pickle_file_name=args.pickle_file_name,
	seed_file=args.seed_file,
	pre_deletion_threshold=args.pre_filter_thresh,
	dataset_name=args.dataset_name,
	stats=StatsTracker(),
	lf_source=args.lf_source,
	deletion_absolute_threshold=args.deletion_absolute_threshold,
	deletion_type=args.deletion_type,
	run_gpt_rules=args.run_gpt_rules,
	gpt_dataset=args.gpt_dataset,
	gpt_pickled_rules_dir=args.gpt_pickled_rules_dir
	)
	
	lf_main(input_arg_obj)


if __name__ == '__main__':
	main()
