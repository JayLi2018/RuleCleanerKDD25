# entry point / driver code
import logging
import argparse
import rbbm_src.logconfig
from rbbm_src.classes import (
	StatsTracker,
	lf_input,
	dc_input)
from rbbm_src.labelling_func_src.src.LFRepair import lf_main 
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


	# Arguments for Both use cases:
	parser.add_argument('-U', '--use_case', metavar='\b', type=str, default='dc',
		help='use case of the run, is it for dc or lf? (default: %(default)s)')

	parser.add_argument('-e', '--experiment_name', metavar="\b", type=str, default='test_blah',
	  help='the name of the experiment, the results will be stored in the directory named with experiment_name_systime (default: %(default)s)')

	parser.add_argument('-R', '--repeatable', metavar="\b", type=str, default='true',
	  help='repeatable? (default: %(default)s)')

	parser.add_argument('-x', '--seed', metavar="\b", type=int, default=123,
	  help='if repeatable, specify a seed number here (default: %(default)s)')

	parser.add_argument('-X', '--seed_file', metavar="\b", type=str, default='seeds.txt',
	  help='if repeatable, specify a seed number here (default: %(default)s)')

	parser.add_argument('-E', '--retrain_every_percent',  metavar="\b", type=float, default=1,
	  help='retrain over every (default: %(default)s*100), the default order is sorted by treesize ascendingly')

	parser.add_argument('-A', '--retrain_accuracy_thresh',  metavar="\b", type=float, default=1,
	  help='when retrain over every retrain_every_percent, the algorithm stops when the fix rate is over this threshold (default: %(default)s)')

	parser.add_argument('-T', '--pre_filter_thresh', metavar='\b', type=float, default=0,
	help='prefilter those DCs that have number of tuples involved in violations above this thresh (default: %(default)s)')

	parser.add_argument('-l', '--log_level', metavar='\b', type=str, default='debug',
	help='loglevel: debug/info/warning/error/critical (default: %(default)s)')

	parser.add_argument('-s', '--user_input_size', metavar='\b', type=int, default=20,
	help='user input size total (the complaint size is decided by user_input_size*complaint_ratio) and confirm size is decided by user_input_size-complaint_size(default: %(default)s)')

	parser.add_argument('-r', '--complaint_ratio', metavar='\b', type=float, default=0.5,
	help='user input complaint ratio (default: %(default)s)')

	parser.add_argument('-G','--strategy', metavar="\b", type=str, default='information_gain',
	  help='method used to repair the rules (naive, information_gain, optimal) (default: %(default)s)')

	parser.add_argument('-D', '--deletion_factor',  metavar="\b", type=float, default=0.5,
	  help='this is a factor controlling how aggressive the algorithm chooses to delete the rule from the rulset (default: %(default)s)')

	parser.add_argument('-W', '--deletion_absolute_threshold',  metavar="\b", type=int, default=10,
	  help='this is threshold for absolute tree size increase (default: %(default)s)')

	parser.add_argument('-b', '--deletion_type',  metavar="\b", type=str, default='ratio',
	  help='deletion type (ratio/absolute) (default: %(default)s)')

	# Arguments for LF use case only:
	parser.add_argument('-d','--dbname', metavar="\b", type=str, default='label',
	  help='database name which stores the dataset, (default: %(default)s)')

	parser.add_argument('-P','--port', metavar="\b", type=int, default=5433,
	  help='database port, (default: %(default)s)')

	parser.add_argument('-p', '--password', metavar="\b", type=int, default=5432,
	  help='database password, (default: %(default)s)')

	parser.add_argument('-u', '--user', metavar="\b", type=str, default='postgres',
	  help='database user, (default: %(default)s)')

	parser.add_argument('-f','--lf_source', metavar="\b", type=str, default='undefined',
	  help='the source of labelling function (intro / system generate) (default: %(default)s)')

	parser.add_argument('-m','--dc_model_type', metavar="\b", type=str, default='muse',
	  help='the source of labelling function (intro / system generate) (default: %(default)s)')

	parser.add_argument('-O','--number_of_funcs', metavar="\b", type=int, default=20,
	  help='if if_source is selected as system generate, how many do you want(default: %(default)s)')

	parser.add_argument('-i', '--run_intro',  metavar="\b", type=str, default='false',
	  help='do you want to run the intro example with pre selected user input? (default: %(default)s)')

	parser.add_argument('-z', '--run_amazon',  metavar="\b", type=str, default='false',
	  help='do you want to run amazon with witan funcs? (need to put dataset_name as amazon) (default: %(default)s)')
	
	parser.add_argument('-w', '--run_painter',  metavar="\b", type=str, default='false',
	  help='do you want to run painter_architect with witan funcs? (need to put dataset_name as painter_architect (default: %(default)s)')
	
	parser.add_argument('-o', '--run_professor',  metavar="\b", type=str, default='false',
	  help='do you want to run professor_teacher with witan funcs? (need to put dataset_name as professor_teacher (default: %(default)s)')

	parser.add_argument('-k', '--load_funcs_from_pickle',  metavar="\b", type=str, default='false',
	  help='(flag indicating if we want to load functions from a pickle file default: %(default)s)')

	parser.add_argument('-K', '--pickle_file_name',  metavar="\b", type=str, default='placeholder_name',
	  help='(if load_funcs_from_pickle, then heres the pickle file name : %(default)s)')

	parser.add_argument('-M','--training_model_type', metavar="\b", type=str, default='snorkel',
	help='the model used to get the label: majority/snorkel (default: %(default)s)')

	parser.add_argument('-n', '--dataset_name', metavar='\b', type=str, default='youtube',
	help='dataset used in the use case of labelling functions (default: %(default)s)' )

	# Arguments for DC use case only:
	parser.add_argument('-C', '--dc_file', metavar='\b', type=str, default='/home/opc/author/RBBM/rbbm_src/muse/data/mas/tax_rules.txt',
	help='holoclean needs a input text file which contains the denial constraints, this will be the file inside dc_dir (default: %(default)s)')

	parser.add_argument('-S', '--semantic_version', metavar='\b', type=str, default='ind',
	help='muse semantic version (ind/stage/end/step) (default: %(default)s)')

	parser.add_argument('-t', '--table_name', metavar='\b', type=str, default='tax',
	help='the table name from database cr that you want to work with (default: %(default)s)')

	parser.add_argument('-F', '--desired_dcs_file', metavar='\b', type=str, default='/home/opc/author/RBBM/rbbm_src/dc_src/user_desired_dcs.txt',
	help='the ground truth DCs that so called user think is correct (default: %(default)s)')

	parser.add_argument('-I', '--user_specify_pairs', metavar='\b', type=str, default='True',
	help='user specify pairs of violations to repair? (default: %(default)s)')

	parser.add_argument('-B', '--repeatable_muse', metavar='\b', type=str, default='False',
	help='run all 3 algoritns with repeatable muse? (default: %(default)s)')
	
	parser.add_argument('-a', '--repeatable_strats', metavar='\b', type=str, default='information_gain,naive,optimal',
	help='what strats to run when its repeatable_muse? (default: %(default)s)')

	args = parser.parse_args()

	logger.critical(args)

	if(args.use_case=='dc'):
		if(args.user_specify_pairs=='True'):
			user_specify_pairs=True
		else:
			user_specify_pairs=False
		if(args.repeatable_muse=='True'):
			repeatable_muse=True
		else:
			repeatable_muse=False
			
		# conn = psycopg2.connect(dbname=args.dbname, user=args.dbuser, password=args.dbpaswd)
		input_arg_obj = dc_input(
			dc_file=args.dc_file,
			stats=StatsTracker(),
			log_level=args.log_level,
			table_name=args.table_name,
			experiment_name=args.experiment_name,
			pre_filter_thresh=args.pre_filter_thresh,
			semantic_version=args.semantic_version,
			user_input_size=args.user_input_size,
			complaint_ratio=args.complaint_ratio,
			desired_dcs_file=args.desired_dcs_file,
			strategy=args.strategy,
			deletion_factor=args.deletion_factor,
			acc_threshold=args.retrain_accuracy_thresh,
			user_specify_pairs=user_specify_pairs,
			retrain_every_percent=args.retrain_every_percent,
			repeatable_muse=repeatable_muse,
			repeatable_strats=args.repeatable_strats,
			deletion_absolute_threshold=args.deletion_absolute_threshold,
			deletion_type=args.deletion_type,
			dc_model_type=args.dc_model_type
			)
	else:
		conn = psycopg2.connect(dbname=args.dbname, user=args.user, password=args.password, port=args.port)

		if(args.run_intro=='false'):
			run_intro=False
		else:
			run_intro=True

		if(args.run_amazon=='false'):
			run_amazon=False
		else:
			run_amazon=True

		if(args.run_painter=='false'):
			run_painter=False
		else:
			run_painter=True

		if(args.run_professor=='false'):
			run_professor=False
		else:
			run_professor=True

		if(args.load_funcs_from_pickle=='true'):
			load_funcs_from_pickle=True
		else:
			load_funcs_from_pickle=False

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
		run_intro=run_intro,
		run_amazon=run_amazon,
		run_painter=run_painter,
		run_professor=run_professor,
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
		)

	if(isinstance(input_arg_obj, lf_input)):
		lf_main(input_arg_obj)
	if(isinstance(input_arg_obj, dc_input)):
		dc_main(input_arg_obj)


if __name__ == '__main__':
	main()
