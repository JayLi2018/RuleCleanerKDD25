"""Use DataSculpt to generate labelling functions"""
import sys
# sys.path.append('/Users/chenjieli/Desktop/')
# sys.path.append('/nfs/users/chenjie/')
sys.path.append('/home/opc/chenjie/')

from LLMDP.main import main
import argparse


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# dataset
	parser.add_argument("--dataset-path", type=str, default="/home/opc/chenjie/LLMDP/data/wrench_data", help="dataset path")
	parser.add_argument("--dataset-name", type=str, default="youtube", help="dataset name")
	parser.add_argument("--feature-extractor", type=str, default="bert", help="feature for training end model")
	parser.add_argument("--stop-words", type=str, default=None)
	parser.add_argument("--stemming", type=str, default="porter")
	parser.add_argument("--append-cdr", action="store_true", help="append cdr snippets to original dataset")
	# sampler
	parser.add_argument("--sampler", type=str, default="uniform", choices=["passive", "uncertain", "QBC", "SEU", "weighted", "uniform"],
	                    help="sample selector")
	parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L12-v2")
	parser.add_argument("--distance", type=str, default="cosine")
	parser.add_argument("--uncertain-metric", type=str, default="entropy")
	parser.add_argument("--neighbor-num", type=int, default=10, help="neighbor count in KNN")
	parser.add_argument("--alpha", type=float, default=1.0, help="trade-off factor for uncertainty. ")
	parser.add_argument("--beta", type=float, default=1.0, help="trade-off factor for class balance. ")
	parser.add_argument("--gamma", type=float, default=1.0, help="trade-off factor for distance to labeled set.")
	# data programming
	parser.add_argument("--label-model", type=str, default="Snorkel", choices=["Snorkel", "MeTaL", "MV"], help="label model used in DP paradigm")
	parser.add_argument("--use-soft-labels", action="store_true", help="set to true if use soft labels when training end model")
	parser.add_argument("--end-model", type=str, default="logistic", choices=["logistic", "mlp"], help="end model in DP paradigm")
	parser.add_argument("--default-class", type=int, default=None)
	parser.add_argument("--tune-label-model", type=bool, default=True, help="tune label model hyperparameters")
	parser.add_argument("--tune-end-model", type=bool, default=True, help="tune end model hyperparameters")
	parser.add_argument("--tune-metric", type=str, default="acc", help="evaluation metric used to tune model hyperparameters")
	# label function
	parser.add_argument("--lf-agent", type=str, default="chatgpt", choices=["chatgpt", "llama-2", "wrench"], help="agent that return candidate LFs")
	parser.add_argument("--lf-type", type=str, default="keyword", choices=["keyword", "regex"], help="LF family")
	parser.add_argument("--lf-filter", type=str, nargs="+", default=["acc", "overlap"], help="filters for LF verification")
	parser.add_argument("--lf-acc-threshold", type=float, default=0.6, help="LF accuracy threshold for verification")
	parser.add_argument("--lf-overlap-threshold", type=float, default=0.95, help="LF overlap threshold for verification")
	parser.add_argument("--max-lf-per-iter", type=int, default=100, help="Maximum LF num per interaction")
	parser.add_argument("--max-ngram", type=int, default=3, help="N-gram in keyword LF")
	# prompting method
	parser.add_argument("--lf-llm-model", type=str, default="gpt-3.5-turbo-0613")
	parser.add_argument("--example-per-class", type=int, default=1)
	parser.add_argument("--sample-instance-per-class", type=int, default=1) 
		# added for uniform sampling in prompt selection
	parser.add_argument("--return-explanation", action="store_true")
	parser.add_argument("--example-selection", type=str, default="random", choices=["random", "neighbor"])
	parser.add_argument("--temperature", type=float, default=0.7)
	parser.add_argument("--top-p", type=float, default=1)
	parser.add_argument("--n-completion", type=int, default=1)
	# experiment
	parser.add_argument("--num-query", type=int, default=50, help="total selected samples")
	parser.add_argument("--train-iter", type=int, default=10, help="evaluation interval")
	parser.add_argument("--sleep-time", type=float, default=0, help="sleep time in seconds before each query")
	parser.add_argument("--early-stop", action="store_true")
	parser.add_argument("--runs", type=int, default=5)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--display", action="store_true")
	parser.add_argument("--save-wandb", action="store_true")
	parser.add_argument("--trails-num", type=int, default=100)
	parser.add_argument("--log-name", type=str, default='log_general.txt')
	# rc arg 
	parser.add_argument("--use-rc-flavor", action="store_true")
	parser.add_argument("--rc-pickle-file-loc", type=str, default="/home/opc/chenjie/RBBM/chatgpt_rbbm/")
	args = parser.parse_args()
	print(args)
	# exit()
	main(args)