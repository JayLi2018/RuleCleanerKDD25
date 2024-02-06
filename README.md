This is the repository for RuleCleaner submitted to VLDB2024

To run code, the following flags are available:

```python
usage: main.py [-h] [-U] [-e EXPERIMENT_NAME] [-R REPEATABLE] [-x SEED] [-X SEED_FILE] [-E RETRAIN_EVERY_PERCENT] [-A RETRAIN_ACCURACY_THRESH] [-T] [-l] [-s] [-r] [-G STRATEGY]
               [-D DELETION_FACTOR] [-W DELETION_ABSOLUTE_THRESHOLD] [-b DELETION_TYPE] [-d DBNAME] [-P PORT] [-p PASSWORD] [-u USER] [-f LF_SOURCE] [-O NUMBER_OF_FUNCS] [-i] [-z] [-w] [-o]
               [-k LOAD_FUNCS_FROM_PICKLE] [-K PICKLE_FILE_NAME] [-M TRAINING_MODEL_TYPE] [-n] [-t] [--run-gpt-rules] [--gpt-dataset] [--gpt-pickled-rules-dir]

Running experiments of LFRepair

optional arguments:
  -h, --help            show this help message and exit
  -U, --use_case    use case of the run, is it for dc or lf? (default: dc)
  -e EXPERIMENT_NAME, --experiment_name EXPERIMENT_NAME
                        the name of the experiment, the results will be stored in the directory named with experiment_name_systime (default: test_blah)
  -R REPEATABLE, --repeatable REPEATABLE
                        repeatable? (default: true)
  -x SEED, --seed SEED  if repeatable, specify a seed number here (default: 123)
  -X SEED_FILE, --seed_file SEED_FILE
                        if repeatable, specify a seed number here (default: seeds.txt)
  -E RETRAIN_EVERY_PERCENT, --retrain_every_percent RETRAIN_EVERY_PERCENT
                        retrain over every (default: 1*100), the default order is sorted by treesize ascendingly
  -A RETRAIN_ACCURACY_THRESH, --retrain_accuracy_thresh RETRAIN_ACCURACY_THRESH
                        when retrain over every retrain_every_percent, the algorithm stops when the fix rate is over this threshold (default: 1)
  -T, --pre_filter_thresh 
                        prefilter those rules that have number of tuples involved in violations above this thresh (default: 0)
  -l, --log_level   loglevel: debug/info/warning/error/critical (default: debug)
  -s, --user_input_size 
                        user input size total (the complaint size is decided by user_input_size*complaint_ratio) and confirm size is decided by user_input_size-complaint_size(default: 20)
  -r, --complaint_ratio 
                        user input complaint ratio (default: 0.5)
  -G STRATEGY, --strategy STRATEGY
                        method used to repair the rules (naive, information_gain, optimal) (default: information_gain)
  -D DELETION_FACTOR, --deletion_factor DELETION_FACTOR
                        this is a factor controlling how aggressive the algorithm chooses to delete the rule from the rulset (default: 0.5)
  -W DELETION_ABSOLUTE_THRESHOLD, --deletion_absolute_threshold DELETION_ABSOLUTE_THRESHOLD
                        this is threshold for absolute tree size increase (default: 10)
  -b DELETION_TYPE, --deletion_type DELETION_TYPE
                        deletion type (ratio/absolute) (default: ratio)
  -d DBNAME, --dbname DBNAME
                        database name which stores the dataset, (default: label)
  -P PORT, --port PORT  database port, (default: 5433)
  -p PASSWORD, --password PASSWORD
                        database password, (default: 5432)
  -u USER, --user USER  database user, (default: postgres)
  -f LF_SOURCE, --lf_source LF_SOURCE
                        the source of labelling function (intro / system generate) (default: undefined)
  -O NUMBER_OF_FUNCS, --number_of_funcs NUMBER_OF_FUNCS
                        if if_source is selected as system generate, how many do you want(default: 20)
  -i, --run_intro
  -z, --run_amazon
  -w, --run_painter
  -o, --run_professor
  -k LOAD_FUNCS_FROM_PICKLE, --load_funcs_from_pickle LOAD_FUNCS_FROM_PICKLE
                        (flag indicating if we want to load functions from a pickle file default: false)
  -K PICKLE_FILE_NAME, --pickle_file_name PICKLE_FILE_NAME
                        (if load_funcs_from_pickle, then heres the pickle file name : placeholder_name)
  -M TRAINING_MODEL_TYPE, --training_model_type TRAINING_MODEL_TYPE
                        the model used to get the label: majority/snorkel (default: snorkel)
  -n, --dataset_name 
                        dataset used in the use case of labelling functions (default: youtube)
  -t, --table_name  the table name from database cr that you want to work with (default: tax)
  --run-gpt-rules
  --gpt-dataset       youtube/amazon/pt/pa
  --gpt-pickled-rules-dir 
```

The full version of the paper is [here](full_version.pdf)
