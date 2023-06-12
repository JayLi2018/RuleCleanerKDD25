#!/bin/bash

python LFRepair.py -U 10 -t 0.5 -e intro_example_40 -r information_gain

python LFRepair.py -U 10 -t 0.5 -e intro_example_40 -r naive


# -h, --help            show this help message and exit
# -d, --dbname      database name which stores the dataset, (default: label)
# -P, --port        database port, (default: 5432)
# -p, --password    database password, (default: 5432)
# -u, --user        database user, (default: postgres)
# -r, --repair_method 
#                       method used to repair the rules (naive, information_gain, optimal) (default: information_gain)
# -U, --userinput_size 
#                       user input size (default: 40)
# -t, --complaint_ratio 
#                       out of the user input, what percentage of it is complaint? (the rest are confirmations) (default: 0.5)
# -f, --lf_source   the source of labelling function (intro / system generate) (default: intro)
# -n, --number_of_funcs 
#                       if if_source is selected as system generate, how many do you want(default: 20)
# -e, --experiment_name 
#                       the name of the experiment, the results will be stored in the directory named with experiment_name_systime (default: test_blah)
# -R, --repeatable  repeatable? (default: True)
# -s, --seed        if repeatable, specify a seed number here (default: 123)

