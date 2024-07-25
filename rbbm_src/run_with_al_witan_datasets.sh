#!/bin/bash

# Number of repetitions
declare -i repetitions=2

# Function to run the python command with the full dataset
run_command() {
    local strategy=$1
    python main.py -e al_witan_results/imdb/ -U lf -x ${rseed} -T 0 -s 40 -r 0.5 -G information_gain -D 0 -m -n imdb -P 5432 --user-input-strat $strategy
    python main.py -e al_witan_results/painter_architect/ -U lf -x ${rseed} -T 0 -s 40 -r 0.5 -G information_gain -D 0 -w -n painter_architect -P 5432 --user-input-strat $strategy
    python main.py -e al_witan_results/professor_teacher/ -U lf -x ${rseed} -T 0 -s 40 -r 0.5 -G information_gain -D 0 -o -n professor_teacher -P 5432 --user-input-strat $strategy
    python main.py -e al_witan_results/amazon/ -U lf -x ${rseed} -T 0 -s 40 -r 0.5 -G information_gain -D 0 -z -n amazon -P 5432 --user-input-strat $strategy
    python main.py -e al_witan_results/photographer_journalist/ -U lf -x ${rseed} -T 0 -s 40 -r 0.5 -G information_gain -D 0 --run-photographer -n photographer_journalist -P 5432 --user-input-strat $strategy
    python main.py -e al_witan_results/physician/ -U lf -x ${rseed} -T 0 -s 40 -r 0.5 -G information_gain -D 0 --run-physician -n physician_professor -P 5432 --user-input-strat $strategy
    python main.py -e al_witan_results/yelp/ -U lf -x ${rseed} -T 0 -s 40 -r 0.5 -G information_gain -D 0 --run-yelp -n yelp -P 5432 --user-input-strat $strategy
    python main.py -e al_witan_results/plots/ -U lf -x ${rseed} -T 0 -s 40 -r 0.5 -G information_gain -D 0 --run-plots -n plots -P 5432 --user-input-strat $strategy
    # python main.py -e al_witan_results/fakenews/ -U lf -x ${rseed} -T 0 -s 40 -r 0.5 -G information_gain -D 0 --run-fakenews -n fakenews -P 5432 --user-input-strat $strategy
    python main.py -e al_witan_results/dbpedia/ -U lf -x ${rseed} -T 0 -s 40 -r 0.5 -G information_gain -D 0 --run-dbpedia -n dbpedia -P 5432 --user-input-strat $strategy
    python main.py -e al_witan_results/agnews/ -U lf -x ${rseed} -T 0 -s 40 -r 0.5 -G information_gain -D 0 --run-agnews -n agnews -P 5432 --user-input-strat $strategy
    python main.py -e al_witan_results/tweets/ -U lf -x ${rseed} -T 0 -s 40 -r 0.5 -G information_gain -D 0 --run-tweets -n tweets -P 5432 --user-input-strat $strategy
    python main.py -e al_witan_results/spam/ -U lf -x ${rseed} -T 0 -s 40 -r 0.5 -G information_gain -D 0 --run-spam -n spam -P 5432 --user-input-strat $strategy
}

# Function to run the python command with a percentage of the dataset
run_command_percentage() {
    local strategy=$1
    local percent=$2
    python main.py -e percentage_al_witan_results/imdb/ -U lf -x ${rseed} -T 0 -s 40 -r 0.5 -G information_gain -D 0 -m -n imdb -P 5432 --user-input-strat $strategy --user-input-sample-strat percentage --user-input-percentage $percent
    python main.py -e percentage_al_witan_results/painter_architect/ -U lf -x ${rseed} -T 0 -s 40 -r 0.5 -G information_gain -D 0 -w -n painter_architect -P 5432 --user-input-strat $strategy --user-input-sample-strat percentage --user-input-percentage $percent
    python main.py -e percentage_al_witan_results/professor_teacher/ -U lf -x ${rseed} -T 0 -s 40 -r 0.5 -G information_gain -D 0 -o -n professor_teacher -P 5432 --user-input-strat $strategy --user-input-sample-strat percentage --user-input-percentage $percent
    python main.py -e percentage_al_witan_results/amazon/ -U lf -x ${rseed} -T 0 -s 40 -r 0.5 -G information_gain -D 0 -z -n amazon -P 5432 --user-input-strat $strategy --user-input-sample-strat percentage --user-input-percentage $percent
    python main.py -e percentage_al_witan_results/photographer_journalist/ -U lf -x ${rseed} -T 0 -s 40 -r 0.5 -G information_gain -D 0 --run-photographer -n photographer_journalist -P 5432 --user-input-strat $strategy --user-input-sample-strat percentage --user-input-percentage $percent
    python main.py -e percentage_al_witan_results/physician/ -U lf -x ${rseed} -T 0 -s 40 -r 0.5 -G information_gain -D 0 --run-physician -n physician_professor -P 5432 --user-input-strat $strategy --user-input-sample-strat percentage --user-input-percentage $percent
    python main.py -e percentage_al_witan_results/yelp/ -U lf -x ${rseed} -T 0 -s 40 -r 0.5 -G information_gain -D 0 --run-yelp -n yelp -P 5432 --user-input-strat $strategy --user-input-sample-strat percentage --user-input-percentage $percent
    python main.py -e percentage_al_witan_results/plots/ -U lf -x ${rseed} -T 0 -s 40 -r 0.5 -G information_gain -D 0 --run-plots -n plots -P 5432 --user-input-strat $strategy --user-input-sample-strat percentage --user-input-percentage $percent
    # python main.py -e percentage_al_witan_results/fakenews/ -U lf -x ${rseed} -T 0 -s 40 -r 0.5 -G information_gain -D 0 --run-fakenews -n fakenews -P 5432 --user-input-strat $strategy --user-input-sample-strat percentage --user-input-percentage $percent
    python main.py -e percentage_al_witan_results/dbpedia/ -U lf -x ${rseed} -T 0 -s 40 -r 0.5 -G information_gain -D 0 --run-dbpedia -n dbpedia -P 5432 --user-input-strat $strategy --user-input-sample-strat percentage --user-input-percentage $percent
    python main.py -e percentage_al_witan_results/agnews/ -U lf -x ${rseed} -T 0 -s 40 -r 0.5 -G information_gain -D 0 --run-agnews -n agnews -P 5432 --user-input-strat $strategy --user-input-sample-strat percentage --user-input-percentage $percent
    python main.py -e percentage_al_witan_results/tweets/ -U lf -x ${rseed} -T 0 -s 40 -r 0.5 -G information_gain -D 0 --run-tweets -n tweets -P 5432 --user-input-strat $strategy --user-input-sample-strat percentage --user-input-percentage $percent
    python main.py -e percentage_al_witan_results/spam/ -U lf -x ${rseed} -T 0 -s 40 -r 0.5 -G information_gain -D 0 --run-spam -n spam -P 5432 --user-input-strat $strategy --user-input-sample-strat percentage --user-input-percentage $percent
}

# Main loop
declare -i x=1
while [ $x -le $repetitions ]; do
    rseed=$(shuf -i 1-2000 -n 1)
    echo "rseed is $rseed"

    # Run commands with active learning strategy
    run_command "active_learning"
    # # Run commands with naive strategy
    run_command "naive"

    # Run commands with percentage of data and active learning strategy
    run_command_percentage "active_learning" 0.01

    x=$(( x+1 ))
done
