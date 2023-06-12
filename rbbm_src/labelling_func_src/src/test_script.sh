#!/bin/bash

python LFRepair.py -U 20 -t 0.5 -e test_609 -r information_gain -f intro -D 0
python LFRepair.py -U 20 -t 0.5 -e test_609 -r brute_force -f intro -D 0
python LFRepair.py -U 20 -t 0.5 -e test_609 -r naive -f intro -D 0
python LFRepair.py -U 20 -t 0.5 -e test_609 -r information_gain -f intro -D 0.5
python LFRepair.py -U 20 -t 0.5 -e test_609 -r brute_force -f intro -D 0.5
python LFRepair.py -U 20 -t 0.5 -e test_609 -r naive -f intro -D 0.5