#!/bin/bash
trap "kill 0" EXIT 

python main.py -p True -U lf -R True -Z 3 -N 1000 -d snorkel_dataset -u postgres -P 123 -W 2 -A 4 -L True -E single_func -D youtube
