#!/bin/bash

python3 --version
day=20241022
ncores=18

python3 preprocess_data.py 2014 201401 $day $ncores
sleep 5
python3 create_data_cube_file.py 2014 201401 $day $ncores
sleep 5
python3 post_process_data_cube.py 2014 201401 $day y
sleep 5
python3 compute_periodograms.py 2014 201401 $day $ncores
