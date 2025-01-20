#!/bin/bash

python3 --version
day=20241022
# folder_path="/mnt/10Tb/GONG_data/train_data"

# Use find to list subdirectories and store them in an array
# mapfile -t days < <(find "$folder_path" -mindepth 1 -maxdepth 1 -type d -exec basename {} \;)

# Sort the array
# IFS=$'\n' days=($(sort <<<"${days[*]}"))

#echo "${days[*]:7}"
# for year_month_day in "${days[@]:7}"; 
# do
     # Extract the year (first 4 characters) from the date
#     year=${year_month_day:0:4}
     # Extract the year-month (first 6 characters) from the date
#     year_month=${year_month_day:0:6}

python3 preprocess_data.py 2014 201401 $day 18
sleep 5
python3 create_data_cube_file.py 2014 201401 $day 18
sleep 5
python3 post_process_data_cube.py 2014 201401 $day y
sleep 5
python3 compute_periodograms.py 2014 201401 $day 18
#done

#start_date=20140101
#end_date=20140131

#current_date=$(date -d $start_date +%Y%m%d)

#while [[ "$current_date" -le "$end_date" ]]; do
#  echo "$current_date"
  # Insert the operations you want to perform with each date here0
#  python3 create_animation.py 2014 201401 $current_date
  # Increment the date by one day
#  current_date=$(date -d "$current_date + 1 day" +%Y%m%d)
#done
