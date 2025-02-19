#!/bin/bash

# File to be parsed
FILE="matmul_problems.csv"

IFS=','

i=0

# Read the file line by line
while read -r M N K Layout 
do
  # Process the data (skip the header line)
  if [ "$M" != "M" ]; then
    #ncu -k regex:nv --section=SpeedOfLight python run_nvjet_matmul.py $M $N $K $Layout
    KERNEL=$(ncu -k regex:nvjet --metrics dram__cycles_elapsed.max python run_nvjet_matmul.py $M $N $K $Layout | grep " nvjet")
    echo $i"," $M $N $K $Layout"," $KERNEL
    i=$((i + 1))
  fi
done < "$FILE"
