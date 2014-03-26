#!/bin/bash

rows=4400
cols=4400
repeats=5

current_dir=`pwd`

export MAD_NUM_THREADS=4

for i in `seq 16 400`
do
    for j in `seq 16 400`
    do 
        if [[ $(($rows % $i)) -eq 0 ]] && \
           [[ $(($cols % $j)) -eq 0 ]]; 
           then
               echo "Doing i = $i and j = $j"
               echo "$i $j" > $current_dir/output_asymm"$i"_"$j".txt
               $current_dir/ta_dense $rows $i $cols $j $repeats >> $current_dir/output_asymm"$i"_"$j".txt
        fi
    done
done
