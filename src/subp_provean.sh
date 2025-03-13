#!/bin/bash

onerr(){ while caller $((n++));do :;done;}
trap onerr ERR

GID=$1
EXPDIR=$2
NUMTHREADS=$3
DATADIR=$4
provean_path=$5
shopt -s nullglob

variants="$EXPDIR/$GID".new.var
fasta="$DATADIR/$GID".fa

if [ -f "$DATADIR/$GID".sss ]; then
    /bin/bash "$provean_path" -q "$fasta" -v "$variants" --supporting_set  "$DATADIR/$GID.sss" --num_threads "$NUMTHREADS" > "$EXPDIR/$GID.txt"
    wait
else
    /bin/bash "$provean_path" -q "$fasta" -v "$variants" --save_supporting_set  "$DATADIR/$GID.sss" --num_threads "$NUMTHREADS" > "$EXPDIR/$GID.txt"
    wait  
fi
     


#make csv
rm -f -- "$EXPDIR/$GID".csv
touch "$EXPDIR/$GID".csv

while read -r line;
do

    if [[ "${line:0:1}" != "#" && "${line:0:1}" != "[" ]]; then
        IFS=$'\t'
        read -ra VSCORE <<< "$line"
        echo "${VSCORE[0]},${VSCORE[1]}" >> "$EXPDIR"/"$GID".csv
    fi

done < "$EXPDIR/$GID".txt
  