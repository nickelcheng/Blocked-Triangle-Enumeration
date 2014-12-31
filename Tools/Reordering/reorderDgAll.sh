#!/bin/bash

RLB_FOLDER=../../TestData/Relabel
DG_FOLDER=../../TestData/Degeneracy
MAP_FOLDER=../../TestData/Mapping
DVALUE_FOLDER=../../TestData/Degeneracy/Analysis/Dvalue

EXEC=degeneracy

g++ degeneracy.cpp -o $EXEC

files=$(ls $RLB_FOLDER)

for f in $files
do
    filename=${f%.*}
    if [ -f $DG_FOLDER/$filename.dg ]; then
        echo "$filename.dg already exists."
        continue
    fi

    echo "Reordering $f"
    ./$EXEC $RLB_FOLDER/$f $DG_FOLDER/$filename.dg $MAP_FOLDER/$filename.rlb2dg $DVALUE_FOLDER/$filename.dvalue
done
