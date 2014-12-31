#!/bin/bash

ORI_FOLDER=../TestData/SNAP
RLB_FOLDER=../TestData/Relabel
MAP_FOLDER=../TestData/Mapping

EXEC=relabel

g++ relabel.cpp -o $EXEC

files=$(ls $ORI_FOLDER/format1)

for f in $files
do
    filename=${f##*/}
    filename=${filename%.*}
    #echo "./$EXEC $ORI_FOLDER/format1/$f $RLB_FOLDER/$filename.rlb $MAP_FOLDER/$filename.ori2rlb 1"
    ./$EXEC $ORI_FOLDER/format1/$f $RLB_FOLDER/$filename.rlb $MAP_FOLDER/$filename.ori2rlb 1
done

files=$(ls $ORI_FOLDER/format2)

for f in $files
do
    filename=${f##*/}
    filename=${filename%.*}
    #echo "./$EXEC $ORI_FOLDER/format2/$f $RLB_FOLDER/$filename.rlb $MAP_FOLDER/$filename.ori2rlb 2"
    ./$EXEC $ORI_FOLDER/format2/$f $RLB_FOLDER/$filename.rlb $MAP_FOLDER/$filename.ori2rlb 2
done

rm -rf $EXEC
