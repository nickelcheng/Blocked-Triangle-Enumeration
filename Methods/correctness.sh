#!/bin/bash

#make clean
#make

file=(in N1000E50254 N100E494 N800E50064 N900E50202)
size=(6 1000 100 800 900)
npt=(128 1024 128 800 928)

for ((i=0;i<5;i++))
do
    echo ${file[$i]}
    echo forward
    ./list forward SampleTest/${file[$i]} ${size[$i]} 2> /dev/null
    echo edge iterator
    ./list edge SampleTest/${file[$i]} ${size[$i]} 2> /dev/null
    echo tiled
    ./mat SampleTest/${file[$i]} ${size[$i]} ${npt[$i]} 2> /dev/null
    echo g_forward
    ./g_list forward SampleTest/${file[$i]} ${size[$i]} 256 ${size[$i]} 2> /dev/null
    echo g_edge
    ./g_list edge SampleTest/${file[$i]} ${size[$i]} 256 ${size[$i]} 2> /dev/null
    echo g_mat
    ./g_mat SampleTest/${file[$i]} ${size[$i]} 32 256 32 2> /dev/null
    echo ===
done
