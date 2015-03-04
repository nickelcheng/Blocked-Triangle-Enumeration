#!/bin/bash

if [ $# != 6 ]; then
    echo "usage: exp.sh <method> <dense/sparse> <dstart> <dend> <nstart> <nend>"
    exit
fi

if [ -f ./$1 ]; then
    echo "$1 running... exit"
    exit
fi

if [ -f ./$1.result ]; then
    echo "$1.result exist, please remove it first"
    exit
fi

algo=$1
dstart=$3
dend=$3
nstart=$5
nend=$6

if [ $2 != "dense" -a $2 != "sparse" ]; then
    echo dense or sparse?
    exit
fi

if [ $dstart -lt 1 -o $dend -gt 9 -o $nstart -lt 1 -o $nend -gt 10 ]; then
    echo "value error"
    exit
fi

make $algo

echo $algo >> $algo.result
for((d=$dstart;d<=$dend;d++))
do
    for((n=$nstart;n<=$nend;n++))
    do
        if [ $2 == "dense" ]; then
            myd=$d"0"
            myn=$n"000"
        else
            myd="0."$d
            myn=$n"0000"
        fi

        echo -n d=$myd%,N=$myn>> $algo.result

#        for((t=0;t<5;t++))
#        do
            echo "./$algo ../Experiments/d$myd/N$myn $myn > /dev/null 2>> $algo.result"
            ./$algo ../Experiments/d$myd/N$myn $myn > /dev/null 2>> $algo.result
            echo "" >> $algo.result
#        done
    done
done

rm $1
