#!/bin/bash
for k in 25 50 100
do
    for p in 25 50 100 
    do
        export K=$k
        export P=$p
        scp -r zeyuliu2@attu.cs.washington.edu:~/d_npmi-basic-ladder-movies-K"$K"P"$P"-namefree.tgz model_logs 
    done
done
