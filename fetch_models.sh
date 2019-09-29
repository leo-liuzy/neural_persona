#!/bin/bash
for k in 25 50 100
do
    for p in 25 50 100 
    do
        export K=$k
        export P=$p
        scp -r zeyuliu2@foch.cs.washington.edu:~/neural_persona/model_logs/d_npmi-basic-ladder-movies-K"$K"P"$P"-namefree model_logs 
    done
done
