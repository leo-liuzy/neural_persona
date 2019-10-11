#!/bin/bash
for k in 25 50 100
do
    for p in 25 50 100 
    do
        export K=$k
        export P=$p
# e_npmi-basic-ladder-movies-K25P100-namefree-div2000.tgz
        scp -r zeyuliu2@foch.cs.washington.edu:~/neural_persona/model_logs/e_npmi-basic-ladder-movies-K"$K"P"$P"-namefree-div4000.tgz model_logs 
    done
done
