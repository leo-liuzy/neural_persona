#!/bin/bash
for k in 25 50 100
do
    for p in 25 50 100 
    do
        export K=$k
        export P=$p
# e_npmi-basic-ladder-movies-K25P100-namefree-div2000.tgz
# scp -r zeyuliu2@foch.cs.washington.edu:~/neural_persona/model_logs/loss-basic-ladder-toy-K"$K"P"$P"-div2000.tgz model_logs 
        scp -r zeyuliu2@foch.cs.washington.edu:/data/zeyuliu2/neural_persona/model_logs/e_npmi-bamman-movies-K"$K"P"$P"-namefree-div1000.tgz model_logs 
    done
done
