#!/usr/bin/env bash
# export DATA_DIR="$(pwd)/examples/nyt/partial-gen"
# export DATA_DIR="$(pwd)/examples/nyt/avitm"
# export DATA_DIR="$(pwd)/examples/nyt/vampire"
export DATA_DIR="$(pwd)/examples/toy/bammanK50P100"
# export VOCAB_SIZE=1995
export VOCAB_SIZE=3000
export LAZY=1   

# python scripts/train.py --config training_config/avitm.jsonnet --serialization-dir model_logs/avitm_default  --environment AVITM --device 0 --seed 6 --override
# python scripts/train.py --config training_config/partial-gen.jsonnet --serialization-dir model_logs/partial-gen_default  --environment PARTIALGEN --device 0 --seed 6 --override
for metric in e_npmi
# -vi  purity
do

    case $metric in e_npmi|d_npmi)
        export METRIC="+$metric"    
        ;;    
        
        loss)
        export METRIC="-$metric"
        ;;
        
    esac

    for doc_kl in 1000 # 4000 6000
    do
        export DOC_LINEAR_SCALING=$doc_kl
        for k in 25 50 100
        do
            for p in 25  50 100 
            do
                export K=$k
                export P=$p
                python -m scripts.train --config training_config/bamman-toy.jsonnet --serialization-dir model_logs/"$metric"-bamman-toy-K"$K"P"$P"-div"$doc_kl"  --environment BAMMAN --device 0 --seed 6 --override
            done
        done
    done
done
# python -m scripts.train --config training_config/basic-l.jsonnet --serialization-dir model_logs/basic-ladder-movies-K100P100  --environment LEO  --device 0 --seed 6 --override
