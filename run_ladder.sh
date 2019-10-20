#!/usr/bin/env bash
export DATA_DIR="$(pwd)/examples/nyt/partial-gen/"
# export DATA_DIR="$(pwd)/examples/nyt/avitm/"
# export DATA_DIR="$(pwd)/examples/nyt/vampire/"
# export DATA_DIR="$(pwd)/examples/ag/"
# export VOCAB_SIZE=1995
export VOCAB_SIZE=30000
export LAZY=1   

python scripts/train.py --config training_config/ladder.jsonnet --serialization-dir model_logs/ladder  --environment LADDER --device 0 --seed 6 --override
