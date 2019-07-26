#!/usr/bin/env bash
export DATA_DIR="$(pwd)/examples/nyt/"
export VOCAB_SIZE=30000
export LAZY=1   

python -m pdb scripts/train.py --config training_config/vampire.jsonnet --serialization-dir model_logs/vampire  --environment VAMPIRE --device -1 --seed 6

