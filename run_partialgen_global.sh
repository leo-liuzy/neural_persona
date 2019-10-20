#!/usr/bin/env bash
export DATA_DIR="$(pwd)/examples/nyt/partial-gen-global"
# export DATA_DIR="$(pwd)/examples/nyt/avitm"
# export DATA_DIR="$(pwd)/examples/ag/"
# export VOCAB_SIZE=1995
export VOCAB_SIZE=30000
export LAZY=1   

# python scripts/train.py --config training_config/avitm.jsonnet --serialization-dir model_logs/avitm_default  --environment AVITM --device 0 --seed 6 --override
python scripts/train.py --config training_config/partial-gen.jsonnet --serialization-dir model_logs/partial-gen --environment PARTIALGEN_GLOB --device 0 --seed 6 --override
# python -m scripts.train --config training_config/vampire.jsonnet --serialization-dir model_logs/vampire  --environment VAMPIRE  --device 0 --override
