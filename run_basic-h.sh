#!/usr/bin/env bash
# export DATA_DIR="$(pwd)/examples/nyt/partial-gen"
# export DATA_DIR="$(pwd)/examples/nyt/avitm"
# export DATA_DIR="$(pwd)/examples/nyt/vampire"
export DATA_DIR="$(pwd)/examples/nyt/entity_based"
# export VOCAB_SIZE=1995
export VOCAB_SIZE=3000
export LAZY=1   

# python scripts/train.py --config training_config/avitm.jsonnet --serialization-dir model_logs/avitm_default  --environment AVITM --device 0 --seed 6 --override
# python scripts/train.py --config training_config/partial-gen.jsonnet --serialization-dir model_logs/partial-gen_default  --environment PARTIALGEN --device 0 --seed 6 --override
python -m scripts.train --config training_config/basic-h.jsonnet --serialization-dir model_logs/basic-hierarchical-gumbel  --environment LEO  --device 0 --seed 6 --override
