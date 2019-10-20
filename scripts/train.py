import argparse
import json
import os
import random
import shutil
import subprocess
import tempfile
from typing import Any, Dict
import pdb
from allennlp.common.params import Params

from environments import ENVIRONMENTS
from environments.random_search import HyperparameterSearch

random_int = random.randint(0, 2**32)

def main():
    parser = argparse.ArgumentParser()  # pylint: disable=invalid-name
    parser.add_argument('-o',
                        '--override',
                        action="store_true",
                        help='remove the specified serialization dir before training')
    parser.add_argument('-c', '--config', type=str, help='training config', required=True)
    parser.add_argument('-s', '--serialization-dir', type=str, help='model serialization directory', required=True)
    parser.add_argument('-e', '--environment', type=str, help='hyperparameter environment', required=True)
    parser.add_argument('-r', '--recover', action='store_true', help = "recover saved model")
    parser.add_argument('-d', '--device', type=str, required=False, help = "device to run model on")
    parser.add_argument('-x', '--seed', type=str, required=False, help = "seed to run on")
    

    args = parser.parse_args()

    env = ENVIRONMENTS[args.environment.upper()]

    
    space = HyperparameterSearch(**env)

    sample = space.sample()
    
    for key, val in sample.items():
        os.environ[key] = str(val)
    
    if args.device:
        os.environ['CUDA_DEVICE'] = args.device

    if args.seed:
        os.environ['SEED'] = args.seed

    data_dir = os.environ["DATA_DIR"]
    fail_message = ""
    # priors = {0: '{"type": "normal", "mu": 0, "var": 1}', 1: '{"type": "laplace-approx", "alpha": 1}'}
    num_repeat = 1 

    # for use_doc_info in [0, 1]:
    #     env["USE_DOC_INFO"] = use_doc_info
    #     if use_doc_info == 1:
    #         os.environ["DATA_DIR"] = data_dir + "-global"
    #     else:
    #         os.environ["DATA_DIR"] = data_dir
    for i in range(num_repeat):
        serialization_dir = f"{args.serialization_dir}." \
            f"DocInfo{env['USE_DOC_INFO']}.lr{env['LEARNING_RATE']}." \
            f"BNonNormal{env['APPLY_BATCHNORM_ON_NORMAL']}.BNonDecoder{env['APPLY_BATCHNORM_ON_DECODER']}/" \
            f"NoRepeat{i}"

        if args.seed:
            os.environ['SEED'] = str(int(args.seed) + i)

        allennlp_command = [
            "allennlp",
            "train",
            "--include-package",
            "neural_persona",
            args.config,
            "-s",
            serialization_dir
            ]

        if args.seed:
            allennlp_command[-1] = allennlp_command[-1] + "_" + args.seed

        if not os.path.exists(allennlp_command[-1]):
            os.makedirs(allennlp_command[-1])

        if args.recover:
            allennlp_command.append("--recover")

        if os.path.exists(allennlp_command[-1]) and args.override:
            print(f"overriding {allennlp_command[-1]}")
            shutil.rmtree(allennlp_command[-1])
        try:
            subprocess.run(" ".join(allennlp_command), shell=True, check=True)
        except:
            fail_message += serialization_dir + "\n"
            continue

    # open("failed_settings.txt", "w+").write(fail_message)


if __name__ == '__main__':
    main()
