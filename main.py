# Copyright (c) 2025 Maxime Fabre and Lyubov Dudchenko
# This file is part of SSM-inspired-LIF, released under the MIT License.
#
# Modified from: https://github.com/idiap/sparch
# Original license: BSD 3-Clause (see third_party/sparch/LICENSE)
###
### SPDX-FileCopyrightText: Copyright © 2022 Idiap Research Institute <contact@idiap.ch>
###
### SPDX-FileContributor: Alexandre Bittar <abittar@idiap.ch>
###
### SPDX-License-Identifier: BSD-3-Clause
###
### This file is part of the sparch package
###
"""
This is the script used to run experiments.
"""
import argparse
import logging

from train import Experiment
from parsers.model_config import add_model_options
from parsers.training_config import add_training_options
import torch

import wandb

logger = logging.getLogger(__name__)

import yaml
from pathlib import Path
import sys
import os

# Load YAML config if passed
yaml_config_path = None
for i, arg in enumerate(sys.argv):
    if arg.startswith("--config="):
        yaml_config_path = arg.split("=", 1)[1]
        sys.argv.pop(i)  # Remove from sys.argv so argparse ignores it
        break

config_defaults = {}
if yaml_config_path:
    base_dir = Path(__file__).resolve().parent  # Directory containing the current script
    config_path = (base_dir / "configs" / yaml_config_path).resolve()
    with open(config_path, "r") as f:
        sweep_config = yaml.safe_load(f)
    config_defaults = {
        k: v['values'][0] if isinstance(v, dict) and 'values' in v else v
        for k, v in sweep_config.get("parameters", {}).items()
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Model training on spiking speech commands datasets."
    )
    parser = add_model_options(parser)
    parser = add_training_options(parser)
    
    # Override default values from config_defaults
    for key, value in config_defaults.items():
        if hasattr(parser, 'set_defaults'):
            parser.set_defaults(**{key: value})

    args = parser.parse_args()
    return args



args = parse_args()

DEBUG = args.debug

if args.gpu_device:
    DEVICE = args.gpu_device
    delattr(args, 'gpu_device')
else:
    DEVICE = 0

def main(debug_config=None):
    """
    Runs model training/testing using the configuration specified
    by the parser arguments. Run `python run_exp.py -h` for details.
    """

    if DEBUG:
        config = debug_config
    else:
        wandb.init()
        config = wandb.config

    # Instantiate class for the desired experiment
    experiment = Experiment(config, DEVICE)

    # Run experiment
    experiment.forward()


if __name__ == "__main__":

    # Get experiment configuration from parser

    if args.sweep_id and not DEBUG:
        wandb.agent("maximes_crew/" + args.sweep_id, function=main)
    else:
        sweep_config = {
            'method': args.method,  # Method (grid or bayes) is set separately here
            'metric': {
                'name': "valid acc",   
                'goal': 'maximize'
            },
            'parameters': {},
        }

        if args.sweep_name:
            sweep_config['name'] = args.sweep_name
        delattr(args, 'sweep_name')

        debug_config = {'seed': 42}

        # Handle lif_feature separately
        if args.lif_feature:
            for feat in args.lif_feature:
                sweep_config['parameters'][feat] = {'values': [True]}
                debug_config[feat] = True

        delattr(args, 'lif_feature')

        # Process arguments into parameters based on method
        for arg, value in vars(args).items():
            # Skip the 'method' argument, as it's handled separately
            if arg == "method":
                continue

            if args.method == "grid":
                if isinstance(value, list):
                    sweep_config['parameters'][arg] = {'values': value}
                    debug_config[arg] = value[0]
                else:
                    sweep_config['parameters'][arg] = {'values': [value]}
                    debug_config[arg] = value

            elif args.method == "bayes":
                if isinstance(value, list) and len(value) == 2 and not isinstance(value[0], bool):  
                    sweep_config['parameters'][arg] = {
                        'min': value[0],
                        'max': value[1]
                    }
                    debug_config[arg] = value[0]  

                elif isinstance(value, list):  
                    sweep_config['parameters'][arg] = {'values': value}
                    debug_config[arg] = value[0]
                else:
                    sweep_config['parameters'][arg] = {'values': [value]}
                    debug_config[arg] = value


        if DEBUG:
            main(debug_config)
        else:
            if args.dataset_name == "shd":
                project_name="S3_SHD_runs"
            elif args.dataset_name == "ssc":
                project_name="S3_SSC_runs"
            elif args.dataset_name == "sc":
                project_name="S3_SC_runs"
            elif args.dataset_name == "hd":
                project_name="S3_HD_runs"


            sweep_id = wandb.sweep(sweep_config,
                                entity="maximes_crew", 
                                project=project_name) 

            print('SWEEP ID: '+sweep_id)
            # Run the sweep
            if not args.only_createsweep:
                wandb.agent(sweep_id, function=main)
