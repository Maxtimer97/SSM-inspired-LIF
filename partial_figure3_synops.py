# Copyright (c) 2025 Maxime Fabre and Lyubov Dudchenko
# This file is part of SSM-inspired-LIF, released under the MIT License.

import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import torch
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from sympy import symbols, Eq, solve, sqrt, pi
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker

from pathlib import Path
import yaml
import argparse

from models.snns import SNN
from models.snns import CSiLIFLayer, SiLIFLayer, CadLIFLayer, ResonateFireLayer
from train import Experiment
from parsers.model_config import add_model_options
from parsers.training_config import add_training_options



model_configs = {'SiLIF': 'ssc_silif_4ms.yaml',
'CSiLIF': 'ssc_csilif_4ms.yaml',
'CadLIF': 'ssc_cadlif_4ms.yaml'}

model_accs = {'SiLIF': 0,
'CSiLIF': 0,
'CadLIF': 0}

model_sops = {'SiLIF': 0,
'CSiLIF': 0,
'CadLIF': 0}

def parse_args(yaml_config):
    parser = argparse.ArgumentParser(
        description="Model training on spiking speech commands datasets."
    )
    parser = add_model_options(parser)
    parser = add_training_options(parser)
    
    # Override default values from yaml_config
    for key, value in yaml_config.items():
        if hasattr(parser, 'set_defaults'):
            parser.set_defaults(**{key: value})

    args = parser.parse_args()
    return args

### Run pretrained models on multiple seeds ###

for model in model_configs:

    yaml_config = {}
    config_path = "./configs/" + model_configs[model]
    with open(config_path, "r") as f:
        sweep_config = yaml.safe_load(f)
    yaml_config = {
        k: v['values'][0] if isinstance(v, dict) and 'values' in v else v
        for k, v in sweep_config.get("parameters", {}).items()
    }

    args = parse_args(yaml_config)

    base_config = {}
    # Process arguments into parameters
    for arg, value in vars(args).items():
        if isinstance(value, list):
            base_config[arg] = value[0]
        else:
            base_config[arg] = value

    base_config['debug'] = True
    base_config["evaluate_pretrained"] = True

    model_acc = 0
    model_sop = 0

    print("Starting "+ model +" evaluation")

    for seed in [13, 42, 73, 128, 268]:
        config = base_config.copy()
        config['seed'] = seed
        experiment = Experiment(config, 0)
        test_acc, test_sop = experiment.forward()

        model_acc += test_acc/5
        model_sop += test_sop/5
    model_accs[model] = model_acc
    model_sops[model] = model_sop

 #########################################


### Prepare Data with analytic S5-RF and Event-SSM values ###

models = ["S5-RF"] + list(model_configs.keys()) + ["Event-SSM"]
accuracy = {
    "mean": np.concatenate([np.array([78.8]), np.array(list(model_accs.values())) * 100, np.array([85.3])]),
}

# Calculate SoP from sparsity if needed, but here we use provided sop_mean directly
sop_mean = np.concatenate([np.array([250*21751]), np.array(list(model_sops.values())), np.array([8000*64*64*2 + 1000*64*64*8])])

desired_order = ["S5-RF", "CadLIF", "CSiLIF", "SiLIF", "Event-SSM"]
idx_order = [models.index(m) for m in desired_order]

# Reorder arrays
models_ordered = ["S5-RF", "cAdLIF", "C-SiLIF", "SiLIF", "Event-SSM"]
acc_mean_ordered = accuracy['mean'][idx_order]
sop_mean_ordered = sop_mean[idx_order]

#########################################

#### Plotting ###

x = np.arange(len(models))

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"]
})
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
})
fig, ax1 = plt.subplots(figsize=(6.0, 4.0))  # Small size for LaTeX
ax2 = ax1.twinx()

# Accuracy with error bars + shaded band
ax1.plot(models_ordered, acc_mean_ordered, marker='o', label="Accuracy")

ax2.plot(models_ordered, sop_mean_ordered, marker='o', label="SoP", color='tab:orange')

# labels & styling
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((7, 7))  # Force scale factor at 1e7
ax1.set_xticks(models_ordered)
ax1.set_xticklabels(models_ordered, rotation=30, ha='center')
# ax1.set_xlabel("Model")
ax1.set_ylabel("Accuracy (%)")
ax2.set_ylabel("Synaptic operations")
ax2.yaxis.set_major_formatter(formatter)
ax2.ticklabel_format(axis='y', style='sci', scilimits=(7,7))

# ax1.set_title("Model Accuracy and Synaptic Operations")
fig.tight_layout(pad=0.5)
fig.savefig("./figs/figure3_partial.png", bbox_inches='tight')
plt.show()

print("Figure 3 partial plot saved as figure3_partial.png")

###########################################