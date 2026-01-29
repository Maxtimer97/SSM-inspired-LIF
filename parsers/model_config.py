#
# SPDX-FileCopyrightText: Copyright © 2022 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Alexandre Bittar <abittar@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This file is part of the sparch package
#
"""
This is where the parser for the model configuration is defined.
"""
import logging
from distutils.util import strtobool

logger = logging.getLogger(__name__)
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def add_model_options(parser):
    parser.add_argument(
        "--model_type",
        type=str,
        default="SiLIF",
        help="Type of ANN or SNN model.",
    )
    parser.add_argument(
        "--dt_min",
        type=float,
        default=[0.01],
        nargs='+',
        help="Min dt initialization for C-SiLIF",
    )
    parser.add_argument(
        "--dt_max",
        type=float,
        default=[0.5],
        nargs='+',
        help="Max dt initializationfor C-SiLIF ",
    )
    parser.add_argument(
        "--nb_layers",
        nargs='+',
        type=int,
        default=[3],
        help="Number of layers (including readout layer).",
    )
    parser.add_argument(
        "--nb_hiddens",
        nargs='+',
        type=int,
        default=[512],
        help="Number of neurons in all hidden layers.",
    )
    parser.add_argument(
        "--state_size",
        nargs='+',
        type=int,
        default=[64],
        help="State size N for SSM models.",
    )
    parser.add_argument(
        "--pdrop",
        nargs='+',
        type=float,
        default=[0.1],
        help="Dropout rate, must be between 0 and 1.",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        nargs='+',
        default=["batchnorm"],
        help="Type of normalization, Every string different from batchnorm "
        "and layernorm will result in no normalization.",
    )
    parser.add_argument(
        "--use_bias",
        type=lambda x: bool(strtobool(str(x))),
        default=False,
        help="Whether to include trainable bias with feedforward weights.",
    )
    parser.add_argument(
        "--bidirectional",
        type=lambda x: bool(strtobool(str(x))),
        default=False,
        help="If True, a bidirectional model that scans the sequence in both "
        "directions is used, which doubles the size of feedforward matrices. ",
    )
    return parser


def print_model_options(config):
    logging.info(
        """
        Model Config
        ------------
        Model Type: {model_type}
        Number of layers: {nb_layers}
        Number of hidden neurons: {nb_hiddens}
        Dropout rate: {pdrop}
        Normalization: {normalization}
        Use bias: {use_bias}
        Bidirectional: {bidirectional}
    """.format(
            **config
        )
    )
