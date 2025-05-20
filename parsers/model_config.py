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
        # choices=["LIF", "CSiLIF", "SiLIF", "LIFfeature", "adLIFnoClamp", "LIFfeatureDim", "adLIF", "CadLIF", "CadLIFAblation", "RingInitLIFcomplex", "RAFAblation", "BRF", "ResonateFire", "RSEadLIF", "RLIF", "RadLIF", "MLP", "RNN", "LiGRU", "GRU","LIFcomplexBroad", "LIFcomplex", "LIFrealcomplex", "ReLULIFcomplex", "RLIFcomplex","RLIFcomplex1MinAlpha", "adLIFclamp", "RLIFcomplex1MinAlphaNoB","LIFcomplex_gatedB", "LIFcomplex_gatedDt", "LIFcomplexDiscr", "DelaySiLIF"],
        default="LIF",
        help="Type of ANN or SNN model.",
    )
    parser.add_argument(
        "--input_layer_type",
        type=str,
        choices=["LIF", "LIFfeature", "adLIFnoClamp", "LIFfeatureDim", "adLIF", "CadLIF", "CadLIFAblation", "RingInitLIFcomplex", "RAFAblation", "BRF", "ResonateFire", "RSEadLIF", "RLIF", "RadLIF", "MLP", "RNN", "LiGRU", "GRU","LIFcomplexBroad", "LIFcomplex", "LIFrealcomplex", "ReLULIFcomplex", "RLIFcomplex","RLIFcomplex1MinAlpha", "adLIFclamp", "RLIFcomplex1MinAlphaNoB","LIFcomplex_gatedB", "LIFcomplex_gatedDt", "LIFcomplexDiscr"],
        default="LIF",
        help="Type of ANN or SNN model.",
    )
    parser.add_argument(
        "--use_input_layer",
        nargs='+',
        type=str2bool,
        default=[False],
        help="Use half reset for LIFcomplex and RLIFcomplex models. True by default",
    )
    parser.add_argument(
        "--lif_feature",
        type=str,
        choices=["logAlpha", "cont", "1-200_1-5", "A0_5", "dtParam", "A0_5Const", "dtLog", "Dt1ms", "Dt1", "alphaConst", "imag", "NoClamp", "B", "dim2"],
        default=None,
        nargs='+',
        help="Feature of LIF",
    )
    parser.add_argument(
        "--half_reset",
        nargs='+',
        type=str2bool,
        default=[True],
        help="Use half reset for LIFcomplex and RLIFcomplex models. True by default",
    )
    parser.add_argument(
        "--gating",
        type=str,
        choices=["mamba", "mamba_dtscalar", "gla", "mLSTM", "hgrn", "rwkv"],
        default=None,
        nargs='+',
        help="Gating Mode",
    )

    parser.add_argument(
        "--recurrent",
        nargs='+',
        type=str2bool,
        default=[False],
        help="Use a recurrent version of the model.",
    )
    parser.add_argument(
        "--shared_alpha",
        nargs='+',
        type=str2bool,
        default=[False],
        help="Use a recurrent version of the model.",
    )
    parser.add_argument(
        "--shared_a",
        nargs='+',
        type=str2bool,
        default=[False],
        help="Use a recurrent version of the model.",
    )
    parser.add_argument(
        "--inp_b",
        nargs='+',
        type=str2bool,
        default=[False],
        help="Use a recurrent version of the model.",
    )
    parser.add_argument(
        "--LRU_norm",
        nargs='+',
        type=str2bool,
        default=[False],
        help="Use a recurrent version of the model.",
    )
    parser.add_argument(
        "--LRU_b",
        nargs='+',
        type=str2bool,
        default=[False],
        help="Use a recurrent version of the model.",
    )
    parser.add_argument(
        "--LRU_re_init",
        nargs='+',
        type=str2bool,
        default=[False],
        help="Use a recurrent version of the model.",
    )
    parser.add_argument(
        "--LRU_img_init",
        nargs='+',
        type=str2bool,
        default=[False],
        help="Use a recurrent version of the model.",
    )
    parser.add_argument(
        "--LRU_no_dt",
        nargs='+',
        type=str2bool,
        default=[False],
        help="Use a recurrent version of the model.",
    )
    parser.add_argument(
        "--LRU_r_max",
        nargs='+',
        type=float,
        default=[1.0],
        help="Use a recurrent version of the model.",
    )
    parser.add_argument(
        "--LRU_r_min",
        nargs='+',
        type=float,
        default=[0],
        help="Use a recurrent version of the model.",
    )
    parser.add_argument(
        "--LRU_max_phase",
        nargs='+',
        type=float,
        default=[6.28],
        help="Use a recurrent version of the model.",
    )
    parser.add_argument(
        "--taylor",
        nargs='+',
        type=str2bool,
        default=[False],
        help="Use half reset for LIFcomplex and RLIFcomplex models. True by default",
    )
    parser.add_argument(
        "--continuous",
        nargs='+',
        type=str2bool,
        default=[False],
        help="Use half reset for LIFcomplex and RLIFcomplex models. True by default",
    )
    parser.add_argument(
        "--reparam",
        nargs='+',
        type=str2bool,
        default=[False],
        help="Use half reset for LIFcomplex and RLIFcomplex models. True by default",
    )
    parser.add_argument(
        "--dt_train",
        nargs='+',
        type=str2bool,
        default=[False],
        help="Use half reset for LIFcomplex and RLIFcomplex models. True by default",
    )
    parser.add_argument(
        "--dt_uniform",
        nargs='+',
        type=str2bool,
        default=[False],
        help="Use half reset for LIFcomplex and RLIFcomplex models. True by default",
    )
    parser.add_argument(
        "--s4_init",
        nargs='+',
        type=str2bool,
        default=[False],
        help="Use half reset for LIFcomplex and RLIFcomplex models. True by default",
    )
    parser.add_argument(
        "--exp_factor",
        nargs='+',
        type=int,
        default=[2],
        help="Number of layers (including readout layer).",
    )
    parser.add_argument(
        "--no_reset",
        nargs='+',
        type=str2bool,
        default=[False],
        help="Use no reset for LIFcomplex and RLIFcomplex models. False by default",
    )
    parser.add_argument(
        "--clamp_alpha",
        nargs='+',
        type=str2bool,
        default=[False],
        help="Clamp the alpha real and imaginary parts.",
    )
    parser.add_argument(
        "--no_clamp",
        nargs='+',
        type=str2bool,
        default=[False],
        help="Clamp the alpha real and imaginary parts.",
    )
    parser.add_argument(
        "--alpha_min",
        type=float,
        default=[0.1],
        nargs='+',
        help="Max dt initializationfor LIFcomplex ",
    )
    parser.add_argument(
        "--alpha_max",
        type=float,
        default=[1.0],
        nargs='+',
        help="Max dt initializationfor LIFcomplex ",
    )
    parser.add_argument(
        "--max_phase",
        type=float,
        default=[6.28],
        nargs='+',
        help="Max dt initializationfor LIFcomplex ",
    )
    parser.add_argument(
        "--r_min",
        type=float,
        default=[0.4],
        nargs='+',
        help="Max dt initializationfor LIFcomplex ",
    )
    parser.add_argument(
        "--r_max",
        type=float,
        default=[0.9],
        nargs='+',
        help="Max dt initializationfor LIFcomplex ",
    )
    parser.add_argument(
        "--gamma_norm",
        nargs='+',
        type=str2bool,
        default=[False],
        help="Use a negative reset for real part and positive for imag part of LIFcomplex.",
    )
    parser.add_argument(
        "--complex_reset",
        nargs='+',
        type=str2bool,
        default=[False],
        help="Use a negative reset for real part and positive for imag part of LIFcomplex.",
    )
    parser.add_argument(
        "--no_b",
        nargs='+',
        type=str2bool,
        default=[False],
        help="Use (1 - alpha) to gate input instead of b for LIFcomplex.",
    )
    parser.add_argument(
        "--extra_b",
        nargs='+',
        type=str2bool,
        default=[False],
        help="Use half reset for LIFcomplex and RLIFcomplex models. True by default",
    )
    parser.add_argument(
        "--c_sum",
        nargs='+',
        type=str2bool,
        default=[False],
        help="Use (1 - alpha) to gate input instead of b for LIFcomplex.",
    )
    parser.add_argument(
        "--superspike",
        nargs='+',
        type=str2bool,
        default=[False],
        help="Use Superspike surrogate gradient. False by default",
    )
    parser.add_argument(
        "--slayer",
        nargs='+',
        type=str2bool,
        default=[False],
        help="Use SLAYER surrogate gradient. False by default",
    )
    parser.add_argument(
        "--relu_spike",
        nargs='+',
        type=str2bool,
        default=[False],
        help="Use Superspike surrogate gradient. False by default",
    )
    parser.add_argument(
        "--xavier_init",
        nargs='+',
        type=str2bool,
        default=[False],
        help="Use Xavier init as initialization for weights. False by default",
    )
    parser.add_argument(
        "--shifted_relu",
        action= 'store_true',
        help="Use threshold shift for ReLULIFcomp model",
    )
    parser.add_argument(
        "--residual",
        nargs='+',
        type=str2bool,
        default=[False],
        help="Use residual connections in all SNNs. False by default",
    )
    parser.add_argument(
        "--rst_detach",
        nargs='+',
        type=str2bool,
        default=[False],
        help="Detach reset signal specifically for autograd. True by default",
    )
    parser.add_argument(
        "--jaxreadout",
        nargs='+',
        type=str2bool,
        default=[False],
        help="Detach reset signal specifically for autograd. True by default",
    )
    parser.add_argument(
        "--dt_min",
        type=float,
        default=[0.01],
        nargs='+',
        help="Min dt initialization for LIFcomplex",
    )
    parser.add_argument(
        "--tau_m",
        type=float,
        default=[0.10768],
        nargs='+',
        help="Decay time constant for Jax readout",
    )
    parser.add_argument(
        "--ro_int",
        nargs='+',
        type=int,
        default=[0],
        help="Number of neurons in all hidden layers.",
    )
    parser.add_argument(
        "--dt_max",
        type=float,
        default=[0.5],
        nargs='+',
        help="Max dt initializationfor LIFcomplex ",
    )
    parser.add_argument(
        "--alpha_img_pi_rat_in",
        type=float,
        default=[1],
        nargs='+',
        help="Max dt initializationfor LIFcomplex ",
    )
    parser.add_argument(
        "--alpha_img_pi_rat",
        type=float,
        default=[1],
        nargs='+',
        help="",
    )
    parser.add_argument(
        "--dt_max_in",
        type=float,
        default=[0.5],
        nargs='+',
        help="Max dt initializationfor LIFcomplex ",
    )
    parser.add_argument(
        "--dt_min_in",
        type=float,
        default=[0.01],
        nargs='+',
        help="Max dt initializationfor LIFcomplex ",
    )
    parser.add_argument(
        "--alpha_range_max_in",
        type=float,
        default=[0.5],
        nargs='+',
        help="Max dt initializationfor LIFcomplex ",
    )
    parser.add_argument(
        "--alpha_range_min_in",
        type=float,
        default=[0.5],
        nargs='+',
        help="Max dt initializationfor LIFcomplex ",
    )
    parser.add_argument(
        "--alpha_rand_in",
        type=str,
        nargs='+',
        choices=["Rand", "RandN"],
        default="Rand",
        help="Type of ANN or SNN model.",
    )
    parser.add_argument(
        "--alpha_rand",
        type=str,
        nargs='+',
        choices=["Rand", "RandN"],
        default="Rand",
        help="Type of ANN or SNN model.",
    )
    parser.add_argument(
        "--alpha_range_max",
        type=float,
        default=[0.5],
        nargs='+',
        help="Max dt initializationfor LIFcomplex ",
    )
    parser.add_argument(
        "--alpha_range_min",
        type=float,
        default=[0.5],
        nargs='+',
        help="Max dt initializationfor LIFcomplex ",
    )
    parser.add_argument(
        "--alpha_in",
        type=float,
        default=[0.5],
        nargs='+',
        help="Max dt initializationfor LIFcomplex ",
    )
    parser.add_argument(
        "--alpha_range_in",
        nargs='+',
        type=str2bool,
        default=[False],
        help="Detach reset signal specifically for autograd. True by default",
    ) 
    parser.add_argument(
        "--alpha_range",
        nargs='+',
        type=str2bool,
        default=[False],
        help="Detach reset signal specifically for autograd. True by default",
    ) 
    
    parser.add_argument(
        "--alpha",
        type=float,
        default=[0.5],
        nargs='+',
        help="Max dt initializationfor LIFcomplex ",
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
        "--pdrop",
        nargs='+',
        type=float,
        default=[0.1],
        help="Dropout rate, must be between 0 and 1.",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="batchnorm",
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
