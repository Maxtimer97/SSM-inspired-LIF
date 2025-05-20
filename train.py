# Copyright (c) 2025 Maxime Fabre and Lyubov Dudchenko
# This file is part of SSM-inspired-LIF, released under the MIT License.
#
# Modified from: https://github.com/idiap/sparch
# Original license: BSD 3-Clause (see third_party/sparch/LICENSE)
#
# SPDX-FileCopyrightText: © 2022 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Alexandre Bittar <abittar@idiap.ch>
# SPDX-License-Identifier: BSD-3-Clause
#
# This file was originally part of the sparch package.

"""
This is to define the experiment class used to perform training and testing
of ANNs and SNNs on all speech command recognition datasets.
"""
import errno
import logging
import os
import time
from datetime import timedelta
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR, CosineAnnealingLR

from dataloaders.nonspiking_datasets import load_hd_or_sc
from dataloaders.spiking_datasets import load_shd_or_ssc
from models.anns import ANN
from models.snns import SNN
from parsers.model_config import print_model_options
from parsers.training_config import print_training_options
from DCLS.construct.modules import Dcls1d

import wandb

import json


logger = logging.getLogger(__name__)


class Experiment:
    """
    Class for training and testing models (ANNs and SNNs) on all four
    datasets for speech command recognition (shd, ssc, hd and sc).
    """

    def __init__(self, config, device):

        # Put wandb config objects into a standard dict
        config = {k: v for k, v in config.items()}

        print_model_options(config)
        print_training_options(config)

        # New model config
        self.model_type = config.pop('model_type')
        self.nb_layers = config.pop('nb_layers')
        self.nb_hiddens = config.pop('nb_hiddens')
        self.pdrop = config.pop('pdrop')
        self.normalization = config.pop('normalization')
        self.use_bias = config.pop('use_bias')
        self.bidirectional = config.pop('bidirectional')

        self.delays = True if 'delay' in self.model_type.lower() else False

        # self.lif_feature = config.pop('lif_feature')

        # Training config
        self.use_pretrained_model = config.pop('use_pretrained_model')
        self.only_do_testing = config.pop('only_do_testing')
        self.load_exp_folder = config.pop('load_exp_folder')
        self.new_exp_folder = config.pop('new_exp_folder')
        self.dataset_name = config.pop('dataset_name')
        self.data_folder = config.pop('data_folder')
        self.log_tofile = config.pop('log_tofile')
        self.save_best = config.pop('save_best')
        self.batch_size = config.pop('batch_size')
        self.nb_epochs = config.pop('nb_epochs')
        self.start_epoch = config.pop('start_epoch')
        self.lr = config.pop('lr')
        self.scheduler_patience = config.pop('scheduler_patience')
        self.scheduler_factor = config.pop('scheduler_factor')
        self.use_regularizers = config.pop('use_regularizers')
        self.reg_factor = config.pop('reg_factor')
        self.reg_fmin = config.pop('reg_fmin')
        self.reg_fmax = config.pop('reg_fmax')
        self.use_augm = config.pop('use_augm')
        self.s4_opt = config.pop('s4_opt')

        self.workers = config.pop('num_workers')
        self.snnax_optim = config.pop('snnax_optim')

        self.nb_steps = config.pop('nb_steps')
        self.max_time = config.pop('max_time')
        self.spatial_bin = config.pop('spatial_bin')

        self.debug = config.pop('debug')

        self.seed = config.pop('seed')
        self.set_seed()
        self.extra_config = config
        # Initialize logging and output folders
        self.init_exp_folders()
        self.init_logging()

        # Set device
        self.device = torch.device(f"cuda:{device}") #torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"\nDevice is set to {self.device}\n")

        

        # Initialize dataloaders and model
        self.init_dataset()
        self.init_model()

        # Define optimizer
        if self.s4_opt:
            # All parameters in the model
            all_parameters = list(self.net.parameters())

            # General parameters don't contain the special _optim key
            params = [p for p in all_parameters if not hasattr(p, "_optim")]

            # Create an optimizer with the general parameters
            self.opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=0.01)

            # Add parameters with special hyperparameters
            hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
            hps = [
                dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
            ]  # Unique dicts
            for hp in hps:
                params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
                self.opt.add_param_group(
                    {"params": params, **hp}
                )
        
        elif self.snnax_optim:
            self.opt = torch.optim.AdamW(self.net.parameters(), self.lr, 
                                         weight_decay=0.01)
            
            num_train_iters = len(self.train_loader)
            # Number of steps for warmup and total decay steps
            warmup_steps = int(num_train_iters * 1)
            total_steps = int(self.nb_epochs * num_train_iters)

            # Linear warmup function
            def lr_lambda_warmup(step):
                if step < warmup_steps:
                    # Warmup from learning_rate / warmup_factor to learning_rate
                    return (step / max(1, warmup_steps)) * (1 - 1 / 3) + (1 / 3)
                return 1.0

            # Cosine decay after warmup
            cosine_scheduler = CosineAnnealingLR(self.opt, T_max=total_steps - warmup_steps, 
                                                eta_min=self.lr * 0.046538863126080535)

            # LambdaLR for warmup phase
            warmup_scheduler = LambdaLR(self.opt, lr_lambda=lr_lambda_warmup)

            # Function to step both warmup and cosine scheduler
            def scheduler_step(step):
                if step < warmup_steps:
                    warmup_scheduler.step()  # Use warmup LR schedule
                else:
                    cosine_scheduler.step()  # Use cosine LR schedule

            self.scheduler_step = lambda x: scheduler_step(x)
        elif self.delays:
        # pick out all params with "positions" in their name
            pos_params = []
            for m in self.net.modules():
                if isinstance(m, Dcls1d):
                    pos_params.append(m.P)

            # 2) build a set of their ids
            pos_param_ids = { id(p) for p in pos_params }

            # 3) the “other” group is every param whose id is NOT in that set
            other_params = [
                p for p in self.net.parameters()
                if id(p) not in pos_param_ids
            ]

            # now build your optimizer with two param-groups
            self.opt = torch.optim.Adam([
                { "params": pos_params,   "lr": self.lr*10 },  
                { "params": other_params, "lr": self.lr              },  
            ])
        else:
            self.opt = torch.optim.Adam(self.net.parameters(), self.lr)

        # Define learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            optimizer=self.opt,
            mode="max",
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
            min_lr=1e-6,
        )
        # Define loss function
        self.loss_fn = nn.CrossEntropyLoss()

        self.best_val_acc = 0



    def forward(self):
        """
        This function performs model training with the configuration
        specified by the class initialization.
        """
        if not self.only_do_testing:

            # Initialize best accuracy
            if self.use_pretrained_model:
                logging.info("\n------ Using pretrained model ------\n")
                best_epoch, best_acc = self.valid_one_epoch(self.start_epoch, 0, 0)
            else:
                best_epoch, best_acc = 0, 0

            # Loop over epochs (training + validation)
            logging.info("\n------ Begin training ------\n")

            for e in range(best_epoch + 1, best_epoch + self.nb_epochs + 1):
                self.train_one_epoch(e)
                best_epoch, new_best_acc = self.valid_one_epoch(e, best_epoch, best_acc)
                if self.dataset_name=="sc":
                    if new_best_acc > 0.92 and new_best_acc>best_acc:
                        self.test_one_epoch(self.test_loader)
                elif self.dataset_name=="ssc":
                    if new_best_acc > 0.74 and new_best_acc>best_acc:
                        self.test_one_epoch(self.test_loader)
                best_acc = new_best_acc



            logging.info(f"\nBest valid acc at epoch {best_epoch}: {best_acc}\n")
            logging.info("\n------ Training finished ------\n")

            if not self.debug:
                wandb.log({"best valid acc":best_acc}, commit=False)
            self.best_val_acc = best_acc

            # Loading best model
            if self.save_best:
                self.net = torch.load(
                    f"{self.checkpoint_dir}/best_model.pth", map_location=self.device
                )
                logging.info(
                    f"Loading best model, epoch={best_epoch}, valid acc={best_acc}"
                )
            else:
                logging.info(
                    "Cannot load best model because save_best option is "
                    "disabled. Model from last epoch is used for testing."
                )

        # Test trained model
        if self.dataset_name in ["sc", "ssc"]:
            self.test_one_epoch(self.test_loader)
        else:
            self.test_one_epoch(self.valid_loader)
            logging.info(
                "\nThis dataset uses the same split for validation and testing.\n"
            )

    def init_exp_folders(self):
        """
        This function defines the output folders for the experiment.
        """
        # Check if path exists for loading pretrained model
        if self.use_pretrained_model:
            exp_folder = self.load_exp_folder
            self.load_path = exp_folder + "/checkpoints/best_model.pth"
            if not os.path.exists(self.load_path):
                raise FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), self.load_path
                )

        # Use given path for new model folder
        elif self.new_exp_folder is not None:
            exp_folder = self.new_exp_folder

        # Generate a path for new model from chosen config
        else:
            outname = self.dataset_name + "_" + self.model_type + "_"
            outname += str(self.nb_layers) + "lay" + str(self.nb_hiddens)
            outname += "_drop" + str(self.pdrop) + "_" + str(self.normalization)
            outname += "_bias" if self.use_bias else "_nobias"
            outname += "_in"+self.extra_config.get('input_layer_type', False) if self.extra_config.get('use_input_layer', False)  else ""
            #outname += "_bdir" if self.bidirectional else "_udir"
            #outname += "_reg" if self.use_regularizers else "_noreg"
            #outname += "_lr" + str(self.lr)
            outname += "_seed" +str(self.seed) #+ '_'.join(self.lif_feature.keys())

            exp_folder = "exp/test_exps/" + outname.replace(".", "_")

        # # For a new model check that out path does not exist
        # if not self.use_pretrained_model and os.path.exists(exp_folder):
        #     raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), exp_folder)
        
        # Create folders to store experiment
        self.log_dir = exp_folder + "/log/"
        self.checkpoint_dir = exp_folder + "/checkpoints/"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.exp_folder = exp_folder

    def init_logging(self):
        """
        This function sets the experimental log to be written either to
        a dedicated log file, or to the terminal.
        """
        if self.log_tofile:
            logging.FileHandler(
                filename=self.log_dir + "exp.log",
                mode="a",
                encoding=None,
                delay=False,
            )
            logging.basicConfig(
                filename=self.log_dir + "exp.log",
                level=logging.INFO,
                format="%(message)s",
            )
        else:
            logging.basicConfig(
                level=logging.INFO,
                format="%(message)s",
            )

    def init_dataset(self):
        """
        This function prepares dataloaders for the desired dataset.
        """
        # For the spiking datasets
        if self.dataset_name in ["shd", "ssc"]:

            self.nb_inputs = 700//self.spatial_bin
            self.nb_outputs = 20 if self.dataset_name == "shd" else 35

            self.train_loader = load_shd_or_ssc(
                dataset_name=self.dataset_name,
                data_folder=self.data_folder,
                split="train",
                batch_size=self.batch_size,
                nb_steps=self.nb_steps,
                max_time = self.max_time,
                spatial_bin = self.spatial_bin,
                shuffle=True,
                workers=self.workers,
            )
            self.valid_loader = load_shd_or_ssc(
                dataset_name=self.dataset_name,
                data_folder=self.data_folder,
                split="valid",
                batch_size=self.batch_size,
                nb_steps=self.nb_steps,
                max_time = self.max_time,
                spatial_bin = self.spatial_bin,
                shuffle=False,
                workers=self.workers,
            )
            if self.dataset_name == "ssc":
                self.test_loader = load_shd_or_ssc(
                    dataset_name=self.dataset_name,
                    data_folder=self.data_folder,
                    split="test",
                    batch_size=self.batch_size,
                    nb_steps=self.nb_steps,
                    max_time = self.max_time,
                    spatial_bin = self.spatial_bin,
                    shuffle=False,
                    workers=self.workers,
                )
            if self.use_augm:
                logging.warning(
                    "\nWarning: Data augmentation not implemented for SHD and SSC.\n"
                )

        # For the non-spiking datasets
        elif self.dataset_name in ["hd", "sc"]:

            self.nb_inputs = 40
            self.nb_outputs = 20 if self.dataset_name == "hd" else 35

            self.train_loader = load_hd_or_sc(
                dataset_name=self.dataset_name,
                data_folder=self.data_folder,
                split="train",
                batch_size=self.batch_size,
                use_augm=self.use_augm,
                shuffle=True,
                workers=self.workers,
            )
            self.valid_loader = load_hd_or_sc(
                dataset_name=self.dataset_name,
                data_folder=self.data_folder,
                split="valid",
                batch_size=self.batch_size,
                use_augm=self.use_augm,
                shuffle=False,
                workers=self.workers,
            )
            if self.dataset_name == "sc":
                self.test_loader = load_hd_or_sc(
                    dataset_name=self.dataset_name,
                    data_folder=self.data_folder,
                    split="test",
                    batch_size=self.batch_size,
                    use_augm=self.use_augm,
                    shuffle=False,
                    workers=self.workers,
                )
            if self.use_augm:
                logging.info("\nData augmentation is used\n")

        else:
            raise ValueError(f"Invalid dataset name {self.dataset_name}")

    def init_model(self):
        """
        This function either loads pretrained model or builds a
        new model (ANN or SNN) depending on chosen config.
        """
        input_shape = (self.batch_size, None, self.nb_inputs)
        layer_sizes = [self.nb_hiddens] * (self.nb_layers - 1) + [self.nb_outputs]

        if self.use_pretrained_model:
            self.net = torch.load(self.load_path, map_location=self.device)
            logging.info(f"\nLoaded model at: {self.load_path}\n {self.net}\n")

        # elif self.model_type in ["LIF", "CSiLIF", "SiLIF", "LIFfeature", "adLIFnoClamp", "LIFfeatureDim", "adLIF", "CadLIF", "CadLIFAblation", "RingInitLIFcomplex", "ResonateFire", "RAFAblation", "BRF", "RSEadLIF", "adLIFclamp", "RLIF", "RadLIF", "LIFcomplex","LIFcomplexBroad", "LIFrealcomplex", "ReLULIFcomplex", "RLIFcomplex","RLIFcomplex1MinAlphaNoB","RLIFcomplex1MinAlpha", "LIFcomplex_gatedB", "LIFcomplex_gatedDt", "LIFcomplexDiscr", "DelaySiLIF"]:

        self.net = SNN(
            input_shape=input_shape,
            layer_sizes=layer_sizes,
            neuron_type=self.model_type,
            dropout=self.pdrop,
            normalization=self.normalization,
            use_bias=self.use_bias,
            bidirectional=self.bidirectional,
            use_readout_layer=True,
            extra_features = self.extra_config
        ).to(self.device)

        logging.info(f"\nCreated new spiking model:\n {self.net}\n")

        # elif self.model_type in ["MLP", "RNN", "LiGRU", "GRU"]:

        #     self.net = ANN(
        #         input_shape=input_shape,
        #         layer_sizes=layer_sizes,
        #         ann_type=self.model_type,
        #         dropout=self.pdrop,
        #         normalization=self.normalization,
        #         use_bias=self.use_bias,
        #         bidirectional=self.bidirectional,
        #         use_readout_layer=True,
        #     ).to(self.device)

        #     logging.info(f"\nCreated new non-spiking model:\n {self.net}\n")

        # else:
        #     raise ValueError(f"Invalid model type {self.model_type}")

        # if 'LIFcomplex' not in self.model_type:
        #     torch.set_float32_matmul_precision('high')
        #     self.net = torch.compile(self.net)

        self.nb_params = sum(
            p.numel() for p in self.net.parameters() if p.requires_grad
        )
        logging.info(f"Total number of trainable parameters is {self.nb_params}")
        

    def set_seed(self):
        seed = self.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def get_acc(self, output, y):
        if self.extra_config['ro_int']!=0:
            maxs = torch.argmax(output, dim=-1)
            pred = []
            # Iterate over the elements in maxs
            for m in maxs:
                most_common = torch.mode(m).values.item()  # Find the mode (most frequent element)
                pred.append(most_common)
        
            pred = torch.tensor(pred).to(y.device)
            # Calculate the accuracy by comparing predictions with true labels
            acc = np.mean((y == pred).detach().cpu().numpy())                
        else:
            pred = torch.argmax(output, dim=1)
            acc = np.mean((y == pred).detach().cpu().numpy())
        
        return acc

    def get_loss(self, output, y):

        if self.extra_config['ro_int']!=0:
            # One-hot encode the labels
            y_one_hot = F.one_hot(y, num_classes=output.size(-1)).float()
            # Tile the one-hot encoding (equivalent to jnp.tile in JAX)
            y_one_hot_tiled = y_one_hot.unsqueeze(1).repeat(1, output.size(1), 1)
            # Calculate the cross-entropy loss (equivalent to jax.nn.log_softmax)
            loss_val = -(y_one_hot_tiled * F.log_softmax(output, dim=-1)).mean()                
        else:
            # Compute loss
            loss_val = self.loss_fn(output, y)
        
        return loss_val

    def train_one_epoch(self, e):
        """
        This function trains the model with a single pass over the
        training split of the dataset.
        """
        start = time.time()
        self.net.train()
        losses, accs = [], []
        epoch_spike_rate = 0
        epoch_sop = 0

        # Loop over batches from train set
        for step, (x, _, y) in enumerate(self.train_loader):

            # Dataloader uses cpu to allow pin memory
            x = x.to(self.device)
            y = y.to(self.device)

            # Forward pass through network
            output, firing_rates, sop = self.net(x)

            loss_val = self.get_loss(output, y)
            losses.append(loss_val.item())

            # Spike activity
            if self.net.is_snn:
                epoch_spike_rate += torch.mean(firing_rates)
                epoch_sop += sop

                if self.use_regularizers:
                    reg_quiet = F.relu(self.reg_fmin - firing_rates).sum()
                    reg_burst = F.relu(firing_rates - self.reg_fmax).sum()
                    loss_val += self.reg_factor * (reg_quiet + reg_burst)

            # Backpropagate
            self.opt.zero_grad()
            loss_val.backward()
            if self.snnax_optim:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.1)
            self.opt.step()

            if self.snnax_optim:
                self.scheduler_step(step)

            # Compute accuracy with labels

            acc = self.get_acc(output, y)
            accs.append(acc)

            if self.delays:
                for layer in self.net.snn:
                    layer.W.clamp_parameters()

        if self.delays:
            self.net.decrease_sig(e)

        # Learning rate of whole epoch
        current_lr = self.opt.param_groups[-1]["lr"]
        logging.info(f"Epoch {e}: lr={current_lr}")

        # Train loss of whole epoch
        train_loss = np.mean(losses)
        logging.info(f"Epoch {e}: train loss={train_loss}")

        # Train accuracy of whole epoch
        train_acc = np.mean(accs)
        logging.info(f"Epoch {e}: train acc={train_acc}")

        # Train spike activity of whole epoch
        if self.net.is_snn:
            epoch_spike_rate /= step
            logging.info(f"Epoch {e}: train mean act rate={epoch_spike_rate}")

        end = time.time()
        elapsed = str(timedelta(seconds=end - start))
        logging.info(f"Epoch {e}: train elapsed time={elapsed}")

        if not self.debug:
            wandb.log({"train_loss":train_loss, "train_acc":train_acc, "train sparsity": 1-epoch_spike_rate, "train sop": epoch_sop/step}, commit=False)

    def valid_one_epoch(self, e, best_epoch, best_acc):
        """
        This function tests the model with a single pass over the
        validation split of the dataset.
        """
        with torch.no_grad():

            self.net.eval()
            losses, accs = [], []
            epoch_spike_rate = 0
            epoch_sop = 0

            # Loop over batches from validation set
            for step, (x, _, y) in enumerate(self.valid_loader):

                # Dataloader uses cpu to allow pin memory
                x = x.to(self.device)
                y = y.to(self.device)

                # Forward pass through network
                output, firing_rates, sop = self.net(x)

                # Compute loss
                loss_val = self.get_loss(output, y)
                losses.append(loss_val.item())

                # Compute accuracy with labels
                acc = self.get_acc(output, y)
                accs.append(acc)

                # Spike activity
                if self.net.is_snn:
                    epoch_spike_rate += torch.mean(firing_rates)
                    epoch_sop += sop

            # Validation loss of whole epoch
            valid_loss = np.mean(losses)
            logging.info(f"Epoch {e}: valid loss={valid_loss}")

            # Validation accuracy of whole epoch
            valid_acc = np.mean(accs)
            logging.info(f"Epoch {e}: valid acc={valid_acc}")

            # Validation spike activity of whole epoch
            if self.net.is_snn:
                epoch_spike_rate /= step
                logging.info(f"Epoch {e}: valid mean act rate={epoch_spike_rate}")


            if not self.debug:
                wandb.log({"valid loss":valid_loss, "valid acc":valid_acc, "valid sparsity": 1-epoch_spike_rate, "valid sop": epoch_sop/step}, commit=True)

            # Update learning rate
            if not self.snnax_optim:
                self.scheduler.step(valid_acc)

            # Update best epoch and accuracy
            if valid_acc > best_acc:
                best_acc = valid_acc
                best_epoch = e

                # Save best model
                if self.save_best:
                    torch.save(self.net, f"{self.checkpoint_dir}/best_model.pth")
                    logging.info(f"\nBest model saved with valid acc={valid_acc}")

            logging.info("\n-----------------------------\n")

            return best_epoch, best_acc

    def test_one_epoch(self, test_loader):
        """
        This function tests the model with a single pass over the
        testing split of the dataset.
        """
        with torch.no_grad():

            self.net.eval()
            losses, accs = [], []
            epoch_spike_rate = 0
            epoch_sop = 0

            logging.info("\n------ Begin Testing ------\n")

            # Loop over batches from test set
            for step, (x, _, y) in enumerate(test_loader):

                # Dataloader uses cpu to allow pin memory
                x = x.to(self.device)
                y = y.to(self.device)

                # Forward pass through network
                output, firing_rates, sop = self.net(x)

                # Compute loss
                loss_val = self.get_loss(output, y)
                losses.append(loss_val.item())

                # Compute accuracy with labels
                acc = self.get_acc(output, y)
                accs.append(acc)

                # Spike activity
                if self.net.is_snn:
                    epoch_spike_rate += torch.mean(firing_rates)
                    epoch_sop += sop

            # Test loss
            test_loss = np.mean(losses)
            logging.info(f"Test loss={test_loss}")

            # Test accuracy
            test_acc = np.mean(accs)
            logging.info(f"Test acc={test_acc}")


            # Test spike activity
            if self.net.is_snn:
                epoch_spike_rate /= step
                logging.info(f"Test mean act rate={epoch_spike_rate}")
            
            if not self.debug:
                wandb.log({"test loss":test_loss, "test acc":test_acc, "test sparsity": 1-epoch_spike_rate, "test sop": epoch_sop/step}, commit=False)
            

            logging.info("\n-----------------------------\n")

            new_result_entry = {
            'test_acc': test_acc,
            'number_layers': self.nb_layers,
            'number_neurons': self.nb_hiddens,
            'model_type': self.model_type, 
            'best_val_acc': self.best_val_acc
            }

            # self.save_results_to_json(new_result_entry)

    def save_results_to_json(self,result_entry):
        if self.dataset_name == "ssc":
            filename='resultsSSC.json'
        if self.dataset_name == "shd":
            filename='resultsSHD.json'

        try:
            with open(filename, 'r') as file:
                results = json.load(file)
        except FileNotFoundError:
            results = []

        results.append(result_entry)
        
        with open(filename, 'w') as file:
            json.dump(results, file, indent=4)
