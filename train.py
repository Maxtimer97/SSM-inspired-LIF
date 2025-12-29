# Copyright (c) 2025 Maxime Fabre and Lyubov Dudchenko
# This file is part of SSM-inspired-LIF, released under the MIT License.

# Modified from: https://github.com/idiap/sparch
# Original license: BSD 3-Clause (see third_party/sparch/LICENSE)

# SPDX-FileCopyrightText: © 2022 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Alexandre Bittar <abittar@idiap.ch>
# SPDX-License-Identifier: BSD-3-Clause

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

        # Training config
        self.evaluate_pretrained = config.pop('evaluate_pretrained')
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

        self.workers = config.pop('num_workers')

        self.nb_steps = config.pop('nb_steps')
        self.max_time = config.pop('max_time')
        self.spatial_bin = config.pop('spatial_bin')

        self.debug = config.pop('debug')

        self.seed = config.pop('seed')
        self.set_seed()

        self.extra_config = config #remaining config options

        # Initialize logging and output folders
        self.init_exp_folders()
        self.init_logging()

        # Set device
        self.device = torch.device(f"cuda:{device}") #torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"\nDevice is set to {self.device}\n")

        
        # Initialize dataloaders and model
        self.init_dataset()
        self.init_model()

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
        if not self.evaluate_pretrained:

            # Initialize best accuracy
            best_epoch, best_acc = 0, 0

            # Loop over epochs (training + validation)
            logging.info("\n------ Begin training ------\n")

            for e in range(best_epoch + 1, best_epoch + self.nb_epochs + 1):
                self.train_one_epoch(e)
                best_epoch, new_best_acc = self.valid_one_epoch(e, best_epoch, best_acc)
                if self.dataset_name=="sc":
                    if new_best_acc > 0.92 and new_best_acc>best_acc:
                        test_acc, test_sop = self.test_one_epoch(self.test_loader)
                elif self.dataset_name=="ssc":
                    if new_best_acc > 0.74 and new_best_acc>best_acc:
                        test_acc, test_sop = self.test_one_epoch(self.test_loader)
                best_acc = new_best_acc



            logging.info(f"\nBest valid acc at epoch {best_epoch}: {best_acc}\n")
            logging.info("\n------ Training finished ------\n")

            if not self.debug:
                wandb.log({"best valid acc":best_acc}, commit=False)
            self.best_val_acc = best_acc

            # Loading best model
            if self.save_best:
                self.net.load_state_dict(torch.load(
                    f"{self.checkpoint_dir}/best_model.pth", map_location=self.device)
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
            test_acc, test_sop = self.test_one_epoch(self.test_loader)
        else:
            test_acc, test_sop = self.test_one_epoch(self.valid_loader)
            logging.info(
                "\nThis dataset uses the same split for validation and testing.\n"
            )

        return test_acc, test_sop

    def init_exp_folders(self):
        """
        This function defines the output folders for the experiment.
        """
        # Check if path exists for loading pretrained model
        if self.evaluate_pretrained:
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
            outname = self.dataset_name + "_" + str(self.nb_steps) + "steps_" + self.model_type + "_"
            outname += str(self.nb_layers) + "lay" + str(self.nb_hiddens)
            outname += "_lr" + str(self.lr) + "_drop" + str(self.pdrop)
            outname += "_seed" +str(self.seed) 

            exp_folder = "exp/test_exps/" + outname.replace(".", "_")

        
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


        if self.model_type in ["LIF", "CSiLIF", "SiLIF", "adLIF", "CadLIF", "ResonateFire"]:

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

        elif self.model_type in ["MLP", "RNN", "LiGRU", "GRU"]:

            self.net = ANN(
                input_shape=input_shape,
                layer_sizes=layer_sizes,
                ann_type=self.model_type,
                dropout=self.pdrop,
                normalization=self.normalization,
                use_bias=self.use_bias,
                bidirectional=self.bidirectional,
                use_readout_layer=True,
            ).to(self.device)

            logging.info(f"\nCreated new non-spiking model:\n {self.net}\n")

        else:
            raise ValueError(f"Invalid model type {self.model_type}")


        if self.evaluate_pretrained:
            self.net.load_state_dict(torch.load(self.load_path, map_location=self.device))
            logging.info(f"\nLoaded model at: {self.load_path}\n {self.net}\n")

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

            loss_val = self.loss_fn(output, y)
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
            self.opt.step()

            # Compute accuracy with labels
            pred = torch.argmax(output, dim=1)
            acc = np.mean((y == pred).detach().cpu().numpy())
            accs.append(acc)

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
                loss_val = self.loss_fn(output, y)
                losses.append(loss_val.item())

                # Compute accuracy with labels
                pred = torch.argmax(output, dim=1)
                acc = np.mean((y == pred).detach().cpu().numpy())
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

            self.scheduler.step(valid_acc)

            # Update best epoch and accuracy
            if valid_acc > best_acc:
                best_acc = valid_acc
                best_epoch = e

                # Save best model
                if self.save_best:
                    torch.save(self.net.state_dict(), f"{self.checkpoint_dir}/best_model.pth")
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
                loss_val = self.loss_fn(output, y)
                losses.append(loss_val.item())

                # Compute accuracy with labels
                pred = torch.argmax(output, dim=1)
                acc = np.mean((y == pred).detach().cpu().numpy())
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

            return test_acc, epoch_sop/step
