# Copyright (c) 2025 XXXX-1 XXXX-2 and XXXX-3 XXXX-4
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

from models.snns import SNN
from models.snns import CSiLIFLayer, SiLIFLayer, CadLIFLayer, ResonateFireLayer

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
})


### Load the trained models ###

model_paths = {'SiLIF': './reference_models/ref_SSC_SiLIF/checkpoints/best_model.pth',
'CSiLIF': './reference_models/ref_SSC_CSiLIF/checkpoints/best_model.pth',
'CadLIF': './reference_models/ref_SSC_CadLIF/checkpoints/best_model.pth',
'ResonateFire': './reference_models/ref_SSC_RF/checkpoints/best_model.pth'}

for model in model_paths:
    base_net = SNN(
                input_shape=(1, None, 140),
                layer_sizes=[512,512,35],
                neuron_type=model,
                normalization='batchnorm',
                use_bias=False,
                use_readout_layer=True,
                extra_features = {'dt_min': 0.01, 'dt_max': 5.0},
            )
    base_net.load_state_dict(torch.load(model_paths[model], map_location='cpu'))
    model_paths[model] = base_net

#################################################


def find_system_eigenvalues_numeric(tau_m_array, tau_w_array, R_array, a_w_array):
    eigenvalue_list = []

    # Assuming tau_m_array, tau_w_array, R_array, a_w_array are all numpy arrays
    for tau_m, tau_w, R, a_w in zip(tau_m_array, tau_w_array, R_array, a_w_array):
        A = np.array([[-1/tau_m, -R/tau_m], [a_w/tau_w, -1/tau_w]])  # Example system matrix
        eigenvalues = np.linalg.eigvals(A)
        eigenvalue_list.append(eigenvalues)
    
    return np.array(eigenvalue_list) 


def compute_rat_R_a(real_value, img_value, tau_m=0.01):
    rat, R_a = symbols('rat R_a')
    eq1 = Eq(-(rat + 1) / (2 * tau_m), real_value)
    eq2 = Eq(sqrt((rat**2 - 2*rat + 1 - 4*R_a * rat)*(-1)) / (2 * tau_m), img_value)
    solutions = solve([eq1, eq2], (rat, R_a))
    return solutions


### Extract eigenvalues from trained models ###

discreteEV_csilif = np.zeros((2,512,2)).astype(complex)
discreteEV_silif = np.zeros((2,512,2)).astype(complex)
discreteEV_adLIF = np.zeros((2,512,2)).astype(complex)
discreteEV_rf = np.zeros((2,512,2)).astype(complex)

for i, trained_layer in enumerate(model_paths['CSiLIF'].snn):
    if isinstance(trained_layer, CSiLIFLayer):
        alpha_img = trained_layer.alpha_img.detach().cpu().numpy()
        log_dt = trained_layer.log_dt.detach().cpu().numpy()
        log_log_alpha = trained_layer.log_log_alpha.detach().cpu().numpy()
        dt =  np.exp(log_dt)
        # dt = 0.01
        alpha_real = -np.exp(log_log_alpha)
        discreteEV_csilif[i,:,0] = np.exp((-np.exp(log_log_alpha)+1j*alpha_img)*dt)
        discreteEV_csilif[i,:,1] = np.exp((-np.exp(log_log_alpha)-1j*alpha_img)*dt)


for i, trained_layer in enumerate(model_paths['SiLIF'].snn):
    print(trained_layer)
    if isinstance(trained_layer, SiLIFLayer):
        
        dt = trained_layer.dt.detach().cpu().numpy()
        alpha = trained_layer.alpha.detach().cpu().numpy()
        alpha = np.exp(-np.exp(alpha)*np.exp(dt))
        beta = trained_layer.beta.detach().cpu().numpy()
        beta = np.exp(-np.exp(beta)*np.exp(dt))
        a = trained_layer.a.detach().cpu().numpy()
        a = np.clip(a, min=trained_layer.a_lim[0], max=trained_layer.a_lim[1])
        trans_matrix = np.stack([
            np.stack([alpha, alpha-1], axis=-1),
            np.stack([(1 - beta)*a, beta], axis=-1)
        ], axis=1)
        eigenvalues = np.linalg.eigvals(trans_matrix)

        discreteEV_silif[i,:, 0] = eigenvalues[:,0]
        discreteEV_silif[i,:, 1] =  eigenvalues[:,1]

for i, trained_layer in enumerate(model_paths['CadLIF'].snn):
    print(trained_layer)
    if isinstance(trained_layer, CadLIFLayer):
        
        alpha = trained_layer.alpha.detach().cpu().numpy()
        beta = trained_layer.beta.detach().cpu().numpy()
        dt = 0.001
        tau_m = - dt/np.log(alpha)
        tau_w = - dt/np.log(beta)
        a = trained_layer.a.detach().cpu().numpy()
        R = np.ones(alpha.shape)
        a_w = a
        eigenvalues = find_system_eigenvalues_numeric(tau_m, tau_w, R, a_w)
        real_parts = np.real(eigenvalues)
        imag_parts = np.imag(eigenvalues)
        discreteEV_adLIF[i,:] = np.exp((real_parts+1j*imag_parts)*dt)


for i, trained_layer in enumerate(model_paths['ResonateFire'].snn):
    print(trained_layer)
    if isinstance(trained_layer, ResonateFireLayer):
        
        alpha_im = trained_layer.alpha_im.detach().cpu().numpy()
        alpha_real = trained_layer.alpha_real.detach().cpu().numpy()
        #alpha_real = np.clip(alpha_real, a_max = -0.1)
        dt = 0.004

        discreteEV_rf[i,:,0] = np.exp((alpha_real+1j*alpha_im)*dt)
        discreteEV_rf[i,:,1] = np.exp((alpha_real-1j*alpha_im)*dt)

def random_subset(arr, fraction=0.5):
    num_elements = arr.shape[1]  # Assuming shape is (i, n, ...)
    selected_indices = np.random.choice(num_elements, size=int(num_elements * fraction), replace=False)
    return arr[:, selected_indices]

# Select 50% randomly
subset_adLIF = random_subset(discreteEV_adLIF)
subset_rf = random_subset(discreteEV_rf)
subset_csilif = random_subset(discreteEV_csilif)
subset_silif = random_subset(discreteEV_silif)

#################################################

colors = {'AdLIF':'#f4a261', 
'CadLIF':'#e9c46a', 
'RF':'#e76f51', 
'C-SiLIF':'#2a9d8f', 
'SiLIF': '#264653'}

i = 0
fig = plt.figure(figsize=(9, 3.5))

# Define GridSpec for two main subfigures
gs_master = GridSpec(1, 2, width_ratios=[1, 1.2], figure=fig, wspace=0.4)

# Left subplot (scatter only, full -1 to 1 range)
ax_left = fig.add_subplot(gs_master[0])
ax_left.grid(True, zorder=0)
ax_left.scatter(discreteEV_adLIF[i,:,:].real, discreteEV_adLIF[i,:,:].imag, color=colors['CadLIF'],marker='o',s=15, zorder=5,alpha=0.7, label='cAdLIF')
ax_left.scatter(discreteEV_csilif[i,:].real, discreteEV_csilif[i,:].imag, color=colors['C-SiLIF'], marker='o',s=15, zorder=5,alpha=0.7, label='C-SiLIF')
ax_left.scatter(discreteEV_silif[i,:].real, discreteEV_silif[i,:].imag, color=colors['SiLIF'], marker='o',s=15, zorder=5,alpha=0.7, label='SiLIF')
ax_left.scatter(discreteEV_rf[i,:].real, discreteEV_rf[i,:].imag, color=colors['RF'], marker='o',s=15, zorder=5,alpha=0.7, label='RF')

circle_full = plt.Circle((0, 0), radius=1, color='black', fill=False, linewidth=2, zorder=3, alpha=0.7)
ax_left.add_artist(circle_full)
ax_left.axhline(0, color='black', linewidth=0.5)
ax_left.axvline(0, color='black', linewidth=0.5)

ax_left.set_xlim((-1, 1))
ax_left.set_ylim((-1, 1))
ax_left.set_xlabel(r'Re($\bar{\lambda}$)', fontsize=15)
ax_left.set_ylabel(r'Im($\bar{\lambda}$)', fontsize=15)
ax_left.tick_params(axis='both', labelsize=14)
ax_left.set_title('(a) Full Range', fontsize=16)

legend_left = ax_left.legend(loc='upper left', fontsize=12, scatterpoints=1, handletextpad=0.5)
for handle in legend_left.legend_handles:
    handle.set_sizes([100])

# Right subplot (scatter + histograms with original limits)
gs_right = gs_master[1].subgridspec(4, 4, hspace=0.1, wspace=0.1)
ax_scatter = fig.add_subplot(gs_right[1:, :-1])
ax_histx = fig.add_subplot(gs_right[0, :-1], sharex=ax_scatter)
ax_histy = fig.add_subplot(gs_right[1:, -1], sharey=ax_scatter)

# Main scatter plot on the right
ax_scatter.grid(True, zorder=0)
ax_scatter.scatter(discreteEV_adLIF[i,:,:].real, discreteEV_adLIF[i,:,:].imag, color=colors['CadLIF'],marker='o',s=15, zorder=5,alpha=0.7, label='cAdLIF')
ax_scatter.scatter(discreteEV_csilif[i,:].real, discreteEV_csilif[i,:].imag, color=colors['C-SiLIF'], marker='o',s=15, zorder=5,alpha=0.7, label='C-SiLIF')
ax_scatter.scatter(discreteEV_silif[i,:].real, discreteEV_silif[i,:].imag, color=colors['SiLIF'], marker='o',s=15, zorder=5,alpha=0.7, label='SiLIF')
ax_scatter.scatter(discreteEV_rf[i,:].real, discreteEV_rf[i,:].imag, color=colors['RF'], marker='o',s=15, zorder=5,alpha=0.7, label='RF')

circle = plt.Circle((0, 0), radius=1, color='black', fill=False, linewidth=2, zorder=3, alpha=0.7)
ax_scatter.add_artist(circle)
ax_scatter.axhline(0, color='black', linewidth=0.5)
ax_scatter.axvline(0, color='black', linewidth=0.5)

ax_scatter.set_xlim((-0.2, 1))
ax_scatter.set_ylim((-0.5, 0.5))
ax_scatter.set_xlabel(r'Re($\bar{\lambda}$)', fontsize=15)
ax_scatter.set_ylabel(r'Im($\bar{\lambda}$)', fontsize=15)
ax_scatter.tick_params(axis='both', labelsize=14)


# Histograms for real (top) and imaginary (right)
ax_histx.hist([discreteEV_adLIF[i,:,:].real.flatten(), discreteEV_csilif[i,:].real.flatten(),
               discreteEV_silif[i,:].real.flatten(), discreteEV_rf[i,:].real.flatten()],
               bins=30, color=[colors['CadLIF'], colors['C-SiLIF'], colors['SiLIF'], colors['RF']], stacked=False, linewidth=1.5)
ax_histx.axis('off')
ax_histx.set_yscale('log')
ax_histx.set_title('(b) Partial Range with Histograms', fontsize=16)

ax_histy.hist([discreteEV_adLIF[i,:,:].imag.flatten(), discreteEV_csilif[i,:].imag.flatten(),
               discreteEV_silif[i,:].imag.flatten(), discreteEV_rf[i,:].imag.flatten()],
               bins=30, orientation='horizontal', color=[colors['CadLIF'], colors['C-SiLIF'], colors['SiLIF'], colors['RF']], stacked=False, linewidth=1.5)
ax_histy.axis('off')
ax_histy.set_xscale('log')


plt.tight_layout()
fig.savefig("./figs/figure2.png", bbox_inches='tight', dpi=300)
plt.show()

print("Figure 2 saved as figure2.png")