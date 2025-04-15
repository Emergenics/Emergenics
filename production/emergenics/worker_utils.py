# worker_utils.py
# Combined Phase 1 and Phase 2 worker functions and helpers.
# CORRECTED: torch.sparse.sum dim=(1,) fix applied.

# Copyright 2025, Michael G. Young II, Emergenics Foundation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import time
import traceback
import numpy as np
import networkx as nx
from scipy.stats import entropy as calculate_scipy_entropy
from scipy.sparse import coo_matrix
import warnings
import copy # Needed for Phase 2 worker


# --- Metrics Helpers (Copied from Notebook Cell 2 / Phase 1&2) ---
def calculate_variance_norm(final_states_array):
    """Calculates variance across nodes, averaged across dimensions."""
    if final_states_array is None: return np.nan
    num_nodes = final_states_array.shape[0]
    if num_nodes == 0: return 0.0
    try:
        variance_per_dim = np.var(final_states_array, axis=0)
        mean_variance = np.mean(variance_per_dim)
        # Check for NaN/Inf in result
        if np.isnan(mean_variance) or np.isinf(mean_variance):
            return np.nan
        else:
            return mean_variance
    except Exception as e_var:
        warnings.warn(f"Variance norm calculation failed: {e_var}", RuntimeWarning)
        return np.nan

def calculate_entropy_binned(data_vector, bins=10, range_lims=(-1.5, 1.5)):
    """Calculates Shannon entropy for a single dimension using numpy histogram."""
    if data_vector is None: return np.nan
    # Ensure data_vector is numpy array
    if not isinstance(data_vector, np.ndarray):
        try: data_vector = np.array(data_vector)
        except Exception: return np.nan

    if data_vector.size <= 1: return 0.0
    try:
        valid_data = data_vector[~np.isnan(data_vector)]
        if valid_data.size <= 1: return 0.0
        # Check range validity
        if range_lims[0] >= range_lims[1]:
            warnings.warn(f"Invalid range_lims for entropy: {range_lims}. Using data min/max.", RuntimeWarning)
            min_val = np.min(valid_data)
            max_val = np.max(valid_data)
            if min_val == max_val: return 0.0 # All values are the same
            hist_range = (min_val, max_val)
        else:
            hist_range = range_lims

        counts, _ = np.histogram(valid_data, bins=bins, range=hist_range)
        # Filter out zero counts before calculating probabilities
        non_zero_counts = counts[counts > 0]
        if non_zero_counts.size == 0: return 0.0 # No counts in any bin
        # Calculate entropy using scipy's entropy function
        entropy_value = calculate_scipy_entropy(non_zero_counts, base=None) # Use natural log
        return entropy_value
    except Exception as e_ent:
        warnings.warn(f"Entropy calculation failed: {e_ent}", RuntimeWarning)
        return np.nan

def calculate_pairwise_dot_energy(final_states_array, adj_matrix_coo):
    """Calculates E = -0.5 * sum_{i<j} A[i,j] * dot(Si, Sj) using numpy and sparse COO"""
    total_energy = 0.0
    if final_states_array is None: return np.nan
    num_nodes = final_states_array.shape[0]
    if num_nodes == 0: return 0.0
    if adj_matrix_coo is None: return 0.0
    try:
        # Ensure COO format
        if not isinstance(adj_matrix_coo, coo_matrix):
             try: adj_matrix_coo = coo_matrix(adj_matrix_coo)
             except Exception: warnings.warn("Failed to convert adj matrix to COO for energy calc.", RuntimeWarning); return np.nan

        adj_row = adj_matrix_coo.row
        adj_col = adj_matrix_coo.col
        adj_data = adj_matrix_coo.data
        num_edges = len(adj_data)
        edge_idx = 0
        # Iterate through sparse matrix non-zero elements using while loop
        while edge_idx < num_edges:
            i = adj_row[edge_idx]
            j = adj_col[edge_idx]
            weight = adj_data[edge_idx]

            # Process only upper triangle (i < j) to avoid double counting for undirected graphs
            if i < j:
                # Bounds check for safety
                if i < num_nodes and j < num_nodes:
                    state_i = final_states_array[i, :]
                    state_j = final_states_array[j, :]
                    # Check if states are valid before dot product
                    if not np.isnan(state_i).any() and not np.isnan(state_j).any():
                         dot_product = np.dot(state_i, state_j)
                         if not np.isnan(dot_product): # Check dot product result
                             total_energy = total_energy + (weight * dot_product)
                         else:
                             warnings.warn(f"NaN dot product encountered for edge ({i},{j}). Skipping.", RuntimeWarning)
                    else:
                         warnings.warn(f"NaN state vector encountered for node {i} or {j}. Skipping edge.", RuntimeWarning)

                else:
                    warnings.warn(f"Index out of bounds during energy calculation ({i},{j} vs N={num_nodes}). Skipping edge.", RuntimeWarning)
            edge_idx = edge_idx + 1 # Increment loop counter

        # Apply the -0.5 factor
        final_energy = -0.5 * total_energy
        return final_energy
    except Exception as e_en:
        warnings.warn(f"Energy calculation failed: {e_en}", RuntimeWarning)
        traceback.print_exc(limit=1)
        return np.nan

# --- Phase 2 Metrics Calculation Helpers ---
def calculate_mean_final_state_entropy(final_states_array, bins=10, range_lims=(-1.5, 1.5)):
    """Calculates entropy for each dimension and averages."""
    if final_states_array is None: return np.nan
    num_dims = final_states_array.shape[1]
    if num_dims == 0: return 0.0

    entropies = []
    dim_idx = 0
    while dim_idx < num_dims:
        entropy_val = calculate_entropy_binned(final_states_array[:, dim_idx], bins=bins, range_lims=range_lims)
        if not np.isnan(entropy_val):
            entropies.append(entropy_val)
        dim_idx = dim_idx + 1

    if len(entropies) > 0:
        mean_entropy = np.mean(entropies)
        return mean_entropy
    else:
        return np.nan

def calculate_relaxation_time(avg_change_history, conv_thresh, perturbation_end_step):
    """Estimates steps to reconverge after perturbation."""
    if avg_change_history is None or len(avg_change_history) <= perturbation_end_step:
        return np.nan
    steps_after_perturb = 0
    index = perturbation_end_step
    while index < len(avg_change_history):
        # Handle potential NaN values in history
        if np.isnan(avg_change_history[index]):
             index += 1 # Skip NaN and continue counting steps
             continue
        if avg_change_history[index] < conv_thresh:
            return steps_after_perturb
        steps_after_perturb += 1
        index += 1
    # If never reconverged within history length
    return steps_after_perturb # Return max steps checked

def calculate_perturbation_spread(final_states_perturbed, final_states_baseline, threshold=0.1):
    """Calculates fraction of nodes significantly changed by perturbation."""
    if final_states_perturbed is None or final_states_baseline is None: return np.nan
    if final_states_perturbed.shape != final_states_baseline.shape: return np.nan
    num_nodes = final_states_perturbed.shape[0]
    if num_nodes == 0: return 0.0
    try:
        # Calculate Euclidean distance between baseline and perturbed state for each node
        diff = final_states_perturbed - final_states_baseline
        dist_sq = np.sum(diff * diff, axis=1)
        distances = np.sqrt(dist_sq)
        # Count nodes where distance exceeds threshold
        nodes_affected = np.sum(distances > threshold)
        spread_fraction = float(nodes_affected) / float(num_nodes)
        return spread_fraction
    except Exception as e_spread:
        warnings.warn(f"Perturbation spread calculation failed: {e_spread}", RuntimeWarning)
        return np.nan

# --- Full GPU Step Function (JIT Compatible) ---
@torch.jit.script
def hdc_5d_step_vectorized_torch(adj_sparse_tensor, current_states_tensor,
                                 rule_params_activation_threshold: float, rule_params_activation_increase_rate: float,
                                 rule_params_activation_decay_rate: float, rule_params_inhibition_threshold: float, # Unused
                                 rule_params_inhibition_increase_rate: float, # Unused
                                 rule_params_inhibition_decay_rate: float,
                                 rule_params_inhibition_feedback_threshold: float, rule_params_inhibition_feedback_strength: float,
                                 rule_params_diffusion_factor: float, rule_params_noise_level: float,
                                 rule_params_harmonic_factor: float, rule_params_w_decay_rate: float,
                                 rule_params_x_decay_rate: float, rule_params_y_decay_rate: float,
                                 device: torch.device):
    """ PyTorch implementation of the 5D HDC step function for GPU (JIT Compatible). """
    num_nodes = current_states_tensor.shape[0]
    state_dim = current_states_tensor.shape[1]
    if num_nodes == 0:
        return current_states_tensor, torch.tensor(0.0, device=device)

    current_u = current_states_tensor[:, 0]
    current_v = current_states_tensor[:, 1] if state_dim > 1 else torch.zeros_like(current_u)
    current_w = current_states_tensor[:, 2] if state_dim > 2 else torch.zeros_like(current_u)
    current_x = current_states_tensor[:, 3] if state_dim > 3 else torch.zeros_like(current_u)
    current_y = current_states_tensor[:, 4] if state_dim > 4 else torch.zeros_like(current_u)

    adj_float = adj_sparse_tensor.float()
    sum_neighbor_states = torch.sparse.mm(adj_float, current_states_tensor)
    # *** CORRECTED DIMENSION FOR SPARSE SUM ***
    degrees = torch.sparse.sum(adj_float, dim=(1,)).to_dense().unsqueeze(1) # Corrected: dim=(1,)
    degrees = torch.max(degrees, torch.tensor(1.0, device=device))
    mean_neighbor_states = sum_neighbor_states / degrees
    neighbor_u_sum = sum_neighbor_states[:, 0]
    activation_influences = neighbor_u_sum

    delta_u = torch.zeros_like(current_u); delta_v = torch.zeros_like(current_v)
    delta_w = torch.zeros_like(current_w); delta_x = torch.zeros_like(current_x)
    delta_y = torch.zeros_like(current_y)

    act_increase_mask = activation_influences > rule_params_activation_threshold
    increase_u_val = rule_params_activation_increase_rate * (1.0 - current_u)
    delta_u = torch.where(act_increase_mask, delta_u + increase_u_val, delta_u)
    delta_u = delta_u - (rule_params_activation_decay_rate * current_u)

    if state_dim > 1:
        inh_fb_mask = current_u > rule_params_inhibition_feedback_threshold
        increase_v_val = rule_params_inhibition_feedback_strength * (1.0 - current_v)
        delta_v = torch.where(inh_fb_mask, delta_v + increase_v_val, delta_v)
        delta_v = delta_v - (rule_params_inhibition_decay_rate * current_v)

    if state_dim > 2: delta_w = delta_w - (rule_params_w_decay_rate * current_w)
    if state_dim > 3: delta_x = delta_x - (rule_params_x_decay_rate * current_x)
    if state_dim > 4: delta_y = delta_y - (rule_params_y_decay_rate * current_y)

    delta_list = [delta_u]
    if state_dim > 1: delta_list.append(delta_v)
    if state_dim > 2: delta_list.append(delta_w)
    if state_dim > 3: delta_list.append(delta_x)
    if state_dim > 4: delta_list.append(delta_y)
    delta_states = torch.stack(delta_list, dim=1)

    next_states_intermediate = current_states_tensor + delta_states
    diffusion_change = rule_params_diffusion_factor * (mean_neighbor_states - current_states_tensor)
    next_states_intermediate = next_states_intermediate + diffusion_change

    if abs(rule_params_harmonic_factor) > 1e-9:
        harmonic_effect = rule_params_harmonic_factor * degrees.squeeze(-1) * torch.sin(neighbor_u_sum)
        next_states_intermediate[:, 0] = next_states_intermediate[:, 0] + harmonic_effect

    noise = torch.rand_like(current_states_tensor).uniform_(-rule_params_noise_level, rule_params_noise_level)
    next_states_noisy = next_states_intermediate + noise
    next_states_clipped = torch.clamp(next_states_noisy, min=-1.5, max=1.5)
    avg_state_change = torch.mean(torch.abs(next_states_clipped - current_states_tensor))

    return next_states_clipped, avg_state_change


# --- Phase 1 Worker Function (Unchanged - Kept for potential backward compatibility tests) ---
def run_single_instance(graph, N, instance_params, trial_seed, rule_params_in, max_steps, conv_thresh, state_dim, calculate_energy=False, store_energy_history=False, energy_type='pairwise_dot', metrics_to_calc=None, device=None):
    """ Runs one NA simulation on GPU, calculates metrics (CPU), includes error handling. """
    # Define default NaN results structure INSIDE the function
    nan_results = {metric: np.nan for metric in (metrics_to_calc or ['variance_norm'])}
    nan_results.update({'convergence_time': 0, 'termination_reason': 'error_before_start',
                        'final_state_vector': None, 'final_energy': np.nan, 'energy_monotonic': False,
                        'error_message': 'Initialization failed'})
    nan_results['sensitivity_param_name'] = instance_params.get('rule_param_name'); nan_results['sensitivity_param_value'] = instance_params.get('rule_param_value')
    primary_metric_name_default = instance_params.get('primary_metric', 'variance_norm'); nan_results['order_parameter'] = np.nan; nan_results['metric_name'] = primary_metric_name_default
    param_key_nan = next((k for k in instance_params if k.endswith('_value')), 'unknown_sweep_param'); nan_results[param_key_nan] = instance_params.get(param_key_nan, np.nan)

    try: # *** WRAP ENTIRE FUNCTION LOGIC ***
        if graph is None or graph.number_of_nodes() == 0:
             nan_results['termination_reason'] = 'empty_graph'; nan_results['error_message'] = 'Received empty graph'; return nan_results
        if device is None: device_name = 'cpu'
        elif isinstance(device, torch.device): device_name = str(device)
        else: device_name = device # Assume string if not torch.device
        local_device = torch.device(device_name) # Ensure device object

        np.random.seed(trial_seed); torch.manual_seed(trial_seed)
        if local_device.type == 'cuda':
             if torch.cuda.is_available(): torch.cuda.manual_seed_all(trial_seed)
             else: nan_results['termination_reason'] = 'cuda_error'; nan_results['error_message'] = f'CUDA specified but unavailable.'; return nan_results


        node_list = sorted(list(graph.nodes())); num_nodes = len(node_list)
        adj_scipy_coo = None; adj_sparse_tensor = None
        try:
             adj_scipy_coo = nx.adjacency_matrix(graph, nodelist=node_list, weight=None).tocoo()
             adj_indices = torch.LongTensor(np.vstack((adj_scipy_coo.row, adj_scipy_coo.col)))
             adj_values = torch.ones(len(adj_scipy_coo.data), dtype=torch.float32) # Use 1 for unweighted
             adj_shape = adj_scipy_coo.shape
             adj_sparse_tensor = torch.sparse_coo_tensor(adj_indices, adj_values, adj_shape, device=local_device)
        except Exception as adj_e: nan_results['termination_reason'] = 'adj_error'; nan_results['error_message'] = f'Adj matrix failed: {adj_e}'; return nan_results

        rule_params = rule_params_in.copy()
        if instance_params.get('rule_param_name') and instance_params.get('rule_param_value') is not None: rule_params[instance_params['rule_param_name']] = instance_params['rule_param_value']
        rp_act_thresh=float(rule_params['activation_threshold']); rp_act_inc=float(rule_params['activation_increase_rate']); rp_act_dec=float(rule_params['activation_decay_rate'])
        rp_inh_thresh=float(rule_params['inhibition_threshold']); rp_inh_inc=float(rule_params['inhibition_increase_rate']); rp_inh_dec=float(rule_params['inhibition_decay_rate'])
        rp_inh_fb_thresh=float(rule_params['inhibition_feedback_threshold']); rp_inh_fb_str=float(rule_params['inhibition_feedback_strength'])
        rp_diff=float(rule_params['diffusion_factor']); rp_noise=float(rule_params['noise_level']); rp_harm=float(rule_params['harmonic_factor'])
        rp_w_dec=float(rule_params['w_decay_rate']); rp_x_dec=float(rule_params['x_decay_rate']); rp_y_dec=float(rule_params['y_decay_rate'])

        initial_states_tensor = torch.FloatTensor(num_nodes, state_dim).uniform_(-0.1, 0.1).to(local_device)
        current_states_tensor = initial_states_tensor
        energy_history_np = []
        if calculate_energy and store_energy_history:
            try:
                initial_energy = calculate_pairwise_dot_energy(current_states_tensor.cpu().numpy(), adj_scipy_coo); energy_history_np.append(initial_energy)
            except Exception: energy_history_np.append(np.nan)


        termination_reason = "max_steps_reached"; steps_run = 0; avg_change_cpu = torch.inf; next_states_tensor = None
        for step in range(max_steps):
            steps_run = step + 1
            try:
                next_states_tensor, avg_change_tensor = hdc_5d_step_vectorized_torch(
                    adj_sparse_tensor, current_states_tensor, rp_act_thresh, rp_act_inc, rp_act_dec, rp_inh_thresh, rp_inh_inc, rp_inh_dec,
                    rp_inh_fb_thresh, rp_inh_fb_str, rp_diff, rp_noise, rp_harm, rp_w_dec, rp_x_dec, rp_y_dec, local_device )
            except Exception as step_e:
                 termination_reason = "error_in_gpu_step"; nan_results['termination_reason'] = termination_reason; nan_results['convergence_time'] = steps_run
                 nan_results['error_message'] = f"GPU step {steps_run} failed: {type(step_e).__name__}: {step_e} | TB: {traceback.format_exc(limit=1)}"
                 final_states_np = current_states_tensor.cpu().numpy() if current_states_tensor is not None else None; nan_results['final_state_vector'] = final_states_np.flatten() if final_states_np is not None else None
                 if final_states_np is not None: # Try metrics on last state
                    if 'variance_norm' in (metrics_to_calc or []): nan_results['variance_norm'] = calculate_variance_norm(final_states_np)
                    if 'entropy_dim_0' in (metrics_to_calc or []) and state_dim > 0: nan_results['entropy_dim_0'] = calculate_entropy_binned(final_states_np[:, 0])
                    if calculate_energy: nan_results['final_energy'] = calculate_pairwise_dot_energy(final_states_np, adj_scipy_coo)
                 del adj_sparse_tensor, current_states_tensor, initial_states_tensor;
                 if next_states_tensor is not None: del next_states_tensor
                 if local_device.type == 'cuda': torch.cuda.empty_cache()
                 return nan_results
            if calculate_energy and store_energy_history:
                 try: current_energy = calculate_pairwise_dot_energy(next_states_tensor.cpu().numpy(), adj_scipy_coo); energy_history_np.append(current_energy)
                 except Exception: energy_history_np.append(np.nan)
            # Check convergence periodically or at end
            # if step % 10 == 0 or step == max_steps - 1: # Less frequent check
            avg_change_cpu = avg_change_tensor.item()
            if avg_change_cpu < conv_thresh:
                termination_reason = f"convergence_at_step_{step+1}"
                current_states_tensor = next_states_tensor # Ensure final state is the converged one
                break # Exit loop
            current_states_tensor = next_states_tensor

        final_states_np = current_states_tensor.cpu().numpy()
        results = {'convergence_time': steps_run, 'termination_reason': termination_reason, 'final_state_vector': final_states_np.flatten()}

        # Add primary sweep param value back into results
        param_key = next((k for k in instance_params if k.endswith('_value')), None)
        if param_key: results[param_key] = instance_params[param_key]
        else: results['unknown_sweep_param'] = np.nan

        # Calculate metrics
        if metrics_to_calc is None: metrics_to_calc = ['variance_norm']
        for metric in metrics_to_calc:
             if metric == 'variance_norm': results[metric] = calculate_variance_norm(final_states_np)
             elif metric == 'entropy_dim_0' and state_dim > 0: results[metric] = calculate_entropy_binned(final_states_np[:, 0])
             elif metric == 'entropy_dim_0': results[metric] = np.nan # Handle case where dim 0 doesn't exist
             else:
                  if metric not in results: # Avoid overwriting energy etc. if already calculated
                      results[metric] = np.nan
        is_monotonic_result = np.nan
        if calculate_energy:
            results['final_energy'] = calculate_pairwise_dot_energy(final_states_np, adj_scipy_coo)
            if store_energy_history and len(energy_history_np) > 1:
                 energy_history_np = np.array(energy_history_np); valid_energy_hist = energy_history_np[~np.isnan(energy_history_np)]
                 if len(valid_energy_hist) > 1: diffs = np.diff(valid_energy_hist); is_monotonic_result = bool(np.all(diffs <= 1e-6))
            results['energy_monotonic'] = is_monotonic_result
        else: results['final_energy'] = np.nan; results['energy_monotonic'] = np.nan
        primary_metric_name = instance_params.get('primary_metric', 'variance_norm'); results['order_parameter'] = results.get(primary_metric_name, np.nan); results['metric_name'] = primary_metric_name
        results['sensitivity_param_name'] = instance_params.get('rule_param_name'); results['sensitivity_param_value'] = instance_params.get('rule_param_value')
        results['error_message'] = None

        del adj_sparse_tensor, current_states_tensor, initial_states_tensor
        if 'next_states_tensor' in locals() and next_states_tensor is not None: del next_states_tensor
        if local_device.type == 'cuda': torch.cuda.empty_cache()
        return results

    except Exception as worker_e:
         tb_str = traceback.format_exc()
         nan_results['termination_reason'] = 'unhandled_worker_error'; nan_results['error_message'] = f"Unhandled: {worker_e} | TB: {tb_str}"
         try:
             if 'current_states_tensor' in locals() and current_states_tensor is not None: final_states_np_err = current_states_tensor.cpu().numpy(); nan_results['final_state_vector'] = final_states_np_err.flatten() if final_states_np_err is not None else None
         except Exception: pass
         return nan_results


# --- Phase 2 Worker Function ---
def run_single_instance_phase2(
    graph, N, instance_params, trial_seed, rule_params_in,
    max_steps, conv_thresh, state_dim,
    calculate_energy=False, store_energy_history=False, # Inherited from Phase 1
    energy_type='pairwise_dot', metrics_to_calc=None, device=None, # Inherited
    # --- Phase 2 Additions ---
    store_state_history=False, # Flag to store history
    state_history_interval=1,  # Interval for storing history
    perturbation_params=None, # Dictionary with perturbation details
    phase2_metrics_to_calc=None # List of Phase 2 specific metrics
    ):
    """
    Runs one NA simulation, MODIFIED for Phase 2.
    Includes state history storage, perturbation application, and calculation
    of Phase 2 metrics (e.g., relaxation time, spread, state entropy).
    """
    # --- Combine metric lists ---
    all_metrics_requested = (metrics_to_calc or []) + (phase2_metrics_to_calc or [])
    all_metrics_requested = sorted(list(set(all_metrics_requested))) # Unique sorted list

    # --- Default Error Result ---
    nan_results = {metric: np.nan for metric in all_metrics_requested}
    nan_results.update({
        'convergence_time': 0, 'termination_reason': 'error_before_start',
        'final_state_vector': None, 'final_energy': np.nan, 'energy_monotonic': False,
        'error_message': 'Initialization failed',
        'state_history': None, 'avg_change_history': None,
        'relaxation_time': np.nan, 'perturbation_spread': np.nan,
        'mean_final_state_entropy': np.nan, # Ensure Phase 2 metrics have defaults
        'baseline_state_for_spread': None
    })
    primary_metric_name_default = instance_params.get('primary_metric', 'variance_norm')
    nan_results['order_parameter'] = np.nan; nan_results['metric_name'] = primary_metric_name_default
    nan_results['sensitivity_param_name'] = instance_params.get('rule_param_name'); nan_results['sensitivity_param_value'] = instance_params.get('rule_param_value')
    param_key_nan = next((k for k in instance_params if k.endswith('_value')), 'unknown_sweep_param')
    nan_results[param_key_nan] = instance_params.get(param_key_nan, np.nan)

    try:
        # --- Setup ---
        if graph is None or graph.number_of_nodes() == 0:
             nan_results['termination_reason']='empty_graph'; nan_results['error_message']='Received empty graph'; return nan_results
        local_device = None
        if isinstance(device, torch.device): local_device = device
        elif isinstance(device, str):
            try: local_device = torch.device(device)
            except Exception as e_dev: nan_results['termination_reason'] = 'device_error'; nan_results['error_message'] = f'Invalid device: {device}, Error: {e_dev}'; return nan_results
        else: local_device = torch.device('cpu')

        np.random.seed(trial_seed); torch.manual_seed(trial_seed)
        if local_device.type == 'cuda':
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(trial_seed)
            else: nan_results['termination_reason'] = 'cuda_error'; nan_results['error_message'] = f'CUDA specified but unavailable.'; return nan_results

        node_list = sorted(list(graph.nodes())); num_nodes = len(node_list); adj_scipy_coo = None; adj_sparse_tensor = None
        try:
             adj_scipy_coo = nx.adjacency_matrix(graph, nodelist=node_list, weight=None).tocoo()
             adj_indices = torch.LongTensor(np.vstack((adj_scipy_coo.row, adj_scipy_coo.col)))
             adj_values = torch.ones(len(adj_scipy_coo.data), dtype=torch.float32)
             adj_shape = adj_scipy_coo.shape
             adj_sparse_tensor = torch.sparse_coo_tensor(adj_indices, adj_values, adj_shape, device=local_device)
        except Exception as adj_e: nan_results['termination_reason'] = 'adj_error'; nan_results['error_message'] = f'Adj matrix failed: {adj_e}'; return nan_results

        rule_params = rule_params_in.copy()
        if instance_params.get('rule_param_name') and instance_params.get('rule_param_value') is not None: rule_params[instance_params['rule_param_name']] = instance_params['rule_param_value']
        rp_act_thresh=float(rule_params['activation_threshold']); rp_act_inc=float(rule_params['activation_increase_rate']); rp_act_dec=float(rule_params['activation_decay_rate'])
        rp_inh_thresh=float(rule_params['inhibition_threshold']); rp_inh_inc=float(rule_params['inhibition_increase_rate']); rp_inh_dec=float(rule_params['inhibition_decay_rate'])
        rp_inh_fb_thresh=float(rule_params['inhibition_feedback_threshold']); rp_inh_fb_str=float(rule_params['inhibition_feedback_strength'])
        rp_diff=float(rule_params['diffusion_factor']); rp_noise=float(rule_params['noise_level']); rp_harm=float(rule_params['harmonic_factor'])
        rp_w_dec=float(rule_params['w_decay_rate']); rp_x_dec=float(rule_params['x_decay_rate']); rp_y_dec=float(rule_params['y_decay_rate'])

        # --- Initialization ---
        initial_states_tensor = torch.FloatTensor(num_nodes, state_dim).uniform_(-0.1, 0.1).to(local_device)
        current_states_tensor = initial_states_tensor
        state_history_list = []
        avg_change_history_list = []
        energy_history_np = []
        baseline_final_state_for_spread = None # Initialize

        if store_state_history and state_history_interval > 0:
            state_history_list.append(current_states_tensor.cpu().numpy().copy())
        if calculate_energy and store_energy_history:
            try: energy_history_np.append(calculate_pairwise_dot_energy(current_states_tensor.cpu().numpy(), adj_scipy_coo))
            except Exception: energy_history_np.append(np.nan)

        # --- Perturbation Setup ---
        is_perturbation_run = perturbation_params is not None and isinstance(perturbation_params, dict)
        perturb_start = -1; perturb_end = -1; perturb_nodes_indices = []; perturb_dim = -1; perturb_val = 0.0
        if is_perturbation_run:
            perturb_start = perturbation_params.get('apply_at_step', -1)
            perturb_duration = perturbation_params.get('duration_steps', 0)
            perturb_end = perturb_start + perturb_duration
            perturb_node_frac = perturbation_params.get('target_node_fraction', 0)
            perturb_dim = perturbation_params.get('target_dimension', -1)
            perturb_val = perturbation_params.get('perturbation_value', 0.0)
            if not (0 <= perturb_start < max_steps and perturb_duration > 0 and 0 < perturb_node_frac <= 1.0 and 0 <= perturb_dim < state_dim):
                 is_perturbation_run = False # Disable if invalid
            else:
                 num_perturb_nodes = max(1, int(num_nodes * perturb_node_frac))
                 perturb_nodes_indices = np.random.choice(num_nodes, num_perturb_nodes, replace=False).tolist()

        # --- Simulation Loop ---
        termination_reason = "max_steps_reached"; steps_run = 0; avg_change_cpu = torch.inf; next_states_tensor = None
        step = 0
        while step < max_steps:
            steps_run = step + 1
            perturbation_active_this_step = False
            if is_perturbation_run and step >= perturb_start and step < perturb_end:
                perturbation_active_this_step = True
                input_state_tensor = current_states_tensor.clone()
                input_state_tensor[perturb_nodes_indices, perturb_dim] = perturb_val
            else: input_state_tensor = current_states_tensor

            try:
                next_states_tensor, avg_change_tensor = hdc_5d_step_vectorized_torch(
                    adj_sparse_tensor, input_state_tensor, rp_act_thresh, rp_act_inc, rp_act_dec, rp_inh_thresh, rp_inh_inc, rp_inh_dec,
                    rp_inh_fb_thresh, rp_inh_fb_str, rp_diff, rp_noise, rp_harm, rp_w_dec, rp_x_dec, rp_y_dec, local_device )
            except Exception as step_e:
                 termination_reason = "error_in_gpu_step"; nan_results['termination_reason'] = termination_reason; nan_results['convergence_time'] = steps_run
                 nan_results['error_message'] = f"GPU step {steps_run} fail: {type(step_e).__name__}: {step_e}|TB:{traceback.format_exc(limit=1)}"
                 try: final_states_np_err = current_states_tensor.cpu().numpy(); nan_results['final_state_vector'] = final_states_np_err.flatten()
                 except Exception: pass
                 del adj_sparse_tensor, current_states_tensor, initial_states_tensor;
                 if 'next_states_tensor' in locals() and next_states_tensor is not None: del next_states_tensor
                 if 'input_state_tensor' in locals() and input_state_tensor is not None: del input_state_tensor
                 if local_device.type == 'cuda': torch.cuda.empty_cache();
                 return nan_results

            avg_change_cpu = avg_change_tensor.item(); avg_change_history_list.append(avg_change_cpu)
            if store_state_history and state_history_interval > 0 and (step % state_history_interval == 0 or step == max_steps - 1):
                 state_history_list.append(next_states_tensor.cpu().numpy().copy())
            if calculate_energy and store_energy_history:
                 try: energy_history_np.append(calculate_pairwise_dot_energy(next_states_tensor.cpu().numpy(), adj_scipy_coo))
                 except Exception: energy_history_np.append(np.nan)

            converged = False
            if not perturbation_active_this_step:
                 if avg_change_cpu < conv_thresh:
                      converged = True; termination_reason = f"convergence_at_step_{step+1}"
                      if is_perturbation_run and step < perturb_start and baseline_final_state_for_spread is None:
                           baseline_final_state_for_spread = next_states_tensor.cpu().numpy().copy()

            current_states_tensor = next_states_tensor
            if converged and not (is_perturbation_run and step < perturb_end): break
            step = step + 1

        # --- Final State & Metrics ---
        final_states_np = current_states_tensor.cpu().numpy()
        results = {'convergence_time': steps_run, 'termination_reason': termination_reason,
                   'final_state_vector': final_states_np.flatten(), 'error_message': None}
        param_key = next((k for k in instance_params if k.endswith('_value')), None)
        if param_key: results[param_key] = instance_params[param_key]
        else: results['unknown_sweep_param'] = np.nan
        results['sensitivity_param_name'] = instance_params.get('rule_param_name'); results['sensitivity_param_value'] = instance_params.get('rule_param_value')

        # Phase 1 Metrics
        if metrics_to_calc is None: metrics_to_calc = []
        metric_idx = 0
        while metric_idx < len(metrics_to_calc):
            metric = metrics_to_calc[metric_idx]
            if metric == 'variance_norm': results[metric] = calculate_variance_norm(final_states_np)
            elif metric == 'entropy_dim_0' and state_dim > 0: results[metric] = calculate_entropy_binned(final_states_np[:, 0])
            elif metric == 'entropy_dim_0': results[metric] = np.nan
            metric_idx += 1

        is_monotonic_result = False
        if calculate_energy:
            results['final_energy'] = calculate_pairwise_dot_energy(final_states_np, adj_scipy_coo)
            if store_energy_history and len(energy_history_np) > 1:
                 valid_energy_hist = np.array(energy_history_np); valid_energy_hist = valid_energy_hist[~np.isnan(valid_energy_hist)]
                 if len(valid_energy_hist) > 1: diffs = np.diff(valid_energy_hist); is_monotonic_result = bool(np.all(diffs <= 1e-6))
            results['energy_monotonic'] = is_monotonic_result
        else: results['final_energy'] = np.nan; results['energy_monotonic'] = np.nan
        primary_metric_name = instance_params.get('primary_metric', 'variance_norm'); results['order_parameter'] = results.get(primary_metric_name, np.nan); results['metric_name'] = primary_metric_name

        # Phase 2 Metrics
        results['avg_change_history'] = avg_change_history_list
        results['state_history'] = state_history_list if store_state_history else None
        if 'mean_final_state_entropy' in all_metrics_requested:
             results['mean_final_state_entropy'] = calculate_mean_final_state_entropy(final_states_np)
        if is_perturbation_run:
             if 'relaxation_time' in all_metrics_requested: results['relaxation_time'] = calculate_relaxation_time(avg_change_history_list, conv_thresh, perturb_end)
             results['baseline_state_for_spread'] = baseline_final_state_for_spread # Store for later comparison
             # Perturbation spread requires baseline - calculated in post-processing
             results['perturbation_spread'] = np.nan # Placeholder
        else:
             results['relaxation_time'] = np.nan; results['perturbation_spread'] = np.nan; results['baseline_state_for_spread'] = None

        # --- Final Cleanup ---
        del adj_sparse_tensor, current_states_tensor, initial_states_tensor;
        if 'next_states_tensor' in locals() and next_states_tensor is not None: del next_states_tensor
        if 'input_state_tensor' in locals() and input_state_tensor is not None: del input_state_tensor
        if local_device.type == 'cuda': torch.cuda.empty_cache()
        return results

    except Exception as worker_e:
         tb_str = traceback.format_exc(limit=2)
         nan_results['termination_reason'] = 'unhandled_worker_error'
         nan_results['error_message'] = f"Unhandled Worker Error: {type(worker_e).__name__}: {worker_e} | TB: {tb_str}"
         try:
             if 'current_states_tensor' in locals() and current_states_tensor is not None: nan_results['final_state_vector'] = current_states_tensor.cpu().numpy().flatten()
         except Exception: pass
         try:
             if 'adj_sparse_tensor' in locals() and adj_sparse_tensor is not None: del adj_sparse_tensor
             if 'current_states_tensor' in locals() and current_states_tensor is not None: del current_states_tensor
             if 'initial_states_tensor' in locals() and initial_states_tensor is not None: del initial_states_tensor
             if 'next_states_tensor' in locals() and next_states_tensor is not None: del next_states_tensor
             if 'input_state_tensor' in locals() and input_state_tensor is not None: del input_state_tensor
             if 'local_device' in locals() and local_device is not None and local_device.type == 'cuda': torch.cuda.empty_cache()
         except NameError: pass
         return nan_results

# --- End of worker_utils.py content ---