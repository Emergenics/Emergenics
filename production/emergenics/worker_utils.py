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

# worker_utils.py
import torch
import time
import traceback
import numpy as np
import networkx as nx
from scipy.stats import entropy as calculate_scipy_entropy
from scipy.sparse import coo_matrix
import warnings

# --- Metrics Helpers (Copied from Notebook Cell 2) ---
# These might be needed if run_single_instance eventually moves here too
def calculate_variance_norm(final_states_array):
    if final_states_array is None or final_states_array.size == 0: return np.nan
    try: return np.mean(np.var(final_states_array, axis=0))
    except Exception: return np.nan

def calculate_entropy_binned(data_vector, bins=10, range_lims=(-1.5, 1.5)):
    if data_vector is None or data_vector.size <= 1: return 0.0
    try:
        counts, _ = np.histogram(data_vector[~np.isnan(data_vector)], bins=bins, range=range_lims)
        return calculate_scipy_entropy(counts[counts > 0])
    except Exception: return np.nan

def calculate_pairwise_dot_energy(final_states_array, adj_matrix_coo):
    total_energy = 0.0
    num_nodes = final_states_array.shape[0]
    if num_nodes == 0 or adj_matrix_coo is None: return 0.0
    try:
        for i, j, weight in zip(adj_matrix_coo.row, adj_matrix_coo.col, adj_matrix_coo.data):
             if i < j:
                  dot_product = np.dot(final_states_array[i, :], final_states_array[j, :])
                  total_energy += weight * dot_product
        return -0.5 * total_energy
    except Exception: return np.nan


# --- Minimal Worker (for testing) ---
def minimal_gpu_worker(task_id, device_name):
    """ A simple function to test GPU access in a worker process. """
    # This function MUST be self-contained or import other needed functions/modules defined in this file or standard libraries
    try:
        if not isinstance(device_name, str) or not device_name: raise ValueError("Invalid device_name")
        local_device = torch.device(device_name)
        torch.manual_seed(task_id + int(time.time()))
        if local_device.type == 'cuda':
            if not torch.cuda.is_available(): raise RuntimeError(f"Worker {task_id}: CUDA unavailable!")
            torch.cuda.manual_seed_all(task_id + int(time.time()))
            _ = torch.randn(1, device=local_device) # Force init
        # Basic GPU computation
        a = torch.randn(10, 10, device=local_device, dtype=torch.float32)
        b = torch.matmul(a, a.T)
        c = b.sum().item()
        return {'task_id': task_id, 'success': True, 'result_sum': c, 'error': None}
    except Exception as e:
        # Return error info instead of crashing
        return {'task_id': task_id, 'success': False, 'result_sum': None, 'error': f"{type(e).__name__}: {e}", 'traceback': traceback.format_exc(limit=2)}

# --- Full GPU Step Function (Copied from Notebook Cell 2) ---
# Ensure this is the corrected version (e.g., with dim=(1,) for sparse sum)
@torch.jit.script
def hdc_5d_step_vectorized_torch(adj_sparse_tensor, current_states_tensor,
                                 rule_params_activation_threshold: float, rule_params_activation_increase_rate: float,
                                 rule_params_activation_decay_rate: float, rule_params_inhibition_threshold: float,
                                 rule_params_inhibition_increase_rate: float, rule_params_inhibition_decay_rate: float,
                                 rule_params_inhibition_feedback_threshold: float, rule_params_inhibition_feedback_strength: float,
                                 rule_params_diffusion_factor: float, rule_params_noise_level: float,
                                 rule_params_harmonic_factor: float, rule_params_w_decay_rate: float,
                                 rule_params_x_decay_rate: float, rule_params_y_decay_rate: float,
                                 device: torch.device):
    """ PyTorch implementation of the 5D HDC step function for GPU (JIT Compatible). """
    num_nodes = current_states_tensor.shape[0]; state_dim = current_states_tensor.shape[1]
    if num_nodes == 0: return current_states_tensor, torch.tensor(0.0, device=device)
    current_u=current_states_tensor[:,0]; current_v=current_states_tensor[:,1]; current_w=current_states_tensor[:,2]; current_x=current_states_tensor[:,3]; current_y=current_states_tensor[:,4]
    adj_float = adj_sparse_tensor.float(); sum_neighbor_states = torch.sparse.mm(adj_float, current_states_tensor)
    degrees = torch.sparse.sum(adj_float, dim=(1,)).to_dense(); degrees = degrees.unsqueeze(1); degrees = torch.max(degrees, torch.tensor(1.0, device=device)) # dim=(1,) fix
    mean_neighbor_states = sum_neighbor_states / degrees; neighbor_u_sum = sum_neighbor_states[:, 0]; activation_influences = neighbor_u_sum
    delta_u=torch.zeros_like(current_u); delta_v=torch.zeros_like(current_v); delta_w=torch.zeros_like(current_w); delta_x=torch.zeros_like(current_x); delta_y=torch.zeros_like(current_y)
    act_increase_mask = activation_influences > rule_params_activation_threshold; delta_u[act_increase_mask] += rule_params_activation_increase_rate * (1.0 - current_u[act_increase_mask]); delta_u -= rule_params_activation_decay_rate * current_u
    inh_fb_mask = current_u > rule_params_inhibition_feedback_threshold; delta_v[inh_fb_mask] += rule_params_inhibition_feedback_strength * (1.0 - current_v[inh_fb_mask]); delta_v -= rule_params_inhibition_decay_rate * current_v
    delta_w -= rule_params_w_decay_rate*current_w; delta_x -= rule_params_x_decay_rate*current_x; delta_y -= rule_params_y_decay_rate*current_y
    delta_states = torch.stack([delta_u, delta_v, delta_w, delta_x, delta_y], dim=1); next_states_intermediate = current_states_tensor + delta_states
    diffusion_change = rule_params_diffusion_factor * (mean_neighbor_states - current_states_tensor); next_states_intermediate += diffusion_change
    if rule_params_harmonic_factor != 0.0: harmonic_effect = rule_params_harmonic_factor * degrees.squeeze(-1) * torch.sin(neighbor_u_sum); next_states_intermediate[:, 0] += harmonic_effect # squeeze -1 fix
    noise = torch.rand_like(current_states_tensor).uniform_(-rule_params_noise_level, rule_params_noise_level); next_states_noisy = next_states_intermediate + noise
    next_states_clipped = torch.clamp(next_states_noisy, min=-1.5, max=1.5); avg_state_change = torch.mean(torch.abs(next_states_clipped - current_states_tensor))
    return next_states_clipped, avg_state_change


# --- Full Worker Function (Copied from Notebook Cell 2 - Robust Version) ---
def run_single_instance(graph, N, instance_params, trial_seed, rule_params_in, max_steps, conv_thresh, state_dim, calculate_energy=False, store_energy_history=False, energy_type='pairwise_dot', metrics_to_calc=None, device=None):
    """ Runs one NA simulation on GPU, calculates metrics (CPU), includes error handling. """
    # Define default NaN results structure INSIDE the function
    nan_results = {metric: np.nan for metric in (metrics_to_calc or ['variance_norm'])}
    nan_results.update({'convergence_time': 0, 'termination_reason': 'error_before_start',
                        'final_state_vector': None, 'final_energy': np.nan, 'energy_monotonic': False,
                        'error_message': 'Initialization failed'})
    nan_results['sensitivity_param_name'] = instance_params.get('rule_param_name'); nan_results['sensitivity_param_value'] = instance_params.get('rule_param_value')
    primary_metric_name_default = instance_params.get('primary_metric', 'variance_norm'); nan_results['order_parameter'] = np.nan; nan_results['metric_name'] = primary_metric_name_default

    try: # *** WRAP ENTIRE FUNCTION LOGIC ***
        if graph is None or graph.number_of_nodes() == 0:
             nan_results['termination_reason'] = 'empty_graph'; nan_results['error_message'] = 'Received empty graph'; return nan_results
        if device is None: device_name = 'cpu'
        elif isinstance(device, torch.device): device_name = str(device)
        else: device_name = device # Assume string if not torch.device
        local_device = torch.device(device_name) # Ensure device object

        np.random.seed(trial_seed); torch.manual_seed(trial_seed)
        if local_device.type == 'cuda': torch.cuda.manual_seed_all(trial_seed)

        node_list = sorted(list(graph.nodes())); num_nodes = len(node_list)
        adj_scipy_coo = None; adj_sparse_tensor = None
        try:
             adj_scipy_coo = nx.adjacency_matrix(graph, nodelist=node_list).tocoo()
             adj_indices = torch.LongTensor(np.vstack((adj_scipy_coo.row, adj_scipy_coo.col)))
             adj_values = torch.FloatTensor(adj_scipy_coo.data)
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
            initial_energy = calculate_pairwise_dot_energy(current_states_tensor.cpu().numpy(), adj_scipy_coo); energy_history_np.append(initial_energy)

        termination_reason = "max_steps_reached"; steps_run = 0; avg_change_cpu = torch.inf; next_states_tensor = None
        for step in range(max_steps):
            steps_run = step + 1
            try:
                next_states_tensor, avg_change_tensor = hdc_5d_step_vectorized_torch(
                    adj_sparse_tensor, current_states_tensor, rp_act_thresh, rp_act_inc, rp_act_dec, rp_inh_thresh, rp_inh_inc, rp_inh_dec,
                    rp_inh_fb_thresh, rp_inh_fb_str, rp_diff, rp_noise, rp_harm, rp_w_dec, rp_x_dec, rp_y_dec, local_device )
            except Exception as step_e:
                 termination_reason = "error_in_gpu_step"; nan_results['termination_reason'] = termination_reason; nan_results['convergence_time'] = steps_run
                 nan_results['error_message'] = f"GPU step {steps_run} failed: {step_e} | TB: {traceback.format_exc(limit=1)}"
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
            if step % 10 == 0 or step == max_steps - 1:
                 avg_change_cpu = avg_change_tensor.item()
                 if avg_change_cpu < conv_thresh: termination_reason = f"convergence_at_step_{step+1}"; break
            current_states_tensor = next_states_tensor

        final_states_np = current_states_tensor.cpu().numpy()
        results = {'convergence_time': steps_run, 'termination_reason': termination_reason, 'final_state_vector': final_states_np.flatten()}
        if metrics_to_calc is None: metrics_to_calc = ['variance_norm']
        for metric in metrics_to_calc:
             if metric == 'variance_norm': results[metric] = calculate_variance_norm(final_states_np)
             elif metric == 'entropy_dim_0' and state_dim > 0: results[metric] = calculate_entropy_binned(final_states_np[:, 0])
             elif metric == 'entropy_dim_0': results[metric] = np.nan
             else: results[metric] = np.nan
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

# --- End of worker_utils.py content ---