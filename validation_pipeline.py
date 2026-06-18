"""
Unified Synthetic Data Learning Pipeline
=========================================

Reproduces the synthetic validation experiment from the paper:
    Stage 1: Learn x_AB inspector variances (AGVI)
    Stage 2: Learn x_A|AB variances (conditional AGVI)
    Stage 3: Learn x_D|CD variances (conditional AGVI)
    Stage 4: Reconstruct grades {A, B, C, D} via Gaussian product

All comparisons are vs the ground truth trajectories.
Last-observed value is included as a naive baseline.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional

from src.core.state_models import SSMparam
from src.core.kalman_filter import KalmanFilter
from src.core.inspector_manager import InspectorManager
from src.core.utils import gaussian_product
from data.synthetic_bridge_data import generate_bridge_network_data

jax.config.update("jax_enable_x64", True)


# =============================================================================
# 1. Configuration
# =============================================================================

# Synthetic data parameters
CONFIG = {
    'num_bridges': 100,  # 100 × 10 = 1000 series
    'categories': ['poutre'],
    'num_elements_per_category': 50,
    'num_inspector_ids': 20,
    'inspector_sigma_range': (0.03, 0.12),  # Known ground truth range
    'sigma_w': 1e-5,  # Process noise for data generation
    'year_start': 2008,
    'year_end': 2050,
    'min_obs': 4,
    'max_obs': 10,
    'gap_range': (2, 4),
    'seed': 42
}

# Initial conditions for synthetic data
INIT_CONDITIONS = {
    'poutre': {
        'a':  {'mean': [1., -0.065, -0.0035], 'cov': np.diag([0.05, 0.005, 0.0005])**2},
        'ab': {'mean': [1., -0.00, -0.0035], 'cov': np.diag([0.05, 0.005, 0.0005])**2},
        'd':  {'mean': [-0.3,  0.00,  0.0008], 'cov': np.diag([0.1,  0.00001, 0.00001])**2},
    }
}

# Training parameters
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Grid search parameters for initial state
# A and AB gates: negative speed (deterioration)
GRID_MU_V0              = [-1/15, -1/50, -1/100]
# D gate: positive speed (improvement, opposite direction)
# GRID_MU_V0_D = [1/50, 1/100, 1/200]
GRID_VAR_V0_RATIOS      = [1.5, 3]            # var_v0 = (mu_v0 / ratio)^2
GRID_ACC_FROM_V0_RATIOS = [5, 10, 20]          # mu_a0 = mu_v0 / ratio (sign preserved)
GRID_VAR_A0_RATIOS      = [3]                  # var_a0 = (mu_a0 / ratio)^2
GRID_SIGMA_W            = [1e-5, 5e-5]         # Process noise raw values


# =============================================================================
# 2. User Specified Functions (Kalman Filter)
# =============================================================================

def bridge_kinematic_transition(x, v, a, clip_x, dt):
    """Kinematic transition function for bridge degradation."""
    x = x + jnp.clip(v, -10000.0, 0.0) * dt + 0.5 * jnp.clip(a, -10000.0, 0.0) * dt**2
    v = v + jnp.clip(a, -10000.0, 0.0) * dt
    a = a
    clip_x = jnp.clip(x, 0, 1)
    return x, v, a, clip_x


def bridge_condition_observation(x):
    """Observation function for bridge condition."""
    y = jnp.clip(x, 0, 1)
    return y

def bridge_kinematic_transition_d(x, v, a, clip_x, dt):
    """Kinematic transition function for bridge degradation."""
    x = x + jnp.clip(v, 0.0, 10000.0) * dt + 0.5 * jnp.clip(a, 0.0, 10000.0) * dt**2
    v = v + jnp.clip(a, 0.0, 10000.0) * dt
    a = a
    clip_x = jnp.clip(x, 0, 1)
    return x, v, a, clip_x


def bridge_condition_observation_d(clip_x):
    """Observation function for bridge condition."""
    return clip_x


# =============================================================================
# 3. Synthetic Data Loader
# =============================================================================

class SyntheticDataLoader:
    """
    Loads and prepares synthetic bridge data for Kalman filtering.

    Converts the obs_dict/true_dict format from synthetic_bridge_data_v2.py
    into the batch format expected by the Kalman filter pipeline.
    """

    def __init__(self, config: Dict, init_conditions: Dict):
        self.config = config
        self.init_conditions = init_conditions
        self.obs_dict = None
        self.true_dict = None
        self.inspector_true_sigmas = None  # Ground truth sigmas per inspector

    def generate_data(self):
        """Generate synthetic data and store ground truth."""
        print("=" * 70)
        print("GENERATING SYNTHETIC DATA")
        print("=" * 70)

        self.obs_dict, self.true_dict = generate_bridge_network_data(
            num_bridges=self.config['num_bridges'],
            categories=self.config['categories'],
            num_elements_per_category=self.config['num_elements_per_category'],
            num_inspector_ids=self.config['num_inspector_ids'],
            inspector_sigma_range=self.config['inspector_sigma_range'],
            init_conditions=self.init_conditions,
            sigma_w=self.config['sigma_w'],
            year_start=self.config['year_start'],
            year_end=self.config['year_end'],
            min_obs=self.config['min_obs'],
            max_obs=self.config['max_obs'],
            gap_range=self.config['gap_range'],
            seed=self.config['seed']
        )

        # Extract ground truth inspector sigmas
        self._extract_inspector_ground_truth()

        N_obs = len(self.obs_dict['years'])
        N_series = len(self.true_dict['metadata']['bridge_ids'])
        print(f"Generated {N_obs} observations from {N_series} element series")
        print(f"Number of inspectors: {self.config['num_inspector_ids']}")
        print(f"Inspector sigma range: {self.config['inspector_sigma_range']}")

        return self.obs_dict, self.true_dict

    def _extract_inspector_ground_truth(self):
        """Extract ground truth sigma for each inspector from observations."""
        # obs_dict['inspector_ids'] and obs_dict['sigmas'] give us the mapping
        inspector_ids = self.obs_dict['inspector_ids']
        sigmas = np.array(self.obs_dict['sigmas'])

        # Build mapping: inspector_id -> sigma (all observations from same inspector have same sigma)
        self.inspector_true_sigmas = {}
        for insp_id, sigma in zip(inspector_ids, sigmas):
            if insp_id not in self.inspector_true_sigmas:
                self.inspector_true_sigmas[insp_id] = float(sigma)

        print(f"Extracted ground truth sigmas for {len(self.inspector_true_sigmas)} inspectors")

    def prepare_gate_data(self, gate: str) -> Dict[str, Any]:
        """
        Prepare batch data for a specific gate ('ab' or 'a').

        Note on gate definitions from synthetic_bridge_data_v2.py:
        - g_ab_obs = y_AB (direct AB observation)
        - g_a_obs = y_A / y_AB (derived - can be >1 or <0)
        - g_d_obs = y_D / (1 - y_AB) (derived)

        For AB learning, we use direct observations.
        For A learning, observations are derived and may need clipping.

        Args:
            gate: 'ab' for gate AB (index 1) or 'a' for gate A (index 0)

        Returns:
            Dictionary with Y, R, inspector_ids, years, true_values, etc.
        """
        print(f"\nPreparing data for gate '{gate}'...")

        gate_idx = {'a': 0, 'ab': 1, 'd': 2}[gate]
        gate_labels = self.obs_dict['gate_labels']

        # Get all unique (bridge_id, element_id) combinations to form series
        bridge_ids = self.obs_dict['bridge_ids']
        element_ids = self.obs_dict['element_ids']
        years = np.array(self.obs_dict['years'])
        g_obs = np.array(self.obs_dict['g'])  # (N_obs, G)
        inspector_ids = self.obs_dict['inspector_ids']
        sigmas = np.array(self.obs_dict['sigmas'])
        grades = np.array(self.obs_dict['grades']) # (N_obs, 4)

        if gate == 'a':
            # y_A / y_AB  — conditioned on S = y_AB (index 1)
            g_obs_gate = g_obs[:, gate_idx]
            g_obs_cond = g_obs[:, 1]              # S = y_AB
            print(f"  Note: Gate 'a' uses raw observations + Gate AB for conditioning (S = y_AB)")
        elif gate == 'd':
            # y_D / y_CD  — conditioned on S = y_CD = 1 - y_AB (index 1)
            g_obs_gate = g_obs[:, gate_idx]
            g_obs_cond = 1.0 - g_obs[:, 1]       # S = y_CD = 1 - y_AB
            print(f"  Note: Gate 'd' uses raw observations + (1 - y_AB) for conditioning (S = y_CD)")
        else:
            g_obs_gate = g_obs[:, gate_idx]
            g_obs_cond = None

        # Group observations by series
        series_keys = list(zip(bridge_ids, element_ids))
        unique_series = sorted(set(series_keys))

        # Get true states: (N_series, G, 3, T)
        true_states = np.array(self.true_dict['states'])
        true_times = np.array(self.true_dict['times'])
        meta_bridges = self.true_dict['metadata']['bridge_ids']
        meta_elements = self.true_dict['metadata']['element_ids']

        # Build series -> index mapping for true_dict
        series_to_idx = {(b, e): i for i, (b, e) in enumerate(zip(meta_bridges, meta_elements))}

        # Determine max duration
        min_year = int(true_times[0])
        max_year = int(true_times[-1])
        T = max_year - min_year + 1

        N_series = len(unique_series)

        # First pass: determine max series length starting from first observation
        # This ensures each series starts at its first observation, not at global t=0
        max_series_len = 0
        series_starts = []  # First observation index for each series

        for series_idx, (b_id, e_id) in enumerate(unique_series):
            mask = (bridge_ids == b_id) & (element_ids == e_id)
            obs_indices = np.where(mask)[0]
            if len(obs_indices) > 0:
                first_year = int(np.min(years[obs_indices]))
                last_year_global = max_year  # To end of simulation
                series_len = last_year_global - first_year + 1
                max_series_len = max(max_series_len, series_len)
                series_starts.append(first_year)
            else:
                series_starts.append(min_year)

        T_aligned = max_series_len  # New aligned length

        # Initialize batch arrays - ALIGNED to start at first observation
        batch_Y = np.full((N_series, T_aligned, 1), np.nan)
        batch_Y_cond = np.full((N_series, T_aligned, 1), np.nan)  # Conditional observations
        batch_grades = np.full((N_series, T_aligned, 4), np.nan)  # Grade observations
        batch_R = np.full((N_series, T_aligned, 1, 1), np.nan)
        batch_inspector_ids = [['' for _ in range(T_aligned)] for _ in range(N_series)]
        batch_years = np.zeros((N_series, T_aligned))
        batch_true_at_first = np.zeros(N_series)  # True gate value at first observation
        batch_first_year = np.zeros(N_series, dtype=int)
        batch_last_obs_year = np.zeros(N_series, dtype=int)
        batch_first_obs_idx = np.zeros(N_series, dtype=int)  # Index of first obs in original time array

        # TRUE initial states for each series (c0, v0, a0) at first observation time
        batch_true_c0 = np.zeros(N_series)  # True position at first obs
        batch_true_v0 = np.zeros(N_series)  # True velocity at first obs
        batch_true_a0 = np.zeros(N_series)  # True acceleration at first obs

        valid_indices = []

        for series_idx, (b_id, e_id) in enumerate(unique_series):
            # Find all observations for this series
            mask = (bridge_ids == b_id) & (element_ids == e_id)
            obs_indices = np.where(mask)[0]

            if len(obs_indices) == 0:
                continue

            series_years_raw = years[obs_indices]
            series_g = g_obs_gate[obs_indices]  # Use potentially clipped values
            series_inspectors = inspector_ids[obs_indices]
            series_sigmas = sigmas[obs_indices]
            series_grades = grades[obs_indices]

            if g_obs_cond is not None:
                series_g_cond = g_obs_cond[obs_indices]
            else:
                series_g_cond = None

            # Sort by year
            sort_order = np.argsort(series_years_raw)
            series_years_raw = series_years_raw[sort_order]
            series_g = series_g[sort_order]
            series_inspectors = series_inspectors[sort_order]
            series_sigmas = series_sigmas[sort_order]
            series_grades = series_grades[sort_order]
            if series_g_cond is not None:
                series_g_cond = series_g_cond[sort_order]

            # First observation year for this series
            first_year = int(series_years_raw[0])
            batch_first_year[series_idx] = first_year
            batch_first_obs_idx[series_idx] = first_year - min_year  # Offset in original time array
            batch_last_obs_year[series_idx] = int(series_years_raw[-1])

            # Fill batch arrays - TIME INDEX IS RELATIVE TO FIRST OBSERVATION
            for i, (yr, g_val, insp, sig, grd) in enumerate(zip(series_years_raw, series_g, series_inspectors, series_sigmas, series_grades)):
                t_idx = int(yr) - first_year  # Relative to first observation!
                if 0 <= t_idx < T_aligned:
                    batch_Y[series_idx, t_idx, 0] = g_val
                    batch_grades[series_idx, t_idx, :] = grd
                    # Fill conditional observation if available
                    if series_g_cond is not None:
                        batch_Y_cond[series_idx, t_idx, 0] = series_g_cond[i]

                    # R will be provided by InspectorManager via get_R_batch - not pre-filled
                    batch_inspector_ids[series_idx][t_idx] = insp

            # Years array for this series - starts at first observation
            batch_years[series_idx] = np.arange(first_year, first_year + T_aligned)

            # Get TRUE state values at first observation time
            # true_states shape: (N_series, G, 3, T) where 3 = [position, velocity, acceleration]
            true_series_idx = series_to_idx[(b_id, e_id)]
            first_t_idx_global = first_year - min_year  # Index in global true_states array

            # Extract true c0, v0, a0 at first observation time
            true_c0 = float(true_states[true_series_idx, gate_idx, 0, first_t_idx_global])  # position
            true_v0 = float(true_states[true_series_idx, gate_idx, 1, first_t_idx_global])  # velocity
            true_a0 = float(true_states[true_series_idx, gate_idx, 2, first_t_idx_global])  # acceleration

            batch_true_c0[series_idx] = true_c0
            batch_true_v0[series_idx] = true_v0
            batch_true_a0[series_idx] = true_a0
            batch_true_at_first[series_idx] = true_c0  # For compatibility
            valid_indices.append((b_id, e_id))

        # Store true trajectories for plotting - ALIGNED to start at first obs
        true_trajectories = np.full((N_series, T_aligned), np.nan)
        for series_idx, (b_id, e_id) in enumerate(unique_series):
            true_series_idx = series_to_idx[(b_id, e_id)]
            first_t_idx = batch_first_obs_idx[series_idx]
            # Slice from first observation to end
            remaining = T - first_t_idx
            len_to_copy = min(remaining, T_aligned)
            true_trajectories[series_idx, :len_to_copy] = true_states[true_series_idx, gate_idx, 0, first_t_idx:first_t_idx+len_to_copy]

        # Also store the true speed
        true_speeds = np.full((N_series, T_aligned), np.nan)
        for series_idx, (b_id, e_id) in enumerate(unique_series):
            true_series_idx = series_to_idx[(b_id, e_id)]
            first_t_idx = batch_first_obs_idx[series_idx]
            remaining = T - first_t_idx
            len_to_copy = min(remaining, T_aligned)
            true_speeds[series_idx, :len_to_copy] = true_states[true_series_idx, gate_idx, 1, first_t_idx:first_t_idx+len_to_copy]

        # Store true grades (A, B, C, D) - ALIGNED to start at first obs
        # true_dict['grades'] shape: (N_series, 4, T_global)
        true_grades_all = np.array(self.true_dict['grades'])  # (N, 4, T)
        batch_true_grades = np.full((N_series, T_aligned, 4), np.nan)
        for series_idx, (b_id, e_id) in enumerate(unique_series):
            true_series_idx = series_to_idx[(b_id, e_id)]
            first_t_idx = batch_first_obs_idx[series_idx]
            remaining = T - first_t_idx
            len_to_copy = min(remaining, T_aligned)
            # true_grades_all is (N, 4, T) -> transpose to (4, T) then slice
            batch_true_grades[series_idx, :len_to_copy, :] = true_grades_all[true_series_idx, :, first_t_idx:first_t_idx+len_to_copy].T

        print(f"  Prepared {N_series} series, T_aligned={T_aligned} timesteps (from first obs)")
        print(f"  First obs year range: [{np.min(batch_first_year):.0f}, {np.max(batch_first_year):.0f}]")
        print(f"  Observation range: [{np.nanmin(batch_Y):.4f}, {np.nanmax(batch_Y):.4f}]")
        print(f"  Mean true value at first obs: {np.mean(batch_true_at_first):.4f}")

        return {
            'Y': jnp.array(batch_Y),
            'Y_cond': jnp.array(batch_Y_cond),  # Conditional observations
            'grades': jnp.array(batch_grades),  # Grade observations (N, T, 4)
            'true_grades': jnp.array(batch_true_grades),  # True grades (N, T, 4)
            'R': jnp.array(batch_R),
            'inspector_ids': batch_inspector_ids,
            'years': batch_years,
            'true_at_first': jnp.array(batch_true_at_first),
            'true_c0': jnp.array(batch_true_c0),  # True position at first obs
            'true_v0': jnp.array(batch_true_v0),  # True velocity at first obs
            'true_a0': jnp.array(batch_true_a0),  # True acceleration at first obs
            'first_years': jnp.array(batch_first_year),
            'last_obs_years': batch_last_obs_year,
            'first_obs_idx': batch_first_obs_idx,  # New: global index of first obs
            'indices': valid_indices,
            'true_trajectories': jnp.array(true_trajectories),
            'true_speeds': jnp.array(true_speeds),
            'gate_name': gate,
            'T': T_aligned,  # Aligned length
            'T_global': T,   # Original global length
            'N': N_series,
            'year_start': min_year
        }


    def get_all_inspector_ids(self) -> List[str]:
        """Get sorted list of all unique inspector IDs."""
        return sorted(set(self.obs_dict['inspector_ids']))

    def get_ground_truth_sigmas(self) -> Dict[str, float]:
        """Get ground truth sigma for each inspector."""
        return self.inspector_true_sigmas


# =============================================================================
# 4. Helper Functions
# =============================================================================

def compute_y0_distribution(first_years: jnp.ndarray) -> Dict[int, float]:
    """Compute distribution of first observation years for weighting."""
    first_years_np = np.array(first_years)
    unique_years, counts = np.unique(first_years_np, return_counts=True)
    total = counts.sum()
    return {int(year): count / total for year, count in zip(unique_years, counts)}


@jax.jit
def get_first_observation(Y: jnp.ndarray) -> jnp.ndarray:
    """Get the first valid observation for each series. Y: (N, T, 1) -> (N,)"""
    mask = ~jnp.isnan(Y[:, :, 0])          # (N, T)
    idx_first = jnp.argmax(mask, axis=1)   # (N,)
    ranges = jnp.arange(Y.shape[0])
    return Y[ranges, idx_first, 0]


def create_initial_state_from_observation(
    Y_batch: jnp.ndarray,
    R_batch: jnp.ndarray,
    mu_v0: float,
    var_v0: float,
    mu_a0: float,
    var_a0: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Create initial state mean/covariance using first observation (grid-search version).

    - c0 = first valid observation,  var_c0 = R at that step (learned inspector var)
    - v0 = mu_v0 (grid param, broadcast across batch)
    - a0 = mu_a0 (grid param, broadcast across batch)
    """
    N = Y_batch.shape[0]
    y0 = get_first_observation(Y_batch)        # (N,)

    mask = ~jnp.isnan(Y_batch[:, :, 0])
    idx_first = jnp.argmax(mask, axis=1)       # (N,)
    R_squeezed = R_batch[:, :, 0, 0]           # (N, T)
    var_c0 = R_squeezed[jnp.arange(N), idx_first]

    m0 = jnp.zeros((N, 3))
    m0 = m0.at[:, 0].set(y0)
    m0 = m0.at[:, 1].set(mu_v0)
    m0 = m0.at[:, 2].set(mu_a0)

    P0 = jnp.zeros((N, 3, 3))
    P0 = P0.at[:, 0, 0].set(var_c0)
    P0 = P0.at[:, 1, 1].set(var_v0)
    P0 = P0.at[:, 2, 2].set(var_a0)

    return m0, P0

# =============================================================================
# 5. Training and Evaluation Functions
# =============================================================================

def train_inspector_variances(
    Y_train: jnp.ndarray,
    Y_cond_train: jnp.ndarray,
    inspector_ids_train: List[List[str]],
    mu_v0: float,
    var_v0: float,
    mu_a0: float,
    var_a0: float,
    manager: InspectorManager,
    process_noise: float,
    transition_fn=None,
    observation_fn=None,
) -> InspectorManager:
    """
    Train inspector variances using AGVI on training data (Single Pass).

    Args:
        Y_train:      (N, T, 1) Gate-A observations
        Y_cond_train: (N, T, 1) y_AB ratio denominator S (None for Stage 1 / AB gate)
        transition_fn: SSM transition function (default: bridge_kinematic_transition)
        observation_fn: SSM observation function (default: bridge_condition_observation)
    """
    if transition_fn is None:
        transition_fn = bridge_kinematic_transition
    if observation_fn is None:
        observation_fn = bridge_condition_observation
    ssm = SSMparam(
        transition=[transition_fn],
        observation=[observation_fn],
        dt_val=1.0
    )
    kf = KalmanFilter(ssm)

    @jax.jit
    def run_filter_batch_learning(obs_batch, R_prior_batch, m0_b, P0_b):
        def single_f(o, r, m, p):
            return kf.filter(o, r, learn_R=True, process_error=process_noise,
                             initial_mean=m, initial_covariance=p)
        return jax.vmap(single_f)(obs_batch, R_prior_batch, m0_b, P0_b)

    @jax.jit
    def run_filter_conditional_batch_learning(obs_batch, R_prior_batch, cond_obs_batch, m0_b, P0_b):
        def single_f(o, r, c_obs, m, p):
            return kf.filter_conditional(
                o, r, c_obs,
                learn_R=True, process_error=process_noise,
                initial_mean=m, initial_covariance=p
            )
        return jax.vmap(single_f)(obs_batch, R_prior_batch, cond_obs_batch, m0_b, P0_b)

    N_train = Y_train.shape[0]
    T_dim = Y_train.shape[1]
    BATCH_SIZE = 100
    num_batches = int(np.ceil(N_train / BATCH_SIZE))

    print(f"  Training on {N_train} series (Single Pass), {num_batches} batches...")

    total_updates = 0

    for i in range(num_batches):
        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, N_train)

        # Batch slicing
        Y_batch = Y_train[start_idx:end_idx]
        current_inspector_ids = inspector_ids_train[start_idx:end_idx]

        # Get R prior from manager
        R_prior_batch = manager.get_R_batch(current_inspector_ids, T_dim)

        # Initial state from first observation + grid parameters
        m0_slice, P0_slice = create_initial_state_from_observation(
            Y_batch, R_prior_batch, mu_v0, var_v0, mu_a0, var_a0
        )

        use_conditional = Y_cond_train is not None

        if use_conditional:
            Y_cond_batch = Y_cond_train[start_idx:end_idx]

            # Use separate Gate AB observation for conditioning
            results = run_filter_conditional_batch_learning(
                Y_batch, R_prior_batch, Y_cond_batch, m0_slice, P0_slice
            )
        else:
            # Standard filter
            results = run_filter_batch_learning(
                Y_batch, R_prior_batch, m0_slice, P0_slice
            )

        mr_post_batch = results[4]
        pr_post_batch = results[5]

        # --- Adaptive boundary mask: exclude observations within 1σ of 0 or 1 ---
        # The valid range is [σ_R, 1 − σ_R] where σ_R is the inspector's learned sigma.
        # R_prior_batch shape: (B, T, 1, 1) → sigma_batch shape: (B, T)
        sigma_batch = jnp.sqrt(jnp.maximum(R_prior_batch[..., 0, 0], 1e-10))
        obs_vals = Y_batch[..., 0]
        # extreme_mask = (obs_vals < sigma_batch) | (obs_vals > (1.0 - sigma_batch))
        extreme_mask = (obs_vals < 0.025) | (obs_vals > (1.0 - 0.025))
        # Apply mask to posteriors (set to NaN to ignore in AGVI)
        # mr_post_batch shape: (B, T, 1)
        # pr_post_batch shape: (B, T, 1, 1) or (B, T, 1)
        mr_post_batch = jnp.where(extreme_mask[..., None], jnp.nan, mr_post_batch)
        if pr_post_batch.ndim == 4:
            pr_post_batch = jnp.where(extreme_mask[..., None, None], jnp.nan, pr_post_batch)
        else:
            pr_post_batch = jnp.where(extreme_mask[..., None], jnp.nan, pr_post_batch)

        # Update inspector manager
        valid_mask_batch = ~jnp.isnan(Y_batch[..., 0])
        # Also include our adaptive extreme mask in the validity check for counting updates
        valid_mask_batch = valid_mask_batch & (~extreme_mask)

        manager.update_batch_from_results(mr_post_batch, pr_post_batch, current_inspector_ids, valid_mask_batch)

        total_updates += int(valid_mask_batch.sum())

    # Report final stats
    learned_sigmas = np.sqrt(np.array(manager.state.means[:, 0]))
    active_mask = manager.state.update_counts > 0
    if active_mask.any():
        mean_sigma = np.mean(learned_sigmas[active_mask])
        print(f"  Training complete: {total_updates} updates, mean_sigma={mean_sigma:.4f}")

    return manager

def evaluate_model(
    Y_eval: jnp.ndarray,
    Y_cond_eval: jnp.ndarray,
    inspector_ids_eval: List[List[str]],
    mu_v0: float,
    var_v0: float,
    mu_a0: float,
    var_a0: float,
    first_years_eval: jnp.ndarray,
    y0_distribution: Dict[int, float],
    manager: InspectorManager,
    process_noise: float,
    transition_fn=None,
    observation_fn=None,
) -> Tuple[float, float, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, list]:
    """
    Evaluate model on validation/test data (Standard or Conditional).

    Returns
    -------
    regular_ll, weighted_ll, m_smooth, P_smooth, m_filt, P_filt, per_series_wll
        ``per_series_wll`` is a list of weighted log-likelihoods, one per
        test/val series in order (used for worst-series ranking).
    """
    if transition_fn is None:
        transition_fn = bridge_kinematic_transition
    if observation_fn is None:
        observation_fn = bridge_condition_observation
    ssm = SSMparam(
        transition=[transition_fn],
        observation=[observation_fn],
        dt_val=1.0
    )
    kf = KalmanFilter(ssm)

    @jax.jit
    def run_filter_batch_eval(obs_batch, R_prior_batch, m0_b, P0_b):
        def single_f(o, r, m, p):
            return kf.filter(o, r, learn_R=False, process_error=process_noise,
                             initial_mean=m, initial_covariance=p)
        return jax.vmap(single_f)(obs_batch, R_prior_batch, m0_b, P0_b)

    @jax.jit
    def run_filter_conditional_batch_eval(
            obs_batch, R_prior_batch, cond_obs_batch, m0_b, P0_b):
        def single_f(o, r, c_obs, m, p):
            return kf.filter_conditional(
                o, r, c_obs,
                learn_R=False, process_error=process_noise,
                initial_mean=m, initial_covariance=p
            )
        return jax.vmap(single_f)(obs_batch, R_prior_batch, cond_obs_batch, m0_b, P0_b)

    smooth_vmap = jax.vmap(kf.smooth, in_axes=(0, 0, 0, 0, None))

    N_eval = Y_eval.shape[0]
    T_dim = Y_eval.shape[1]
    BATCH_SIZE = 50
    num_batches = int(np.ceil(N_eval / BATCH_SIZE))

    regular_ll   = 0.0
    weighted_ll  = 0.0
    per_series_wll: list = []
    all_m_smooth = []
    all_P_smooth = []
    all_m_filt   = []
    all_P_filt   = []
    all_m_pred   = []
    all_P_pred   = []

    for i in range(num_batches):
        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, N_eval)

        Y_batch = Y_eval[start_idx:end_idx]
        current_inspector_ids = inspector_ids_eval[start_idx:end_idx]
        first_years_batch = first_years_eval[start_idx:end_idx]

        # Get R prior: (B, T, 1, 1)
        R_prior_batch = manager.get_R_batch(current_inspector_ids, T_dim)

        # Initial state from first observation + grid parameters
        m0_slice, P0_slice = create_initial_state_from_observation(
            Y_batch, R_prior_batch, mu_v0, var_v0, mu_a0, var_a0
        )

        use_conditional = Y_cond_eval is not None

        if use_conditional:
            Y_cond_batch = Y_cond_eval[start_idx:end_idx]

            results = run_filter_conditional_batch_eval(
                Y_batch, R_prior_batch, Y_cond_batch, m0_slice, P0_slice
            )
        else:
            results = run_filter_batch_eval(
                Y_batch, R_prior_batch, m0_slice, P0_slice
            )

        b_m_filt, b_P_filt, b_m_pred, b_P_pred, _, _, ll_batch = results

        # Smooth
        b_m_smooth, b_P_smooth = smooth_vmap(b_m_filt, b_P_filt, b_m_pred, b_P_pred, None)

        all_m_smooth.append(b_m_smooth)
        all_P_smooth.append(b_P_smooth)
        all_m_filt.append(b_m_filt)
        all_P_filt.append(b_P_filt)
        all_m_pred.append(b_m_pred)
        all_P_pred.append(b_P_pred)

        # Compute log-likelihood (regular + weighted, tracked per series)
        for ll, first_year in zip(ll_batch, first_years_batch):
            ll_value   = float(ll)
            repartition = y0_distribution.get(int(first_year), 1.0)
            wll        = ll_value * (1.0 / repartition)
            regular_ll += ll_value
            weighted_ll += wll
            per_series_wll.append(wll)

    m_smooth = jnp.concatenate(all_m_smooth, axis=0)
    P_smooth = jnp.concatenate(all_P_smooth, axis=0)
    m_filt   = jnp.concatenate(all_m_filt,   axis=0)
    P_filt   = jnp.concatenate(all_P_filt,   axis=0)
    m_pred   = jnp.concatenate(all_m_pred,   axis=0)
    P_pred   = jnp.concatenate(all_P_pred,   axis=0)

    return regular_ll, weighted_ll, m_smooth, P_smooth, m_filt, P_filt, m_pred, P_pred, per_series_wll

# =============================================================================
# 6. Analysis Helpers
# =============================================================================

# =============================================================================
# 6.5  Hidden-Last-Observation Forecast Helpers (vs Ground Truth)
# =============================================================================

def extract_hidden_truth_forecast_pairs(
    data: Dict,
    m_pred: jnp.ndarray,
    P_pred: jnp.ndarray,
    true_traj: np.ndarray,
    Y_cond: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    For each series:
      - Find last valid observation time t_last and second-to-last t_prev
      - Δt = t_last - t_prev
      - pred_mean = clip(m_pred[k, t_last+1, 0], 0, 1)  ← prediction BEFORE update
      - pred_std  = sqrt(P_pred[k, t_last+1, 0, 0])
      - ref_val   = true_traj[k, t_last]   (ground truth, no noise)
      - ref_std   = 0.0
      - S_val     = Y_cond[k, t_last, 0]   (conditional denominator for gate A/D)

    Mirroring extract_hidden_obs_forecast_pairs from the real-data pipeline,
    but using the ground truth instead of the noisy observation as reference.
    """
    pred_means = []; pred_stds = []
    ref_vals   = []; ref_stds  = []
    delta_ts   = []
    S_vals     = []

    Y       = np.array(data['Y'])           # (N, T, 1)
    true_np = np.array(true_traj)           # (N, T)
    Yc      = np.array(Y_cond) if Y_cond is not None else None

    N = Y.shape[0]
    for k in range(N):
        obs = Y[k, :, 0]
        valid_t = np.where(~np.isnan(obs))[0]
        if len(valid_t) < 2:
            continue

        t_last = int(valid_t[-1])
        t_prev = int(valid_t[-2])
        dt     = t_last - t_prev

        # Ground truth at t_last
        if t_last >= true_np.shape[1]:
            continue
        y_true = float(np.clip(true_np[k, t_last], 0.0, 1.0))
        if np.isnan(y_true):
            continue

        # Filter prediction BEFORE update at t_last
        pred_t = min(t_last + 1, m_pred.shape[1] - 1)
        pm     = float(np.clip(float(m_pred[k, pred_t, 0]), 0.0, 1.0))
        pv     = float(P_pred[k, pred_t, 0, 0])
        ps     = float(np.sqrt(max(pv, 0.0)))

        # Conditioning denominator S at t_last (for gate A: y_AB, gate D: 1-y_AB)
        if Yc is not None and t_last < Yc.shape[1]:
            S = max(float(Yc[k, t_last, 0]), 0.0)
        else:
            S = 1.0

        pred_means.append(pm)
        pred_stds.append(ps)
        ref_vals.append(y_true)
        ref_stds.append(0.0)   # truth has no observation noise
        delta_ts.append(dt)
        S_vals.append(S)

    return {
        'pred_mean': np.array(pred_means),
        'pred_std':  np.array(pred_stds),
        'ref_val':   np.array(ref_vals),
        'ref_std':   np.array(ref_stds),
        'delta_t':   np.array(delta_ts),
        'S_val':     np.array(S_vals),
    }


def extract_hidden_grade_truth_forecast_pairs(
    data_ab: Dict,
    mp_ab: jnp.ndarray,
    mp_a:  jnp.ndarray,
    mp_d:  jnp.ndarray,
    Pp_ab: jnp.ndarray,
    Pp_a:  jnp.ndarray,
    Pp_d:  jnp.ndarray,
    all_idx: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Hidden-last-observation grade forecast vs true grades.

    Uses m_pred (filter prediction BEFORE update at t_last) to compute
    predicted grades via gaussian_product, then compares vs true grades.
    ref_std = 0 since the truth has no observation noise.
    """
    pred_means_all = {g: [] for g in ['A', 'B', 'C', 'D']}
    pred_stds_all  = {g: [] for g in ['A', 'B', 'C', 'D']}
    ref_vals_all   = {g: [] for g in ['A', 'B', 'C', 'D']}
    ref_stds_all   = {g: [] for g in ['A', 'B', 'C', 'D']}
    delta_ts_all   = []

    T_pred = mp_ab.shape[1]                           # T + 1 (includes prepended t=0)
    true_grades_np = np.array(data_ab['true_grades'])  # (N, T_aligned, 4)
    Y_ab_np = np.array(data_ab['Y'])                  # (N, T, 1)

    for local_k, global_k in enumerate(all_idx):
        obs = Y_ab_np[global_k, :, 0]
        valid_t = np.where(~np.isnan(obs))[0]
        if len(valid_t) < 2:
            continue

        t_last = int(valid_t[-1])
        t_prev = int(valid_t[-2])
        dt     = t_last - t_prev

        # Gate predictions BEFORE update at t_last
        pred_t   = min(t_last + 1, T_pred - 1)
        mu_ab_t  = float(np.clip(float(mp_ab[local_k, pred_t, 0]), 0.0, 1.0))
        var_ab_t = max(float(Pp_ab[local_k, pred_t, 0, 0]), 0.0)
        mu_a_t   = float(np.clip(float(mp_a[local_k,  pred_t, 0]), 0.0, 1.0))
        var_a_t  = max(float(Pp_a[local_k,  pred_t, 0, 0]), 0.0)
        mu_d_t   = float(np.clip(float(mp_d[local_k,  pred_t, 0]), 0.0, 1.0))
        var_d_t  = max(float(Pp_d[local_k,  pred_t, 0, 0]), 0.0)

        mu_A_t, var_A_t = gaussian_product(mu_a_t,       var_a_t,  mu_ab_t,     var_ab_t)
        mu_B_t, var_B_t = gaussian_product(1.0-mu_a_t,   var_a_t,  mu_ab_t,     var_ab_t)
        mu_C_t, var_C_t = gaussian_product(1.0-mu_d_t,   var_d_t,  1.0-mu_ab_t, var_ab_t)
        mu_D_t, var_D_t = gaussian_product(mu_d_t,       var_d_t,  1.0-mu_ab_t, var_ab_t)

        grade_preds = {
            'A': (float(mu_A_t), float(np.sqrt(max(float(var_A_t), 0.0)))),
            'B': (float(mu_B_t), float(np.sqrt(max(float(var_B_t), 0.0)))),
            'C': (float(mu_C_t), float(np.sqrt(max(float(var_C_t), 0.0)))),
            'D': (float(mu_D_t), float(np.sqrt(max(float(var_D_t), 0.0)))),
        }
        if not all(np.isfinite(grade_preds[g][0]) for g in ['A', 'B', 'C', 'D']):
            continue

        # True grades at t_last
        if t_last >= true_grades_np.shape[1]:
            continue
        true_g = true_grades_np[global_k, t_last, :]  # (4,) = [A, B, C, D]

        for i, g in enumerate(['A', 'B', 'C', 'D']):
            pm, ps = grade_preds[g]
            pred_means_all[g].append(pm)
            pred_stds_all[g].append(ps)
            ref_vals_all[g].append(float(np.clip(true_g[i], 0.0, 1.0)))
            ref_stds_all[g].append(0.0)   # truth has no noise
        delta_ts_all.append(dt)

    return {
        'pred_mean': {g: np.array(pred_means_all[g]) for g in ['A', 'B', 'C', 'D']},
        'pred_std':  {g: np.array(pred_stds_all[g])  for g in ['A', 'B', 'C', 'D']},
        'ref_val':   {g: np.array(ref_vals_all[g])   for g in ['A', 'B', 'C', 'D']},
        'ref_std':   {g: np.array(ref_stds_all[g])   for g in ['A', 'B', 'C', 'D']},
        'delta_t':   np.array(delta_ts_all),
    }


# =============================================================================
# 7. Main Pipeline
# =============================================================================

def run_pipeline(
    mu_v0_ab, var_v0_ab, mu_a0_ab, var_a0_ab, sigma_w_ab,
    mu_v0_a,  var_v0_a,  mu_a0_a,  var_a0_a,  sigma_w_a,
    mu_v0_d,  var_v0_d,  mu_a0_d,  var_a0_d,  sigma_w_d,
):
    """Three-gate learning pipeline followed by a printed numerical summary."""
    # ── Data ────────────────────────────────────────────────────────────────
    loader = SyntheticDataLoader(CONFIG, INIT_CONDITIONS)
    loader.generate_data()
    all_inspector_ids   = loader.get_all_inspector_ids()
    ground_truth_sigmas = loader.get_ground_truth_sigmas()

    data_ab = loader.prepare_gate_data('ab')
    data_a  = loader.prepare_gate_data('a')
    data_d  = loader.prepare_gate_data('d')

    N = data_ab['N']
    np.random.seed(42)
    idx_perm  = np.random.permutation(N)
    train_end = int(N * TRAIN_RATIO)
    val_end   = int(N * (TRAIN_RATIO + VAL_RATIO))
    train_idx = idx_perm[:train_end]
    val_idx   = idx_perm[train_end:val_end]
    test_idx  = idx_perm[val_end:]
    print(f"Split → train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    y0_dist = compute_y0_distribution(data_ab['first_years'])

    # ── Stage 1: AB ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STAGE 1 — Gate x_AB  (standard AGVI)")
    print("=" * 70)

    Y_ab_train = data_ab['Y'][train_idx]
    Y_ab_test  = data_ab['Y'][test_idx]
    insp_ab_train = [data_ab['inspector_ids'][i] for i in train_idx]
    insp_ab_test  = [data_ab['inspector_ids'][i] for i in test_idx]
    fy_test        = data_ab['first_years'][test_idx]

    manager_ab = InspectorManager(all_inspector_ids, init_std=0.075, init_var=0.1**2)
    manager_ab = train_inspector_variances(
        Y_ab_train, None, insp_ab_train,
        mu_v0_ab, var_v0_ab, mu_a0_ab, var_a0_ab,
        manager_ab, sigma_w_ab,
    )
    (reg_ll_ab, wgt_ll_ab,
     ms_ab, Ps_ab, _, _, mp_ab_test, Pp_ab_test,
     wll_ab) = evaluate_model(
        Y_ab_test, None, insp_ab_test,
        mu_v0_ab, var_v0_ab, mu_a0_ab, var_a0_ab,
        fy_test, y0_dist, manager_ab, sigma_w_ab,
    )
    print(f"  AB  reg_ll={reg_ll_ab:.2f}  wgt_ll={wgt_ll_ab:.2f}")

    # ── Stage 2: A (conditional on y_AB) ────────────────────────────────────
    print("\n" + "=" * 70)
    print("STAGE 2 — Gate x_A|AB  (y_A conditioned on y_AB)")
    print("=" * 70)

    Y_a_train      = data_a['Y'][train_idx]
    Y_a_cond_train = data_a['Y_cond'][train_idx]
    Y_a_test       = data_a['Y'][test_idx]
    Y_a_cond_test  = data_a['Y_cond'][test_idx]
    insp_a_train   = [data_a['inspector_ids'][i] for i in train_idx]
    insp_a_test    = [data_a['inspector_ids'][i] for i in test_idx]

    manager_a = InspectorManager(all_inspector_ids, init_std=0.075, init_var=0.1**2)
    manager_a = train_inspector_variances(
        Y_a_train, Y_a_cond_train, insp_a_train,
        mu_v0_a, var_v0_a, mu_a0_a, var_a0_a,
        manager_a, sigma_w_a,
    )
    (reg_ll_a, wgt_ll_a,
     ms_a, Ps_a, _, _, mp_a_test, Pp_a_test,
     wll_a) = evaluate_model(
        Y_a_test, Y_a_cond_test, insp_a_test,
        mu_v0_a, var_v0_a, mu_a0_a, var_a0_a,
        fy_test, y0_dist, manager_a, sigma_w_a,
    )
    print(f"  A   reg_ll={reg_ll_a:.2f}  wgt_ll={wgt_ll_a:.2f}")

    # ── Stage 3: D (conditional on y_CD = 1 - y_AB) ─────────────────────────
    print("\n" + "=" * 70)
    print("STAGE 3 — Gate x_D|CD  (y_D conditioned on y_CD = 1 - y_AB)")
    print("=" * 70)

    Y_d_train      = data_d['Y'][train_idx]
    Y_d_cond_train = data_d['Y_cond'][train_idx]
    Y_d_test       = data_d['Y'][test_idx]
    Y_d_cond_test  = data_d['Y_cond'][test_idx]
    insp_d_train   = [data_d['inspector_ids'][i] for i in train_idx]
    insp_d_test    = [data_d['inspector_ids'][i] for i in test_idx]

    manager_d = InspectorManager(all_inspector_ids, init_std=0.075, init_var=0.1**2)
    manager_d = train_inspector_variances(
        Y_d_train, Y_d_cond_train, insp_d_train,
        mu_v0_d, var_v0_d, mu_a0_d, var_a0_d,
        manager_d, sigma_w_d,
        transition_fn=bridge_kinematic_transition_d,
        observation_fn=bridge_condition_observation_d,
    )
    (reg_ll_d, wgt_ll_d,
     ms_d, Ps_d, _, _, mp_d_test, Pp_d_test,
     wll_d) = evaluate_model(
        Y_d_test, Y_d_cond_test, insp_d_test,
        mu_v0_d, var_v0_d, mu_a0_d, var_a0_d,
        fy_test, y0_dist, manager_d, sigma_w_d,
        transition_fn=bridge_kinematic_transition_d,
        observation_fn=bridge_condition_observation_d,
    )
    print(f"  D   reg_ll={reg_ll_d:.2f}  wgt_ll={wgt_ll_d:.2f}")

    # ── Stage 4: Global grades via Gaussian product (test set) ───────────────
    print("\n" + "=" * 70)
    print("STAGE 4 — Global Grades (A, B, C, D via Gaussian Product)")
    print("=" * 70)

    mu_gate_ab_te  = ms_ab[:, 1:, 3]
    var_gate_ab_te = jnp.maximum(Ps_ab[:, 1:, 3, 3], 0.0)
    mu_gate_a_te   = ms_a[:,  1:, 3]
    var_gate_a_te  = jnp.maximum(Ps_a[:,  1:, 3, 3], 0.0)
    mu_gate_d_te   = ms_d[:,  1:, 3]
    var_gate_d_te  = jnp.maximum(Ps_d[:,  1:, 3, 3], 0.0)

    mu_A_te, var_A_te = gaussian_product(mu_gate_a_te, var_gate_a_te,
                                         mu_gate_ab_te, var_gate_ab_te)
    mu_B_te, var_B_te = gaussian_product(1.0 - mu_gate_a_te, var_gate_a_te,
                                         mu_gate_ab_te, var_gate_ab_te)
    mu_C_te, var_C_te = gaussian_product(1.0 - mu_gate_d_te, var_gate_d_te,
                                         1.0 - mu_gate_ab_te, var_gate_ab_te)
    mu_D_te, var_D_te = gaussian_product(mu_gate_d_te, var_gate_d_te,
                                         1.0 - mu_gate_ab_te, var_gate_ab_te)

    combined_test_ll = wgt_ll_ab + wgt_ll_a + wgt_ll_d
    print(f"  Test weighted LL → AB={wgt_ll_ab:.2f}  A={wgt_ll_a:.2f}  "
          f"D={wgt_ll_d:.2f}  total={combined_test_ll:.2f}")
    print(f"  Test grade means (last t): "
          f"A={float(jnp.nanmean(mu_A_te[:, -1])):.4f}  "
          f"B={float(jnp.nanmean(mu_B_te[:, -1])):.4f}  "
          f"C={float(jnp.nanmean(mu_C_te[:, -1])):.4f}  "
          f"D={float(jnp.nanmean(mu_D_te[:, -1])):.4f}")

    # ── Numerical Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 71)
    print("RESULTS SUMMARY")
    print("=" * 71)

    N_train = len(train_idx)
    N_val   = len(val_idx)
    N_test  = len(test_idx)

    print(f"\nDataset")
    print(f"  N series   : {N}   "
          f"({CONFIG['num_bridges']} bridges × "
          f"{CONFIG['num_elements_per_category']} elements)")
    print(f"  Inspectors : {CONFIG['num_inspector_ids']}")
    print(f"  Train/Val/Test split: {N_train} / {N_val} / {N_test}")

    true_s_arr = np.array([ground_truth_sigmas.get(uid, np.nan)
                            for uid in all_inspector_ids])

    for stage_label, mgr, reg_ll, wgt_ll in [
        ("Stage 1 — x_AB  (inter-group: P(A∪B))",       manager_ab, reg_ll_ab, wgt_ll_ab),
        ("Stage 2 — x_A|AB  (intra-group A: P(A | A∪B))", manager_a,  reg_ll_a,  wgt_ll_a),
        ("Stage 3 — x_D|CD  (intra-group D: P(D | C∪D))", manager_d,  reg_ll_d,  wgt_ll_d),
    ]:
        print(f"\n{stage_label}")
        print(f"  Test log-likelihood (regular):  {reg_ll:.2f}")
        print(f"  Test log-likelihood (weighted): {wgt_ll:.2f}")
        mask_m = mgr.state.update_counts > 0
        l_var  = np.array(mgr.state.means[:, 0])[mask_m]
        l_sig  = np.sqrt(np.maximum(l_var, 1e-10))
        t_sig  = true_s_arr[mask_m]
        valid  = np.isfinite(t_sig)
        print(f"  Inspector σ — learned: mean={l_sig.mean():.4f}  std={l_sig.std():.4f}")
        if valid.any():
            print(f"                ground truth: mean={t_sig[valid].mean():.4f}  "
                  f"std={t_sig[valid].std():.4f}")
            mae = float(np.mean(np.abs(l_sig[valid] - t_sig[valid])))
            print(f"                MAE vs truth: {mae:.4f}")

    mu_A_last = float(jnp.nanmean(mu_A_te[:, -1]))
    mu_B_last = float(jnp.nanmean(mu_B_te[:, -1]))
    mu_C_last = float(jnp.nanmean(mu_C_te[:, -1]))
    mu_D_last = float(jnp.nanmean(mu_D_te[:, -1]))
    print(f"\nStage 4 — Global grades  {{A, B, C, D}}  (Gaussian product)")
    print(f"  Mean grade proportions at last timestep (test set):")
    print(f"    A = {mu_A_last:.4f}  B = {mu_B_last:.4f}  "
          f"C = {mu_C_last:.4f}  D = {mu_D_last:.4f}")
    print(f"    Sum = {mu_A_last + mu_B_last + mu_C_last + mu_D_last:.4f}  (≈ 1.0)")

    # --- Forecast accuracy vs ground truth -----------------------------------
    CAL_HORIZONS = [1, 2, 3]

    def _rmse(pred, ref):
        return float(np.sqrt(np.mean((pred - ref) ** 2))) if len(pred) > 0 else float('nan')

    def _zstd(pred_mean, pred_std, ref):
        if len(pred_mean) == 0:
            return float('nan')
        z = (ref - pred_mean) / np.maximum(pred_std, 1e-10)
        return float(np.std(z))

    def _last_obs_rmse(Y, true_traj, dt_filter=None):
        """RMSE of last (hidden) observation vs ground truth — naive baseline."""
        Y_np   = np.array(Y)
        tr_np  = np.array(true_traj)
        sq_err = []
        for k in range(Y_np.shape[0]):
            obs    = Y_np[k, :, 0]
            vt     = np.where(~np.isnan(obs))[0]
            if len(vt) < 2:
                continue
            t_last = int(vt[-1])
            dt     = t_last - int(vt[-2])
            if dt_filter is not None and dt != dt_filter:
                continue
            if t_last >= tr_np.shape[1]:
                continue
            y_true = float(np.clip(tr_np[k, t_last], 0.0, 1.0))
            if np.isnan(y_true):
                continue
            y_obs  = float(np.clip(obs[t_last], 0.0, 1.0))
            if np.isnan(y_obs):
                continue
            sq_err.append((y_obs - y_true) ** 2)
        return float(np.sqrt(np.mean(sq_err))) if sq_err else float('nan')

    true_traj_ab_test = np.array(data_ab['true_trajectories'])[test_idx]
    true_traj_a_test  = np.array(data_a['true_trajectories'])[test_idx]
    true_traj_d_test  = np.array(data_d['true_trajectories'])[test_idx]

    _test_fc_spec = [
        ('x_AB  ', Y_ab_test, mp_ab_test, Pp_ab_test, true_traj_ab_test, None),
        ('x_A|AB', Y_a_test,  mp_a_test,  Pp_a_test,  true_traj_a_test,  Y_a_cond_test),
        ('x_D|CD', Y_d_test,  mp_d_test,  Pp_d_test,  true_traj_d_test,  Y_d_cond_test),
    ]

    print(f"\nForecast accuracy vs GROUND TRUTH (test set, hidden last observation)")
    print(f"  [baseline = RMSE of last observation vs ground truth]")
    _gate_fc_cache = {}
    for lbl, Y_g, mp_g, Pp_g, tt_g, Yc_g in _test_fc_spec:
        fc = extract_hidden_truth_forecast_pairs(
            {'Y': Y_g}, mp_g, Pp_g, tt_g, Y_cond=Yc_g,
        )
        _gate_fc_cache[lbl] = fc
        parts = []
        for dt in CAL_HORIZONS:
            sel = fc['delta_t'] == dt
            if sel.sum() >= 5:
                rmse     = _rmse(fc['pred_mean'][sel], fc['ref_val'][sel])
                baseline = _last_obs_rmse(Y_g, tt_g, dt_filter=dt)
                parts.append(f"Δt={dt}yr RMSE={rmse:.4f} (baseline={baseline:.4f})")
            else:
                parts.append(f"Δt={dt}yr n/a")
        print(f"  Gate {lbl}  " + "  ".join(parts))

    fc_grade_test = extract_hidden_grade_truth_forecast_pairs(
        data_ab, mp_ab_test, mp_a_test, mp_d_test,
        Pp_ab_test, Pp_a_test, Pp_d_test, test_idx,
    )
    for grade_name in ['A', 'B', 'C', 'D']:
        pm_g = fc_grade_test['pred_mean'][grade_name]
        rv_g = fc_grade_test['ref_val'][grade_name]
        dt_g = fc_grade_test['delta_t']
        parts = []
        for dt in CAL_HORIZONS:
            sel = dt_g == dt
            if sel.sum() >= 5:
                parts.append(f"Δt={dt}yr RMSE={_rmse(pm_g[sel], rv_g[sel]):.4f}")
            else:
                parts.append(f"Δt={dt}yr n/a")
        print(f"  Grade {grade_name}         " + "  ".join(parts))

    # --- Calibration ---------------------------------------------------------
    print(f"\nCalibration vs GROUND TRUTH (z-score std, should be ≈ 1.0)")
    print(f"  z = (truth − pred_mean) / pred_std")
    for lbl, Y_g, mp_g, Pp_g, tt_g, Yc_g in _test_fc_spec:
        fc   = _gate_fc_cache[lbl]
        parts = []
        for dt in CAL_HORIZONS:
            sel = fc['delta_t'] == dt
            if sel.sum() >= 5:
                zs = _zstd(fc['pred_mean'][sel], fc['pred_std'][sel], fc['ref_val'][sel])
                parts.append(f"Δt={dt}yr z_std={zs:.2f}")
            else:
                parts.append(f"Δt={dt}yr n/a")
        print(f"  Gate {lbl}  " + "  ".join(parts))

    for grade_name in ['A', 'D']:
        pm_g = fc_grade_test['pred_mean'][grade_name]
        ps_g = fc_grade_test['pred_std'][grade_name]
        rv_g = fc_grade_test['ref_val'][grade_name]
        dt_g = fc_grade_test['delta_t']
        parts = []
        for dt in CAL_HORIZONS:
            sel = dt_g == dt
            if sel.sum() >= 5:
                parts.append(f"Δt={dt}yr z_std={_zstd(pm_g[sel], ps_g[sel], rv_g[sel]):.2f}")
            else:
                parts.append(f"Δt={dt}yr n/a")
        print(f"  Grade {grade_name}         " + "  ".join(parts))

    print("=" * 71)

    return manager_ab, manager_a, manager_d, ground_truth_sigmas


def run_grid_search():
    """
    Hierarchical grid search over initial state parameters for all 3 gates.

    Strategy for shared sigma_w
    ---------------------------
    sigma_w is an outer loop.  For each sigma_w candidate:
      • The kinematic params (mu_v0, var_v0, mu_a0, var_a0) are swept as before,
        and the BEST combination is chosen **independently per gate**.
      • The contribution of that sigma_w is the sum of the three per-gate best wLLs.
    The single sigma_w that maximises this sum is adopted for all gates in the
    final run_pipeline() call, while each gate still keeps its own kinematic params.

    Fallback: if somehow the search yields -inf, sigma_w falls back to
    max(individual best sigma_w found per gate).
    """
    # ── Generate data once ───────────────────────────────────────────────────
    loader = SyntheticDataLoader(CONFIG, INIT_CONDITIONS)
    loader.generate_data()
    all_inspector_ids = loader.get_all_inspector_ids()

    data_ab = loader.prepare_gate_data('ab')
    data_a  = loader.prepare_gate_data('a')
    data_d  = loader.prepare_gate_data('d')

    N = data_ab['N']
    np.random.seed(42)
    idx_perm  = np.random.permutation(N)
    train_end = int(N * TRAIN_RATIO)
    val_end   = int(N * (TRAIN_RATIO + VAL_RATIO))
    train_idx = idx_perm[:train_end]
    val_idx   = idx_perm[train_end:val_end]

    print(f"Split → train={len(train_idx)}, val={len(val_idx)}, "
          f"test={N - val_end}")

    y0_dist = compute_y0_distribution(data_ab['first_years'])

    # Pre-slice into train/val sets for all gates
    def _splits(data_gate, use_cond=True):
        return {
            'Y_tr':   data_gate['Y'][train_idx],
            'Y_val':  data_gate['Y'][val_idx],
            'Yc_tr':  data_gate['Y_cond'][train_idx] if use_cond else None,
            'Yc_val': data_gate['Y_cond'][val_idx]   if use_cond else None,
            'id_tr':  [data_gate['inspector_ids'][i] for i in train_idx],
            'id_val': [data_gate['inspector_ids'][i] for i in val_idx],
            'fy_val': data_gate['first_years'][val_idx],
        }

    sp_ab = _splits(data_ab, use_cond=False)
    sp_a  = _splits(data_a,  use_cond=True)
    sp_d  = _splits(data_d,  use_cond=True)

    # ── Grid search ──────────────────────────────────────────────────────────
    # Kinematic combos (gate-specific, same magnitude for AB/A; flipped for D)
    kin_combos = [
        (mu_v0, (mu_v0 / var_v0_ratio) ** 2,
         mu_v0 / acc_ratio, ((mu_v0 / acc_ratio) / var_a0_ratio) ** 2)
        for mu_v0      in GRID_MU_V0
        for var_v0_ratio in GRID_VAR_V0_RATIOS
        for acc_ratio  in GRID_ACC_FROM_V0_RATIOS
        for var_a0_ratio in GRID_VAR_A0_RATIOS
    ]
    n_kin   = len(kin_combos)
    n_sw    = len(GRID_SIGMA_W)
    total   = n_kin * n_sw

    print(f"\nRunning 2-level grid search: {n_sw} sigma_w × {n_kin} kinematic combos = {total} total.")
    print("  Shared sigma_w selected by: max_σw Σ_gate best_per_gate_wLL(σw)")
    print("  A & AB → negative speed (deterioration)")
    print("  D      → positive speed (improvement)")

    results           = []
    combo_idx         = 0

    # Outer: sigma_w candidates — looking for a single shared value
    best_shared_sw_score = -np.inf
    best_shared_sw       = None
    best_shared_params   = None   # (best_params_ab, best_params_a, best_params_d) at best_shared_sw

    # Also track per-gate individual bests (for fallback + reporting)
    best_val_ll_ab = -np.inf;  best_params_ab_any = None
    best_val_ll_a  = -np.inf;  best_params_a_any  = None
    best_val_ll_d  = -np.inf;  best_params_d_any  = None

    for sigma_w in GRID_SIGMA_W:
        print(f"\n{'─'*60}")
        print(f"sigma_w = {sigma_w:.2e}  (evaluating all kinematic combos)")
        print(f"{'─'*60}")

        sw_best_ll_ab = -np.inf;  sw_best_params_ab = None
        sw_best_ll_a  = -np.inf;  sw_best_params_a  = None
        sw_best_ll_d  = -np.inf;  sw_best_params_d  = None

        for (mu_v0, var_v0, mu_a0, var_a0) in kin_combos:
            combo_idx += 1
            mu_v0_d  = -mu_v0
            mu_a0_d  = -mu_a0
            var_v0_d =  var_v0
            var_a0_d =  var_a0

            print(f"  [{combo_idx}/{total}] "
                  f"μv={mu_v0:.4f} σv²={var_v0:.2e} "
                  f"μa={mu_a0:.4f} σa²={var_a0:.2e}")

            # ── Train fresh managers ─────────────────────────────────────
            m_ab = InspectorManager(all_inspector_ids,
                                    init_std=0.075, init_var=0.1**2)
            m_ab = train_inspector_variances(
                sp_ab['Y_tr'], sp_ab['Yc_tr'], sp_ab['id_tr'],
                mu_v0, var_v0, mu_a0, var_a0, m_ab, sigma_w,
            )

            m_a = InspectorManager(all_inspector_ids,
                                   init_std=0.075, init_var=0.1**2)
            m_a = train_inspector_variances(
                sp_a['Y_tr'], sp_a['Yc_tr'], sp_a['id_tr'],
                mu_v0, var_v0, mu_a0, var_a0, m_a, sigma_w,
            )

            m_d = InspectorManager(all_inspector_ids,
                                   init_std=0.075, init_var=0.1**2)
            m_d = train_inspector_variances(
                sp_d['Y_tr'], sp_d['Yc_tr'], sp_d['id_tr'],
                mu_v0_d, var_v0_d, mu_a0_d, var_a0_d, m_d, sigma_w,
                transition_fn=bridge_kinematic_transition_d,
                observation_fn=bridge_condition_observation_d,
            )

            # ── Evaluate on validation set ───────────────────────────────
            _, wll_ab, *_ = evaluate_model(
                sp_ab['Y_val'], sp_ab['Yc_val'], sp_ab['id_val'],
                mu_v0, var_v0, mu_a0, var_a0,
                sp_ab['fy_val'], y0_dist, m_ab, sigma_w,
            )
            _, wll_a, *_ = evaluate_model(
                sp_a['Y_val'], sp_a['Yc_val'], sp_a['id_val'],
                mu_v0, var_v0, mu_a0, var_a0,
                sp_a['fy_val'], y0_dist, m_a, sigma_w,
            )
            _, wll_d, *_ = evaluate_model(
                sp_d['Y_val'], sp_d['Yc_val'], sp_d['id_val'],
                mu_v0_d, var_v0_d, mu_a0_d, var_a0_d,
                sp_d['fy_val'], y0_dist, m_d, sigma_w,
                transition_fn=bridge_kinematic_transition_d,
                observation_fn=bridge_condition_observation_d,
            )

            combined = wll_ab + wll_a + wll_d
            print(f"    Val wLL → AB={wll_ab:.2f}  A={wll_a:.2f}  "
                  f"D={wll_d:.2f}  total={combined:.2f}")

            # Per-gate best within this sigma_w
            if wll_ab > sw_best_ll_ab:
                sw_best_ll_ab = wll_ab
                sw_best_params_ab = {
                    'mu_v0': mu_v0,   'var_v0': var_v0,
                    'mu_a0': mu_a0,   'var_a0': var_a0,
                    'sigma_w': sigma_w, 'paramset': combo_idx}
            if wll_a > sw_best_ll_a:
                sw_best_ll_a = wll_a
                sw_best_params_a = {
                    'mu_v0': mu_v0,   'var_v0': var_v0,
                    'mu_a0': mu_a0,   'var_a0': var_a0,
                    'sigma_w': sigma_w, 'paramset': combo_idx}
            if wll_d > sw_best_ll_d:
                sw_best_ll_d = wll_d
                sw_best_params_d = {
                    'mu_v0': mu_v0_d, 'var_v0': var_v0_d,
                    'mu_a0': mu_a0_d, 'var_a0': var_a0_d,
                    'sigma_w': sigma_w, 'paramset': combo_idx}

            # All-time per-gate bests (for fallback / reporting)
            if wll_ab > best_val_ll_ab:
                best_val_ll_ab = wll_ab
                best_params_ab_any = dict(sw_best_params_ab)
            if wll_a > best_val_ll_a:
                best_val_ll_a = wll_a
                best_params_a_any = dict(sw_best_params_a) if sw_best_params_a else None
            if wll_d > best_val_ll_d:
                best_val_ll_d = wll_d
                best_params_d_any = dict(sw_best_params_d) if sw_best_params_d else None

            results.append({
                'paramset':         combo_idx,
                'sigma_w':          sigma_w,
                'mu_v0':            mu_v0,   'var_v0':  var_v0,
                'mu_a0':            mu_a0,   'var_a0':  var_a0,
                'val_wll_ab':       wll_ab,
                'val_wll_a':        wll_a,
                'val_wll_d':        wll_d,
                'val_wll_combined': combined,
            })

        # Score this sigma_w: sum of per-gate best wLLs achieved under it
        sw_score = sw_best_ll_ab + sw_best_ll_a + sw_best_ll_d
        print(f"\n  sigma_w={sigma_w:.2e} → per-gate-best sum = {sw_score:.2f} "
              f"(AB={sw_best_ll_ab:.2f}, A={sw_best_ll_a:.2f}, D={sw_best_ll_d:.2f})")

        if sw_score > best_shared_sw_score:
            best_shared_sw_score = sw_score
            best_shared_sw       = sigma_w
            best_shared_params   = (
                sw_best_params_ab,
                sw_best_params_a,
                sw_best_params_d,
            )

    # ── Resolve final parameters ──────────────────────────────────────────────
    if best_shared_params is not None:
        best_params_ab, best_params_a, best_params_d = best_shared_params
        print(f"\n★  Shared sigma_w chosen: {best_shared_sw:.2e}  "
              f"(sum of per-gate best wLLs = {best_shared_sw_score:.2f})")
    else:
        # Fallback: take largest individually-best sigma_w
        best_shared_sw = max(
            best_params_ab_any['sigma_w'],
            best_params_a_any['sigma_w'],
            best_params_d_any['sigma_w'],
        )
        # Re-stamp the shared value into each gate's params
        best_params_ab = dict(best_params_ab_any); best_params_ab['sigma_w'] = best_shared_sw
        best_params_a  = dict(best_params_a_any);  best_params_a['sigma_w']  = best_shared_sw
        best_params_d  = dict(best_params_d_any);  best_params_d['sigma_w']  = best_shared_sw
        print(f"\n⚠  Fallback shared sigma_w = max of individual bests: {best_shared_sw:.2e}")

    # ── Save results ─────────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    csv_path   = 'grid_search_results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\nGrid search results saved to: {csv_path}")

    print(f"\nBest AB (combo #{best_params_ab['paramset']}): "
          f"μv={best_params_ab['mu_v0']:.4f} μa={best_params_ab['mu_a0']:.4f} "
          f"σw={best_shared_sw:.2e}  wLL={best_val_ll_ab:.2f}")
    print(f"Best A  (combo #{best_params_a['paramset']}): "
          f"μv={best_params_a['mu_v0']:.4f} μa={best_params_a['mu_a0']:.4f} "
          f"σw={best_shared_sw:.2e}  wLL={best_val_ll_a:.2f}")
    print(f"Best D  (combo #{best_params_d['paramset']}): "
          f"μv={best_params_d['mu_v0']:.4f} μa={best_params_d['mu_a0']:.4f} "
          f"σw={best_shared_sw:.2e}  wLL={best_val_ll_d:.2f}")

    with open('best_summary.txt', 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("BEST PARAMETER SETS\n")
        f.write("(shared sigma_w, gate-independent kinematic params)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Shared sigma_w = {best_shared_sw:.2e}\n")
        f.write(f"  (chosen by: argmax_σw Σ_gate best_per_gate_wLL(σw) = {best_shared_sw_score:.4f})\n\n")
        for gate_name, bp, wll in [
            ("AB", best_params_ab, best_val_ll_ab),
            ("A",  best_params_a,  best_val_ll_a),
            ("D",  best_params_d,  best_val_ll_d),
        ]:
            f.write(f"Gate {gate_name} (combo #{bp['paramset']}, val wLL={wll:.4f}):\n")
            f.write(f"  mu_v0:   {bp['mu_v0']:.6f}\n")
            f.write(f"  var_v0:  {bp['var_v0']:.2e}\n")
            f.write(f"  mu_a0:   {bp['mu_a0']:.6f}\n")
            f.write(f"  var_a0:  {bp['var_a0']:.2e}\n")
            f.write(f"  sigma_w: {best_shared_sw:.2e}  (shared)\n\n")
        f.write("Note: c0 from first observation (not ground truth)\n")

    # ── Full pipeline with shared sigma_w + per-gate kinematic params ─────────
    print("\n" + "=" * 70)
    print("RUNNING FULL PIPELINE WITH SHARED sigma_w + PER-GATE KINEMATIC PARAMS")
    print("=" * 70)
    run_pipeline(
        mu_v0_ab=best_params_ab['mu_v0'],  var_v0_ab=best_params_ab['var_v0'],
        mu_a0_ab=best_params_ab['mu_a0'],  var_a0_ab=best_params_ab['var_a0'],
        sigma_w_ab=best_shared_sw,
        mu_v0_a =best_params_a['mu_v0'],   var_v0_a =best_params_a['var_v0'],
        mu_a0_a =best_params_a['mu_a0'],   var_a0_a =best_params_a['var_a0'],
        sigma_w_a =best_shared_sw,
        mu_v0_d =best_params_d['mu_v0'],   var_v0_d =best_params_d['var_v0'],
        mu_a0_d =best_params_d['mu_a0'],   var_a0_d =best_params_d['var_a0'],
        sigma_w_d =best_shared_sw,
    )


if __name__ == "__main__":
    run_grid_search()
