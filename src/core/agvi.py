import jax
import jax.numpy as jnp
from typing import Tuple, NamedTuple, Optional
from functools import partial

# State container
class AGVIState(NamedTuple):
    mu: jnp.ndarray
    var: jnp.ndarray

# ---------------------------------------------------------
# 1. Math Kernels (JIT Compiled)
# ---------------------------------------------------------

@jax.jit
def square_posterior_jax(
    mu_posterior: float, 
    var_posterior: float
) -> Tuple[float, float]:
    """
    Compute moments of the squared observation (Y²).
    Handles noisy observations correctly.
    
    If y ~ N(μ, Σ), then:
    E[y²] = μ² + Σ
    Var[y²] = 2Σ² + 4Σμ²
    """
    # E[X²]
    mu_sq = mu_posterior**2 + var_posterior
    
    # Var[X²] -> This now accounts for input uncertainty (var_posterior)
    var_sq = 2*(var_posterior**2) + 4*var_posterior*(mu_posterior**2)
    
    return mu_sq, var_sq

@jax.jit
def square_prior_jax(
    mu_prior: float, 
    var_prior: float
) -> Tuple[float, float]:
    """
    Compute moments of the squared prior (V²).
    """
    mu_sq = mu_prior
    # Var[V²] = 3Σ_v + 2μ_v²
    var_sq = 3 * var_prior + 2 * (mu_prior**2)
    
    return mu_sq, var_sq

@jax.jit
def bar_posterior_jax(
    mu_prior_sq: float, var_prior_sq: float,
    mu_post_sq: float, var_post_sq: float,
    mu_prior: float, var_prior: float
) -> Tuple[float, float]:
    """
    Gaussian Conjugate Update (Kalman-like).
    """
    # Gain Calculation: k = Cov(V, V²) / Var(V²)
    k = var_prior / var_prior_sq
    
    # Mean Update
    mu_new = mu_prior + k * (mu_post_sq - mu_prior_sq)
    
    # Variance Update
    var_new = var_prior + (k**2) * (var_post_sq - var_prior_sq)
    
    return mu_new, var_new

# ---------------------------------------------------------
# 2. Update Logic (Generalized)
# ---------------------------------------------------------

@jax.jit
def single_update_step(
    state: AGVIState, 
    obs_mean: float,
    obs_var: float
) -> AGVIState:
    """
    Performs a single AGVI update step for a SCALAR trajectory.
    Now accepts explicit observation variance.
    """
    # 1. Prior State
    mu_p, var_p = state.mu, state.var
    
    # 2. Transform Prior to Squared Space
    mu_p_sq, var_p_sq = square_prior_jax(mu_p, var_p)
    
    # 3. Transform Observation to Squared Space
    # This now utilizes the obs_var passed from the user
    mu_obs_sq, var_obs_sq = square_posterior_jax(obs_mean, obs_var)
    
    # 4. Update (Inference)
    mu_new, var_new = bar_posterior_jax(
        mu_p_sq, var_p_sq, 
        mu_obs_sq, var_obs_sq, 
        mu_p, var_p
    )
    
    return AGVIState(mu=mu_new, var=var_new)

# ---------------------------------------------------------
# 3. Main Batch Process (Scan + Vmap)
# ---------------------------------------------------------

@partial(jax.jit, static_argnames=['return_history'])
def agvi_batch_jax(
    data_batch: jnp.ndarray,   # [n_samples, n_batches] -> Means (y)
    noise_batch: jnp.ndarray,  # [n_samples, n_batches] -> Variances (Σ_y)
    mu_init: jnp.ndarray,      # [n_batches]
    var_init: jnp.ndarray,     # [n_batches]
    return_history: bool = True
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generalized Optimized Batch AGVI.
    Skips steps where the provided mean or variance is NaN, keeping the prior state unchanged.
    """
    
    # 1. Vectorize the single step over the 'batch' dimension (axis 0 of inputs)
    # state: (B,), obs_mean: (B,), obs_var: (B,)
    batch_update_fn = jax.vmap(single_update_step, in_axes=(0, 0, 0))

    # 2. Define the scan body
    def scan_body(carry_state, inputs):
        # inputs is a tuple: (current_means, current_vars)
        curr_mean, curr_var = inputs

        # Run the vmapped update
        updated_state = batch_update_fn(carry_state, curr_mean, curr_var)

        # Guard against NaN observations: retain the previous state when any input is NaN
        valid_mask = ~(jnp.isnan(curr_mean) | jnp.isnan(curr_var))
        mu_safe = jnp.where(valid_mask, updated_state.mu, carry_state.mu)
        var_safe = jnp.where(valid_mask, updated_state.var, carry_state.var)
        safe_state = AGVIState(mu=mu_safe, var=var_safe)

        return safe_state, safe_state

    # 3. Initial State
    init_state = AGVIState(mu=mu_init, var=var_init)
    
    # 4. Run Scan (The Loop)
    # We assume data_batch and noise_batch have same shape [T, B]
    # scan iterates over the first dimension (T)
    final_state, history_state = jax.lax.scan(
        scan_body, 
        init_state, 
        (data_batch, noise_batch) # Tuple passed to 'inputs' in scan_body
    )
    
    # 5. Format Output
    if return_history:
        mu_full = jnp.concatenate([mu_init[None, :], history_state.mu], axis=0)
        var_full = jnp.concatenate([var_init[None, :], history_state.var], axis=0)
        return mu_full, var_full
    else:
        return final_state.mu, final_state.var

# ---------------------------------------------------------
# Execution Example
# ---------------------------------------------------------
if __name__ == "__main__":
    import numpy as np
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from graphics.agvi_plots import plot_agvi_results
    print("=== JAX AGVI (Generalized) ===")
    
    n_samples = 500
    n_batches = 2
    
    # 1. Generate Data (Means)
    # Generate random true standard deviations using uniform distribution
    np.random.seed(42)  # For reproducibility
    true_stds = np.random.uniform(0.5, 10.0, n_batches)  # Uniform distribution between 0.5 and 5.0
    print(f"Generated true standard deviations: {true_stds}")
    data_np = np.random.randn(n_samples, n_batches) * true_stds
    
    # 2. Define Observation Noise (Variances) -> no noise because data generated are perfect 
    noise_np = np.zeros((n_samples, n_batches))
    
    # Inject NaNs (for verification that NaN steps are skipped)
    nan_rows = np.arange(5)
    data_np[nan_rows, 0] = np.nan
    noise_np[nan_rows, 1] = np.nan

    # Convert to JAX
    data_jax = jnp.array(data_np)
    noise_jax = jnp.array(noise_np)
    
    mu_init = jnp.ones(n_batches) * 10.0
    var_init = jnp.ones(n_batches) * 500.0
    
    # Run
    print("Running AGVI with explicit measurement noise...")
    mu_est, var_est = agvi_batch_jax(data_jax, noise_jax, mu_init, var_init)
    mu_est.block_until_ready()
    
    # Check Results
    final_mu = mu_est[-1]
    final_std = jnp.sqrt(var_est[-1])
    
    print("\nResults:")
    for i in range(n_batches):
        print(f"Batch {i}:")
        print(f"  True Std: {true_stds[i]:.3f}")
        print(f"  True Var: {true_stds[i]**2:.3f}")
        print(f"  Est Var:  {final_mu[i]:.3f} ± {final_std[i]:.3f}")
    
    # Generate plots
    print("\nGenerating plots...")
    fig = plot_agvi_results(
        mu_est=mu_est,
        var_est=var_est,
        true_stds=true_stds,
        title="AGVI Convergence Analysis",
        save_path="agvi_results.png"
    )
    
    # Show the plot
    import matplotlib.pyplot as plt
    plt.show()
    
    print("AGVI analysis complete!")
