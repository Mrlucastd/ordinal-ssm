"""
JAX-based closed-form moment matching for nonlinear transformations.

This module provides exact analytical moments for various nonlinear functions
applied to Gaussian variables, enabling accurate uncertainty propagation
without sampling or linearization approximations.

Optimized for:
- JAX-based JIT compilation and full vectorization (no Python loops in JIT).
- Numerical stability, handling near-zero variance and edge cases.
- Homogeneous batching for maximum performance (e.g., apply_batch_clip_transforms).
- Correct covariance propagation using exact moment derivatives.
"""

import jax
import jax.numpy as jnp
import jax.scipy.special
from typing import Dict, List, Tuple, Optional, Union
from functools import partial
from .utils import stabilize_covariance_block

# Small constant for numerical stability
_EPS = 1e-15


class CloseFormMoments:
    """
    JAX-optimized closed-form moment matching for nonlinear Gaussian transformations.
    
    Provides numerically stable, vectorized, and JIT-compatible functions for
    computing moments of Y = f(Z) where Z ~ N(mu, var).
    """
    
    def __init__(self):
        """Initialize with precomputed constants for numerical efficiency."""
        # Use stop_gradient to ensure these are treated as true constants
        self.inv_sqrt_2 = jax.lax.stop_gradient(1.0 / jnp.sqrt(2.0))
        self.inv_sqrt_2pi = jax.lax.stop_gradient(1.0 / jnp.sqrt(2.0 * jnp.pi))
        self.log_2pi = jax.lax.stop_gradient(jnp.log(2.0 * jnp.pi))

    # ==================== PRIVATE HELPERS ====================

    @partial(jax.jit, static_argnums=0)
    def _compute_std_normal_quantities(self, z: jnp.ndarray):
        """
        Compute PDF and CDF of standard normal, numerically stable.
        """
        # PDF calculation is already good
        log_pdf = -0.5 * z**2 - 0.5 * self.log_2pi
        pdf = jnp.exp(log_pdf)

        # --- OPTIMIZED CDF ---
        # This one line replaces the entire jnp.where block.
        # erfc (complementary error function) is numerically
        # stable in the tails, handling both z > 6 and z < -6 correctly.
        cdf = 0.5 * jax.scipy.special.erfc(-z * self.inv_sqrt_2)
        # --- END OPTIMIZATION ---

        return pdf, cdf

    # ==================== TRANSITION MOMENTS ====================
    # These functions are for state transitions (no cov_xy needed)
    # They are fully batched and JIT-compiled for speed.

    @partial(jax.jit, static_argnums=(0,))
    def clip_transition_moments(
        self,
        mu_z: jnp.ndarray,      # (batch,)
        var_z: jnp.ndarray,     # (batch,)
        cov_zx: jnp.ndarray,    # (batch, n_other)
        lower_bound: float,
        upper_bound: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Moment matching for Y = clip(Z, a, b), where Z ~ N(mu_z, var_z).

        Returns:
            mu_clip:      E[Y]
            var_clip:     Var[Y]
            cov_clip_x:   Cov(Y, X)
            ratio:        Cov(Y, Z) / Var(Z)
        """

        # --- Standardization ---
        sigma_z = jnp.sqrt(jnp.maximum(var_z, _EPS))

        alpha = (lower_bound - mu_z) / sigma_z
        beta  = (upper_bound - mu_z) / sigma_z

        pdf_a, cdf_a = self._compute_std_normal_quantities(alpha)
        pdf_b, cdf_b = self._compute_std_normal_quantities(beta)

        # --- Region probabilities ---
        P_lower = cdf_a
        P_upper = 1.0 - cdf_b
        P_mid   = cdf_b - cdf_a

        # --- Mean ---
        mu_clip = (
            lower_bound * P_lower +
            upper_bound * P_upper +
            mu_z * P_mid -
            sigma_z * (pdf_b - pdf_a)
        )

        # --- Second moment ---
        # E[Z^2 1_{a <= Z <= b}] = (μ² + σ²) P_mid + σ(μ + a) φ(α) - σ(μ + b) φ(β)
        Ez2 = (
            lower_bound**2 * P_lower +
            upper_bound**2 * P_upper +
            (mu_z**2 + var_z) * P_mid +
            sigma_z * (mu_z + lower_bound) * pdf_a -
            sigma_z * (mu_z + upper_bound) * pdf_b
        )

        var_clip = jnp.clip(Ez2 - mu_clip**2, a_min=1e-12)

        # =========================================================
        # --- Correct E[Z * clip(Z)] ---
        # =========================================================

        # E[Z 1_{Z < a}]
        EZ_lower = mu_z * P_lower - sigma_z * pdf_a

        # E[Z 1_{Z > b}]
        EZ_upper = mu_z * P_upper + sigma_z * pdf_b

        # E[Z^2 1_{a <= Z <= b}] - corrected formula
        EZ2_mid = (
            (mu_z**2 + var_z) * P_mid +
            sigma_z * (mu_z + lower_bound) * pdf_a -
            sigma_z * (mu_z + upper_bound) * pdf_b
        )

        Efz = (
            lower_bound * EZ_lower +
            EZ2_mid +
            upper_bound * EZ_upper
        )

        cov_fz = Efz - mu_clip * mu_z

        # --- Linearized covariance scaling ---
        ratio = cov_fz / jnp.maximum(var_z, _EPS)
        cov_clip_other = ratio[:, None] * cov_zx

        return mu_clip, var_clip, cov_clip_other, ratio

    @partial(jax.jit, static_argnums=0)
    def sin_transition_moments(
        self,
        mu_z: jnp.ndarray,      # (batch,) input means
        var_z: jnp.ndarray,     # (batch,) input variances
        cov_zx: jnp.ndarray,    # (batch, n_other) covariance with other states
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Exact moments of Y = sin(Z) where Z ~ N(μ, σ²) for state transitions.
        Returns (mu, var, cov_other, ratio) where ratio is used for cov_xy scaling.
        """
        exp_neg_var_half = jnp.exp(-var_z / 2.0)
        exp_neg_2var = jnp.exp(-2.0 * var_z)
        
        mu_sin = jnp.sin(mu_z) * exp_neg_var_half
        var_sin = 0.5 * (1.0 - exp_neg_2var * jnp.cos(2.0 * mu_z)) - mu_sin**2
        var_sin = jnp.maximum(var_sin, _EPS)
        
        # Ratio A = Cov[Y, Z] / Var[Z] = cos(μ) * exp(-σ²/2)
        ratio = jnp.cos(mu_z) * exp_neg_var_half
        cov_sin_other = ratio[:, None] * cov_zx  # Cov[Y, X] = A * Cov[Z, X]

        return mu_sin, var_sin, cov_sin_other, ratio

    @partial(jax.jit, static_argnums=0)
    def exp_transition_moments(
        self,
        mu_z: jnp.ndarray,      # (batch,) input means
        var_z: jnp.ndarray,     # (batch,) input variances
        cov_zx: jnp.ndarray,    # (batch, n_other) covariance with other states
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Exact moments of Y = exp(Z) where Z ~ N(μ, σ²) for state transitions.
        Returns (mu, var, cov_other, ratio) where ratio is used for cov_xy scaling.
        """
        mu_exp = jnp.exp(mu_z + var_z / 2.0)
        var_exp = (jnp.exp(var_z) - 1.0) * jnp.exp(2.0 * mu_z + var_z)
        var_exp = jnp.maximum(var_exp, _EPS)
        
        # Ratio A = Cov[Y, Z] / Var[Z] = exp(μ + σ²/2) = E[Y]
        ratio = mu_exp
        cov_exp_other = ratio[:, None] * cov_zx # Cov[Y, X] = A * Cov[Z, X]

        return mu_exp, var_exp, cov_exp_other, ratio

    # ==================== OBSERVATION MOMENTS ====================
    # These functions are for observations (need cov_xy for observation cross-covariance)
    # They reuse transition computations and only add cov_xy scaling

    @partial(jax.jit, static_argnums=(0,))
    def clip_observation_moments(
        self,
        mu_z: jnp.ndarray,      # (batch,) input means
        var_z: jnp.ndarray,     # (batch,) input variances
        cov_zx: jnp.ndarray,    # (batch, n_other) covariance with other states
        cov_xy: jnp.ndarray,    # (batch, n_other_obs) covariance with other observations
        lower_bound: float,     # scalar lower bound
        upper_bound: float      # scalar upper bound
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        JAX-optimized clip moment matching Y = clip(Z, a, b) for observations.
        Reuses transition computation and adds cov_xy scaling.
        """
        # Reuse transition moments computation (gets mu, var, cov_other, ratio)
        mu_clip, var_clip, cov_clip_other, ratio = self.clip_transition_moments(
            mu_z, var_z, cov_zx, lower_bound, upper_bound
        )
        
        # Scale cov_xy using the same ratio
        cov_xy_scaled = ratio[:, None] * cov_xy
        
        return mu_clip, var_clip, cov_clip_other, cov_xy_scaled

    @partial(jax.jit, static_argnums=0)
    def sin_observation_moments(
        self,
        mu_z: jnp.ndarray,      # (batch,) input means
        var_z: jnp.ndarray,     # (batch,) input variances
        cov_zx: jnp.ndarray,    # (batch, n_other) covariance with other states
        cov_xy: jnp.ndarray,    # (batch, n_other_obs) covariance with other observations
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Exact moments of Y = sin(Z) where Z ~ N(μ, σ²) for observations.
        Reuses transition computation and adds cov_xy scaling.
        """
        # Reuse transition moments computation (gets mu, var, cov_other, ratio)
        mu_sin, var_sin, cov_sin_other, ratio = self.sin_transition_moments(mu_z, var_z, cov_zx)
        
        # Scale cov_xy using the same ratio
        cov_xy_scaled = ratio[:, None] * cov_xy
        
        return mu_sin, var_sin, cov_sin_other, cov_xy_scaled

    @partial(jax.jit, static_argnums=0)
    def exp_observation_moments(
        self,
        mu_z: jnp.ndarray,      # (batch,) input means
        var_z: jnp.ndarray,     # (batch,) input variances
        cov_zx: jnp.ndarray,    # (batch, n_other) covariance with other states
        cov_xy: jnp.ndarray,    # (batch, n_other_obs) covariance with other observations
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Exact moments of Y = exp(Z) where Z ~ N(μ, σ²) for observations.
        Reuses transition computation and adds cov_xy scaling.
        """
        # Reuse transition moments computation (gets mu, var, cov_other, ratio)
        mu_exp, var_exp, cov_exp_other, ratio = self.exp_transition_moments(mu_z, var_z, cov_zx)
        
        # Scale cov_xy using the same ratio
        cov_xy_scaled = ratio[:, None] * cov_xy
        
        return mu_exp, var_exp, cov_exp_other, cov_xy_scaled

    # ==================== LEGACY FUNCTIONS ====================
    # Keep original functions for backward compatibility
    # These are now wrappers around the new separated functions

    @partial(jax.jit, static_argnums=(0,))
    def clip_moments(
        self,
        mu_z: jnp.ndarray,      # (batch,) input means
        var_z: jnp.ndarray,     # (batch,) input variances
        cov_zx: jnp.ndarray,    # (batch, n_other) covariance with other states
        lower_bound: float,     # scalar lower bound
        upper_bound: float,      # scalar upper bound
        cov_xy: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]]:
        """
        Legacy clip moment matching function - delegates to appropriate function.
        """
        if cov_xy is not None:
            mu_clip, var_clip, cov_clip_other, cov_xy_scaled = self.clip_observation_moments(
                mu_z, var_z, cov_zx, cov_xy, lower_bound, upper_bound
            )
            return mu_clip, var_clip, cov_clip_other, cov_xy_scaled
        else:
            mu_clip, var_clip, cov_clip_other, ratio = self.clip_transition_moments(
                mu_z, var_z, cov_zx, lower_bound, upper_bound
            )
            return mu_clip, var_clip, cov_clip_other, None

    @partial(jax.jit, static_argnums=0)
    def sin_moments(
        self,
        mu_z: jnp.ndarray,      # (batch,) input means
        var_z: jnp.ndarray,     # (batch,) input variances
        cov_zx: jnp.ndarray,    # (batch, n_other) covariance with other states
        cov_xy: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]]:
        """
        Legacy sin moment matching function - delegates to appropriate function.
        """
        if cov_xy is not None:
            mu_sin, var_sin, cov_sin_other, cov_xy_scaled = self.sin_observation_moments(
                mu_z, var_z, cov_zx, cov_xy
            )
            return mu_sin, var_sin, cov_sin_other, cov_xy_scaled
        else:
            mu_sin, var_sin, cov_sin_other, ratio = self.sin_transition_moments(mu_z, var_z, cov_zx)
            return mu_sin, var_sin, cov_sin_other, None

    @partial(jax.jit, static_argnums=0)
    def exp_moments(
        self,
        mu_z: jnp.ndarray,      # (batch,) input means
        var_z: jnp.ndarray,     # (batch,) input variances
        cov_zx: jnp.ndarray,    # (batch, n_other) covariance with other states
        cov_xy: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]]:
        """
        Legacy exp moment matching function - delegates to appropriate function.
        """
        if cov_xy is not None:
            mu_exp, var_exp, cov_exp_other, cov_xy_scaled = self.exp_observation_moments(
                mu_z, var_z, cov_zx, cov_xy
            )
            return mu_exp, var_exp, cov_exp_other, cov_xy_scaled
        else:
            mu_exp, var_exp, cov_exp_other, ratio = self.exp_transition_moments(mu_z, var_z, cov_zx)
            return mu_exp, var_exp, cov_exp_other, None

    # ==================== STATE TRANSFORMATION INTERFACE ====================
    # These functions apply transformations to a *full* (n, n) covariance matrix.

    @partial(jax.jit, static_argnums=(0, 3, 4, 5, 6, 7))
    def apply_transformation(
        self,
        mean: jnp.ndarray,            # (n,) state mean
        cov: jnp.ndarray,             # (n, n) state covariance
        input_idx: int,               # input state index
        output_idx: int,              # output state index
        transform_type: str,          # 'sin', 'exp', 'clip'
        lower_bound: Optional[float] = None,  # for clip only
        upper_bound: Optional[float] = None   # for clip only
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Apply a *single* nonlinear transformation to a full state vector.
        
        JIT-compiled for scalar indices and transform types.
        """
        # Extract input state moments (keep batch dimension for consistency)
        mu_z = mean[input_idx:input_idx+1]      # Shape: (1,)
        var_z = cov[input_idx, input_idx:input_idx+1] # Shape: (1,)
        cov_zx = cov[input_idx:input_idx+1, :]    # Shape: (1, n)

        # Apply appropriate transformation
        # JAX's JIT will optimize this. Since transform_type is static,
        # it will compile a version of this function where only one
        # branch is ever taken.
        if transform_type == 'sin':
            mu_out, var_out, cov_other, _ = self.sin_transition_moments(mu_z, var_z, cov_zx)
        elif transform_type == 'exp':
            mu_out, var_out, cov_other, _ = self.exp_transition_moments(mu_z, var_z, cov_zx)
        elif transform_type == 'clip':
            if lower_bound is None or upper_bound is None:
                raise ValueError("Clip transformation requires lower_bound and upper_bound")
            mu_out, var_out, cov_other, _ = self.clip_transition_moments(
                mu_z, var_z, cov_zx, lower_bound, upper_bound
            )
        else:
            raise ValueError(f"Unsupported transformation type: {transform_type}")

        # Extract scalar values from batch results
        mu_out_scalar = mu_out[0]
        var_out_scalar = var_out[0]
        cov_out_vector = cov_other[0, :]  # Shape: (n,)

        # Update mean and covariance
        mean = mean.at[output_idx].set(mu_out_scalar)
        
        # Symmetrically update covariance matrix
        # This is a robust way to set the row and column
        cov = cov.at[output_idx, :].set(cov_out_vector)
        cov = cov.at[:, output_idx].set(cov_out_vector)
        cov = cov.at[output_idx, output_idx].set(var_out_scalar)

        return mean, cov

    # ==================== OPTIMIZED BATCHED TRANSFORMATIONS ====================
    # This is the new, high-performance API.
    # The filter should group nonlinearities by type and call these functions.

    @partial(jax.jit, static_argnums=0)
    def apply_batch_clip_transforms(
        self,
        mean: jnp.ndarray,            # (n,) state mean
        cov: jnp.ndarray,             # (n, n) state covariance
        input_indices: jnp.ndarray,   # (batch,) input indices
        output_indices: jnp.ndarray,  # (batch,) output indices
        lower_bounds: jnp.ndarray,    # (batch,) lower bounds
        upper_bounds: jnp.ndarray,     # (batch,) upper bounds
        cov_xy: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]]:
        """
        Apply a batch of *clip* transformations in a fully vectorized,
        JIT-compiled way. Contains no Python loops.
        """
        # Extract all input moments in a single vectorized operation
        mu_inputs = mean[input_indices]                # (batch,)
        var_inputs = cov[input_indices, input_indices] # (batch,)
        cov_with_others = cov[input_indices, :]        # (batch, n)
        
        # Apply all clip moments in one batch operation
        # --- FIX: Rename return variables for clarity ---
        # The 4th return is cov_xy_scaled (if cov_xy input exists) OR the scaling ratio (if input is None)
        mu_clipped, var_clipped, cov_scaled, ratio_or_cov_xy = self.clip_moments(
            mu_inputs, var_inputs, cov_with_others, 
            lower_bounds, upper_bounds, cov_xy
        )
        # ------------------------------------------------
        
        # --- Vectorized Update ---
        # 1. Update all means
        mean = mean.at[output_indices].set(mu_clipped)
        
        # 2. Symmetrically update covariance
        cov = cov.at[output_indices, :].set(cov_scaled)
        cov = cov.at[:, output_indices].set(cov_scaled.T)
        cov = cov.at[output_indices, output_indices].set(var_clipped)
        
        cov = stabilize_covariance_block(cov, output_indices)
        
        return mean, cov, ratio_or_cov_xy

    @partial(jax.jit, static_argnums=0)
    def apply_batch_sin_transforms(
        self,
        mean: jnp.ndarray,            # (n,) state mean
        cov: jnp.ndarray,             # (n, n) state covariance
        input_indices: jnp.ndarray,   # (batch,) input indices
        output_indices: jnp.ndarray,  # (batch,) output indices
        cov_xy: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]]:
        """
        Apply a batch of *sin* transformations in a fully vectorized,
        JIT-compiled way.
        """
        # Extract all input moments in a single vectorized operation
        mu_inputs = mean[input_indices]
        var_inputs = cov[input_indices, input_indices]
        cov_with_others = cov[input_indices, :]
        
        # Apply all sin moments in one batch operation
        mu_sins, var_sins, cov_scaled, cov_xy = self.sin_moments(
            mu_inputs, var_inputs, cov_with_others, cov_xy
        )
        
        # --- Vectorized Update ---
        mean = mean.at[output_indices].set(mu_sins)
        cov = cov.at[output_indices, :].set(cov_scaled)
        cov = cov.at[:, output_indices].set(cov_scaled.T)
        cov = cov.at[output_indices, output_indices].set(var_sins)
        
        cov = stabilize_covariance_block(cov, output_indices)
        
        return mean, cov, cov_xy

    @partial(jax.jit, static_argnums=0)
    def apply_batch_exp_transforms(
        self,
        mean: jnp.ndarray,            # (n,) state mean
        cov: jnp.ndarray,             # (n, n) state covariance
        input_indices: jnp.ndarray,   # (batch,) input indices
        output_indices: jnp.ndarray,  # (batch,) output indices
        cov_xy: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]]:
        """
        Apply a batch of *exp* transformations in a fully vectorized,
        JIT-compiled way.
        """
        # Extract all input moments in a single vectorized operation
        mu_inputs = mean[input_indices]
        var_inputs = cov[input_indices, input_indices]
        cov_with_others = cov[input_indices, :]
        
        # Apply all exp moments in one batch operation
        mu_exps, var_exps, cov_scaled, cov_xy = self.exp_moments(
            mu_inputs, var_inputs, cov_with_others, cov_xy
        )
        
        # --- Vectorized Update ---
        mean = mean.at[output_indices].set(mu_exps)
        cov = cov.at[output_indices, :].set(cov_scaled)
        cov = cov.at[:, output_indices].set(cov_scaled.T)
        cov = cov.at[output_indices, output_indices].set(var_exps)
        
        cov = stabilize_covariance_block(cov, output_indices)
        
        return mean, cov, cov_xy


if __name__ == "__main__":
    # Comprehensive test case
    
    # Create moment matching instance
    mm = CloseFormMoments()
    
    print("=" * 60)
    print("COMPREHENSIVE CLOSED-FORM MOMENT MATCHING TEST")
    print("=" * 60)
    
    # --- Shared Inputs ---
    mu_z_batch = jnp.array([0.0, 1.0, -0.5])
    var_z_batch = jnp.array([0.0001, 0.5, 2.0])**2
    cov_zx_batch = jnp.array([[0.1, 0.2], [0.3, 0.1], [0.15, 0.25]])
    cov_xy_batch = jnp.array([[0.05, 0.1], [0.15, 0.05], [0.08, 0.12]])
    
    # --- TRANSITION SIN TEST ---
    print("\n" + "=" * 40)
    print("TRANSITION SIN MOMENT MATCHING TEST")
    print("=" * 40)
    
    mu_sin, var_sin, cov_sin, ratio = mm.sin_transition_moments(mu_z_batch, var_z_batch, cov_zx_batch)
    
    print("Transition Sin Moment Matching Test:")
    print(f"Input means: {mu_z_batch}")
    print(f"Input variances: {var_z_batch}")
    print(f"Output means: {mu_sin}")
    print(f"Output variances: {var_sin}")
    print(f"Scaled covariances: {cov_sin}")
    print(f"Ratio for cov_xy: {ratio}")
    
    # --- OBSERVATION SIN TEST ---
    print("\n" + "=" * 40)
    print("OBSERVATION SIN MOMENT MATCHING TEST")
    print("=" * 40)
    
    mu_sin_obs, var_sin_obs, cov_sin_obs, cov_xy_scaled = mm.sin_observation_moments(
        mu_z_batch, var_z_batch, cov_zx_batch, cov_xy_batch
    )
    
    print("Observation Sin Moment Matching Test:")
    print(f"Input means: {mu_z_batch}")
    print(f"Input variances: {var_z_batch}")
    print(f"Input cov_xy: {cov_xy_batch}")
    print(f"Output means: {mu_sin_obs}")
    print(f"Output variances: {var_sin_obs}")
    print(f"Scaled state covariances: {cov_sin_obs}")
    print(f"Scaled obs covariances: {cov_xy_scaled}")

    # --- TRANSITION EXP TEST ---
    print("\n" + "=" * 40)
    print("TRANSITION EXP MOMENT MATCHING TEST")
    print("=" * 40)
    
    mu_exp, var_exp, cov_exp, ratio = mm.exp_transition_moments(mu_z_batch, var_z_batch, cov_zx_batch)
    
    print("Transition Exp Moment Matching Test:")
    print(f"Input means: {mu_z_batch}")
    print(f"Input variances: {var_z_batch}")
    print(f"Output means: {mu_exp}")
    print(f"Output variances: {var_exp}")
    print(f"Scaled covariances: {cov_exp}")
    print(f"Ratio for cov_xy: {ratio}")

    # --- OBSERVATION EXP TEST ---
    print("\n" + "=" * 40)
    print("OBSERVATION EXP MOMENT MATCHING TEST")
    print("=" * 40)
    
    mu_exp_obs, var_exp_obs, cov_exp_obs, cov_xy_scaled_exp = mm.exp_observation_moments(
        mu_z_batch, var_z_batch, cov_zx_batch, cov_xy_batch
    )
    
    print("Observation Exp Moment Matching Test:")
    print(f"Input means: {mu_z_batch}")
    print(f"Input variances: {var_z_batch}")
    print(f"Input cov_xy: {cov_xy_batch}")
    print(f"Output means: {mu_exp_obs}")
    print(f"Output variances: {var_exp_obs}")
    print(f"Scaled state covariances: {cov_exp_obs}")
    print(f"Scaled obs covariances: {cov_xy_scaled_exp}")

    # --- TRANSITION CLIP TEST ---
    print("\n" + "=" * 40)
    print("TRANSITION CLIP MOMENT MATCHING TEST")
    print("=" * 40)
    
    lower_bound = -1.0
    upper_bound = 1.0
    
    mu_clip, var_clip, cov_clip, ratio = mm.clip_transition_moments(
        mu_z_batch, var_z_batch, cov_zx_batch, lower_bound, upper_bound
    )
    
    print("Transition Clip Moment Matching Test:")
    print(f"Input means: {mu_z_batch}")
    print(f"Input variances: {var_z_batch}")
    print(f"Output means: {mu_clip}")
    print(f"Output variances: {var_clip}")
    print(f"Scaled covariances: {cov_clip}")
    print(f"Ratio for cov_xy: {ratio}")

    # --- OBSERVATION CLIP TEST ---
    print("\n" + "=" * 40)
    print("OBSERVATION CLIP MOMENT MATCHING TEST")
    print("=" * 40)
    
    mu_clip_obs, var_clip_obs, cov_clip_obs, cov_xy_scaled_clip = mm.clip_observation_moments(
        mu_z_batch, var_z_batch, cov_zx_batch, cov_xy_batch, lower_bound, upper_bound
    )
    
    print("Observation Clip Moment Matching Test:")
    print(f"Input means: {mu_z_batch}")
    print(f"Input variances: {var_z_batch}")
    print(f"Input cov_xy: {cov_xy_batch}")
    print(f"Output means: {mu_clip_obs}")
    print(f"Output variances: {var_clip_obs}")
    print(f"Scaled state covariances: {cov_clip_obs}")
    print(f"Scaled obs covariances: {cov_xy_scaled_clip}")

    print("\n" + "=" * 40)
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 40)
