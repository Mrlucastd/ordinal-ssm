"""
JAX-based Kalman filter implementation with scan-based optimization.

This module provides a high-performance, JIT-compiled Kalman filter that supports
linear state-space models with optimized architecture.
"""

from typing import Optional, Tuple
import jax
import jax.numpy as jnp
from jax import lax

from .state_models import SSMparam
from functools import partial
from .state_models import SSMparam
from .utils import (
    symmetrize,
    kalman_gain,
    predict,
    obs_moments,
    obs_moments_conditional,
    conditional_cov,
    nan_mask,
    blend,
    update,
    rts_smoother_gain,
    rts_smooth,
    _materialize_R_seq,
    log_likelihood_step,
    truncated_normal_moments,
    stabilize_observation_moments,
    censored_kalman_update,
    gaussian_product
)
from .nonlinear_processor import TransformationConfig


class KalmanFilter:
    """
    High-performance JAX-based Kalman filter.
    
    This implementation uses jax.lax.scan for efficient time stepping and supports
    linear state-space models with optimized JIT boundaries and utility functions.
    """
    
    def __init__(self, ssm_params: SSMparam):
        """
        Initialize the Kalman filter with state-space parameters.
        
        Args:
            ssm_params: State-space model parameters
        """
        self.params = ssm_params
    
    
    def filter(self, Y: jnp.ndarray, R_seq: jnp.ndarray, U: Optional[jnp.ndarray] = None, params=None, learn_R=False,
               initial_mean: Optional[jnp.ndarray] = None, initial_covariance: Optional[jnp.ndarray] = None,
               process_error: Optional[float] = None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Run the Kalman filter on observations using optimized implementation.
        
        Args:
            Y: Observations (T, obs_dim)
            R_seq: Observation noise covariance sequence (T, obs_dim, obs_dim)
            U: Control inputs (T, control_dim), optional
            params: Optional JAX parameter structure for gradient-based learning.
                   If None, uses self.params (for backward compatibility).
            learn_R: Whether to learn observation noise.
            initial_mean: Optional override for initial state mean (state_dim,)
            initial_covariance: Optional override for initial state covariance (state_dim, state_dim)
            process_error: Optional scaling factor for process noise covariance Q.
                           If provided, Q = Q * process_error**2
            
        Returns:
            Tuple of (filtered_means, filtered_covariances, predicted_means, predicted_covariances, log_likelihood)
            - filtered_means: (T, state_dim)
            - filtered_covariances: (T, state_dim, state_dim)
            - predicted_means: (T, state_dim)
            - predicted_covariances: (T, state_dim, state_dim)
            - log_likelihood: total log-likelihood over all time steps
        """
        T, obs_dim = Y.shape
        
        # Use provided params or fall back to self.params
        if params is not None:
            F = params.F if hasattr(params, 'F') else params.get('F')
            Q = params.Q if hasattr(params, 'Q') else params.get('Q')
            H = params.H if hasattr(params, 'H') else params.get('H')
            B = params.B if hasattr(params, 'B') else params.get('B')
            m0 = params.m0 if hasattr(params, 'm0') else params.get('m0')
            P0 = params.P0 if hasattr(params, 'P0') else params.get('P0')
            trans_config = params.trans_config if hasattr(params, 'trans_config') else params.get('trans_config')
            obs_config = params.obs_config if hasattr(params, 'obs_config') else params.get('obs_config')
        else:
            F = self.params.transition_matrix
            Q = self.params.transition_covariance
            H = self.params.observation_matrix
            B = self.params.control_matrix
            m0 = self.params.initial_mean
            P0 = self.params.initial_covariance
            trans_config = self.params.trans_config
            obs_config = self.params.obs_config
            params = self.params
            
            params = self.params
            
        # apply overrides if provided
        if initial_mean is not None:
            m0 = initial_mean
        if initial_covariance is not None:
            P0 = initial_covariance

        # --- Runtime Initialization Handling ---
        # If m0/P0 are still None (or invalid), we must check if we can pad user-provided inputs
        # We assume if m0 is None, the user MUST have provided initial_mean (handled above)
        
        if m0 is None or P0 is None:
             raise ValueError(
                "Initial mean 'm0' and covariance 'P0' are missing. "
                "Since SSMparam no longer stores default initial conditions, "
                "you MUST provide `initial_mean` and `initial_covariance` to `kf.filter()`."
            )
            
        # Check dimensions
        # F is (state_dim, state_dim)
        state_dim = F.shape[0]
        
        # If mismatch, try to pad if params has the capability
        if m0.shape[0] != state_dim:
            if hasattr(params, 'pad_initial_state'):
                if isinstance(m0, list) and isinstance(P0, list):
                     m0, P0 = params.pad_initial_state(m0, P0)
                else:
                    # If it's an array but wrong size, we can't split it automatically safely without more info.
                    # Maybe it's a single model system?
                    # If params.model_info has length 1:
                    if hasattr(params, 'model_info') and len(params.model_info) == 1:
                         m0, P0 = params.pad_initial_state([m0], [P0])
                    else:
                         raise ValueError(
                            f"Initial mean dimension {m0.shape[0]} does not match state dimension {state_dim}. "
                            "And `pad_initial_state` could not be automatically applied (require List inputs for multi-model)."
                        )
            else:
                 raise ValueError(f"Initial mean dimension {m0.shape[0]} does not match state dimension {state_dim}.")

        # Apply process error scaling to Q if provided
        if process_error is not None:
            Q = Q * (process_error**2)
        
        # Validate input dimensions
        if obs_dim != H.shape[-2]:
            raise ValueError(f"Observation dimension mismatch: expected {H.shape[-2]}, got {obs_dim}")
        
        # Validate R_seq dimensions
        if R_seq.shape != (T, obs_dim, obs_dim):
            raise ValueError(f"R_seq shape mismatch: expected ({T}, {obs_dim}, {obs_dim}), got {R_seq.shape}")
        
        m0, P0 = params.process_transition_nonlinearities(m0, P0, trans_config)
        # Call the optimized stateless filter function and return all results
        m_filt, P_filt, m_pred, P_pred, mu_r_hist, P_r_hist, log_likelihood = kf_filter(
            params, trans_config, obs_config,
            F, Q, H, B, Y, U, R_seq, m0, P0, learn_R=learn_R
        )
        
        
        return m_filt, P_filt, m_pred, P_pred, mu_r_hist, P_r_hist, log_likelihood
    
    def filter_conditional(self, Y: jnp.ndarray, R_seq: jnp.ndarray, 
                           conditional_obs_seq: jnp.ndarray,
                           U: Optional[jnp.ndarray] = None, params=None, learn_R=False,
                           initial_mean: Optional[jnp.ndarray] = None, initial_covariance: Optional[jnp.ndarray] = None,
                           process_error: Optional[float] = None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Run the Kalman filter with conditional observation model.
        
        Args:
            Y: Primary Gate-A observations (T, obs_dim)
            R_seq: Gate-A noise covariance sequence (T, obs_dim, obs_dim)
            conditional_obs_seq: S = y_AB per timestep (T, obs_dim) — ratio denominator
            U: Control inputs (T, control_dim), optional
            params: Optional JAX parameter structure.
            learn_R: Whether to learn observation noise.
            initial_mean: Optional initial state mean (state_dim,)
            initial_covariance: Optional initial state covariance (state_dim, state_dim)
            process_error: Optional scaling factor for process noise Q.
            
        Returns:
            Same structure as filter()
        """
        T, obs_dim = Y.shape
        
        # Use provided params or fall back to self.params
        if params is not None:
            F = params.F if hasattr(params, 'F') else params.get('F')
            Q = params.Q if hasattr(params, 'Q') else params.get('Q')
            H = params.H if hasattr(params, 'H') else params.get('H')
            B = params.B if hasattr(params, 'B') else params.get('B')
            m0 = params.m0 if hasattr(params, 'm0') else params.get('m0')
            P0 = params.P0 if hasattr(params, 'P0') else params.get('P0')
            trans_config = params.trans_config if hasattr(params, 'trans_config') else params.get('trans_config')
            obs_config = params.obs_config if hasattr(params, 'obs_config') else params.get('obs_config')
        else:
            F = self.params.transition_matrix
            Q = self.params.transition_covariance
            H = self.params.observation_matrix
            B = self.params.control_matrix
            m0 = self.params.initial_mean
            P0 = self.params.initial_covariance
            trans_config = self.params.trans_config
            obs_config = self.params.obs_config
            params = self.params
            
        # apply overrides if provided
        if initial_mean is not None:
            m0 = initial_mean
        if initial_covariance is not None:
            P0 = initial_covariance

        # --- Runtime Initialization Handling ---
        if m0 is None or P0 is None:
             raise ValueError(
                "Initial mean 'm0' and covariance 'P0' are missing. "
                "You MUST provide `initial_mean` and `initial_covariance`."
            )
            
        # Check dimensions and pad if necessary
        state_dim = F.shape[0]
        if m0.shape[0] != state_dim:
            if hasattr(params, 'pad_initial_state'):
                if isinstance(m0, list) and isinstance(P0, list):
                     m0, P0 = params.pad_initial_state(m0, P0)
                else:
                    if hasattr(params, 'model_info') and len(params.model_info) == 1:
                         m0, P0 = params.pad_initial_state([m0], [P0])
                    else:
                         raise ValueError(
                            f"Initial mean dimension {m0.shape[0]} does not match state dimension {state_dim}. "
                        )
            else:
                 raise ValueError(f"Initial mean dimension {m0.shape[0]} does not match state dimension {state_dim}.")

        # Apply process error scaling to Q if provided
        if process_error is not None:
            Q = Q * (process_error**2)
        
        # Validate input dimensions
        if obs_dim != H.shape[-2]:
            raise ValueError(f"Observation dimension mismatch: expected {H.shape[-2]}, got {obs_dim}")
        
        m0, P0 = params.process_transition_nonlinearities(m0, P0, trans_config)
        
        # Call the optimized stateless filter function
        return kf_filter_conditional(
            params, trans_config, obs_config,
            F, Q, H, B, Y, U, R_seq, 
            conditional_obs_seq,
            m0, P0, learn_R=learn_R
        )

    
    def smooth(self, filtered_means: jnp.ndarray, filtered_covs: jnp.ndarray, 
               pred_means: jnp.ndarray, pred_covs: jnp.ndarray, params=None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Run the RTS smoother on filtered estimates using optimized implementation.
        
        Args:
            filtered_means: Filtered state means (T, state_dim)
            filtered_covs: Filtered state covariances (T, state_dim, state_dim)
            pred_means: Predicted state means (T, state_dim)
            pred_covs: Predicted state covariances (T, state_dim, state_dim)
            params: Optional JAX parameter structure for gradient-based learning.
                   If None, uses self.params (for backward compatibility).
            
        Returns:
            Tuple of (smoothed_means, smoothed_covariances)
            - smoothed_means: (T, state_dim)
            - smoothed_covariances: (T, state_dim, state_dim)
        """
        T, state_dim = filtered_means.shape
        
        # Use provided params or fall back to self.params
        if params is not None:
            F = params.F if hasattr(params, 'F') else params.get('F')
            Q = params.Q if hasattr(params, 'Q') else params.get('Q')
            trans_config = params.trans_config if hasattr(params, 'trans_config') else params.get('trans_config')
            processor = params if hasattr(params, 'process_transition_nonlinearities') else self.params
        else:
            F = self.params.transition_matrix
            Q = self.params.transition_covariance
            trans_config = self.params.trans_config
            processor = self.params
        
        # Validate input dimensions
        if state_dim != F.shape[-1]:
            raise ValueError(f"State dimension mismatch: expected {F.shape[-1]}, got {state_dim}")
        
        if filtered_covs.shape != (T, state_dim, state_dim):
            raise ValueError(f"Filtered covariance shape mismatch: expected ({T}, {state_dim}, {state_dim}), got {filtered_covs.shape}")
        
        if pred_means.shape != (T, state_dim):
            raise ValueError(f"Predicted means shape mismatch: expected ({T}, {state_dim}), got {pred_means.shape}")
        
        if pred_covs.shape != (T, state_dim, state_dim):
            raise ValueError(f"Predicted covariances shape mismatch: expected ({T}, {state_dim}, {state_dim}), got {pred_covs.shape}")
        
        # Call the optimized stateless smoother function with processor and trans_config
        return kf_rts_smoother(processor, trans_config, F, Q, filtered_means, filtered_covs, pred_means, pred_covs)


@partial(jax.jit, static_argnames=['processor', 'learn_R'])
def kf_filter(processor, trans_config, obs_config,
              F: jnp.ndarray, Q: jnp.ndarray, H: jnp.ndarray, B: Optional[jnp.ndarray],
              Y: jnp.ndarray, U: Optional[jnp.ndarray], R_seq: jnp.ndarray, 
              m0: jnp.ndarray, P0: jnp.ndarray, learn_R: bool = False):
    
    # ... (Début identique : T, def step, m, P, etc.) ...
    T = Y.shape[-2]

    def step(carry, inp):
        m, P, log_likelihood, t = carry
        y_t, u_t, R_t_prior = inp 
        
        # --- Check if we're at t=0 ---
        at_t0 = t == 0
        
        # --- 1. Prediction (skip if t=0) ---
        m_p, P_p = predict(F, Q, m, P, B, u_t)
        m_p_nlin, P_p_nlin = m_p, P_p
        # m_p_nlin, P_p_nlin = processor.process_transition_nonlinearities(m_p, P_p, trans_config)
        
        # --- 2. Update (skip if t=0) ---
        # Apply observation
        m_y, P_y, cov_xy = obs_moments(H, R_t_prior, m_p_nlin, P_p_nlin)
        
        # FIX: For observation nonlinearities, we need to apply them to the LATENT state
        # variance (H @ P @ H.T), not the noisy observation variance (H @ P @ H.T + R).
        # So we subtract R before processing, then add it back after.
        P_y_latent = H @ P_p_nlin @ H.T 
        m_y_nlin, P_y_nlin, cov_xy_nlin = processor.process_observation_nonlinearities(m_y, P_y_latent, cov_xy, obs_config)
        P_y_nlin = P_y_nlin + R_t_prior  # Add R back after nonlinearity processing
        
        # # FIX: Stabilize observation moments to prevent Kalman gain explosion
        # # When state is near clip boundary, Cov(clip(Z), Z) can exceed Var(Z),
        # # leading to ratio > 1 and oversized Kalman gain. This clips cov_xy
        # # to ensure valid correlations and bounded gain.
        P_x_diag = jnp.diag(P_p_nlin)
        P_y_nlin, cov_xy_nlin = stabilize_observation_moments(P_y_nlin, cov_xy_nlin, P_x_diag)
        
        # Standard Kalman update using NONLINEAR observation moments
        K = kalman_gain(cov_xy_nlin, P_y_nlin)    
        m_upd, P_upd = update(m_p_nlin, P_p_nlin, K, y_t, m_y_nlin, P_y_nlin)
        
        
        # Symmetrize
        P_upd = symmetrize(P_upd)
        
        # Masking for observations
        mask = nan_mask(y_t)
        m_with_mask = blend(mask, m_p, m_upd)
        P_with_mask = blend(mask, P_p, P_upd)
        
        # --- 3. Select: use initial m, P if t=0, else use updated values ---
        m_new = jnp.where(at_t0, m, m_with_mask)
        P_new = jnp.where(at_t0, P, P_with_mask)
        m_pred = jnp.where(at_t0, m, m_p)
        P_pred = jnp.where(at_t0, P, P_p)
        
        # --- 4. LEARN R (skip if t=0) ---
        if learn_R:
            # Calcul réel du postérieur de R
            cov_ry = R_t_prior
            K_r = kalman_gain(cov_ry, P_y)
            
            mu_R_t_prior, P_R_t_prior = jnp.zeros_like(y_t), R_t_prior
            # Update (Prior: mean=0, var=R_t_prior)
            mu_r_post, P_r_post = update(mu_R_t_prior, P_R_t_prior, K_r, y_t, m_y, P_y)
            
            # Masking for R: set to NaN if observation is missing OR if t=0
            is_valid = (mask > 0.5) & ~at_t0
            mu_r_post = jnp.where(is_valid, mu_r_post, jnp.full_like(y_t, jnp.nan))
            P_r_post = jnp.where(is_valid, P_r_post, jnp.full_like(R_t_prior, jnp.nan))
        else:
            mu_r_post = jnp.full_like(y_t, jnp.nan)
            P_r_post = jnp.full_like(R_t_prior, jnp.nan)
        
        # --- 5. Transition & LogLikelihood ---
        m_new, P_new = processor.process_transition_nonlinearities(m_new, P_new, trans_config)
        
        # Skip log-likelihood computation at t=0
        ll_step = log_likelihood_step(y_t, m_y, P_y, H)
        ll_step_masked = jnp.where(at_t0, 0.0, ll_step)
        log_likelihood = jnp.where(jnp.isfinite(ll_step_masked), log_likelihood + ll_step_masked, log_likelihood)
        
        # --- RETURNS ---
        new_carry = (m_new, P_new, log_likelihood, t + 1)
        stack_out = (m_new, P_new, m_pred, P_pred, mu_r_post, P_r_post)
        
        return new_carry, stack_out 

    # ... (Le reste de la fonction scan reste identique) ...
    # Pack inputs
    if U is None:
        k = B.shape[-1] if B is not None else 0
        U = jnp.zeros((*Y.shape[:-1], k), dtype=Y.dtype)

    Y_scan = jnp.moveaxis(Y, -2, 0)
    U_scan = jnp.moveaxis(U, -2, 0)
    R_seq_scan = jnp.moveaxis(R_seq, -3, 0)

    (m_final, P_final, total_ll, _), history = lax.scan(
        step,
        (m0, P0, jnp.array(0.0), jnp.array(0, dtype=jnp.int32)),
        (Y_scan, U_scan, R_seq_scan)
    )
    
    m_hist, P_hist, m_pred_hist, P_pred_hist, mu_r_hist, P_r_hist = history

    # Prepend initial state to history
    m_hist = jnp.concatenate([m0[jnp.newaxis, ...], m_hist], axis=0)
    P_hist = jnp.concatenate([P0[jnp.newaxis, ...], P_hist], axis=0)
    
    # Prepend initial state to prediction history as well to match dimensions
    m_pred_hist = jnp.concatenate([m0[jnp.newaxis, ...], m_pred_hist], axis=0)
    P_pred_hist = jnp.concatenate([P0[jnp.newaxis, ...], P_pred_hist], axis=0)

    return m_hist, P_hist, m_pred_hist, P_pred_hist, mu_r_hist, P_r_hist, total_ll


@partial(jax.jit, static_argnames=['processor', 'learn_R'])
def kf_filter_conditional(processor, trans_config, obs_config,
                          F: jnp.ndarray, Q: jnp.ndarray, H: jnp.ndarray, B: Optional[jnp.ndarray],
                          Y: jnp.ndarray, U: Optional[jnp.ndarray], R_seq: jnp.ndarray,
                          conditional_obs_seq: jnp.ndarray,
                          m0: jnp.ndarray, P0: jnp.ndarray, learn_R: bool = False):
    """
    Kalman filter with conditional observation model.

    The observation is  y_A = x_A / S + v_A / S  where S = y_AB.
    conditional_obs_seq carries S = y_AB for each timestep (T, obs_dim).
    Gate-A noise R_seq is scaled by 1/S² inside obs_moments_conditional.
    """
    T = Y.shape[-2]

    def step(carry, inp):
        m, P, log_likelihood, t = carry
        y_t, u_t, R_t_prior, cond_obs_t = inp
        
        # --- Check if we're at t=0 ---
        at_t0 = t == 0
        
        # --- 1. Prediction (skip if t=0) ---
        m_p, P_p = predict(F, Q, m, P, B, u_t)
        m_p_nlin, P_p_nlin = processor.process_transition_nonlinearities(m_p, P_p, trans_config)
        
        # --- 2. Update with conditional observation model ---
        # Use conditional observation moments when cond_obs is valid (not NaN)
        cond_valid = ~jnp.any(jnp.isnan(cond_obs_t))
        
        # Conditional observation moments (verified in test_moments.py)
        m_y, P_y, cov_xy = obs_moments_conditional(
            H, R_t_prior, m_p_nlin, P_p_nlin, cond_obs_t
        )
        
        # Standard Kalman update
        K = kalman_gain(cov_xy, P_y)
        m_upd, P_upd = update(m_p_nlin, P_p_nlin, K, y_t, m_y, P_y)
        
        # Symmetrize
        P_upd = symmetrize(P_upd)
        
        # Masking for observations
        mask = nan_mask(y_t)
        m_with_mask = blend(mask, m_p, m_upd)
        P_with_mask = blend(mask, P_p, P_upd)
        
        # --- 3. Select: use initial m, P if t=0, else use updated values ---
        m_new = jnp.where(at_t0, m, m_with_mask)
        P_new = jnp.where(at_t0, P, P_with_mask)
        m_pred = jnp.where(at_t0, m, m_p)
        P_pred = jnp.where(at_t0, P, P_p)
        
        # --- 4. LEARN R (skip if t=0) ---
        if learn_R:
            # Cov(z, v_A) from conditional observation model (analytically derived)
            # Cov(z, v_A) = -σ_A·σ_AB/S · E[φ(g_A)] + σ_A²/S
            # mu_phi = H @ m_p_nlin
            # cov_ry = conditional_cov(mu_phi, R_t_prior, cond_obs_t, cond_var_t)
            # cov_ry_2d = jnp.diag(cov_ry)    # (obs_dim, obs_dim) diagonal
            ratio = jnp.where(cond_obs_t > 0, 1/cond_obs_t, 1.0)
            cov_ry = ratio * R_t_prior
            K_r = kalman_gain(cov_ry, P_y)  # conditional 
            
            mu_R_t_prior, P_R_t_prior = jnp.zeros_like(y_t), R_t_prior
            mu_r_post, P_r_post = update(mu_R_t_prior, P_R_t_prior, K_r, y_t, m_y, P_y)
            
            is_valid = (mask > 0.5) & ~at_t0
            mu_r_post = jnp.where(is_valid, mu_r_post, jnp.full_like(y_t, jnp.nan))
            P_r_post = jnp.where(is_valid, P_r_post, jnp.full_like(R_t_prior, jnp.nan))
        else:
            mu_r_post = jnp.full_like(y_t, jnp.nan)
            P_r_post = jnp.full_like(R_t_prior, jnp.nan)
        
        # --- 5. Transition & LogLikelihood ---
        m_new, P_new = processor.process_transition_nonlinearities(m_new, P_new, trans_config)
        
        ll_step = log_likelihood_step(y_t, m_y, P_y, H)
        ll_step_masked = jnp.where(at_t0, 0.0, ll_step)
        log_likelihood = jnp.where(jnp.isfinite(ll_step_masked), log_likelihood + ll_step_masked, log_likelihood)
        
        # --- RETURNS ---
        new_carry = (m_new, P_new, log_likelihood, t + 1)
        stack_out = (m_new, P_new, m_pred, P_pred, mu_r_post, P_r_post)
        
        return new_carry, stack_out

    # Pack inputs
    if U is None:
        k = B.shape[-1] if B is not None else 0
        U = jnp.zeros((*Y.shape[:-1], k), dtype=Y.dtype)

    Y_scan = jnp.moveaxis(Y, -2, 0)
    U_scan = jnp.moveaxis(U, -2, 0)
    R_seq_scan = jnp.moveaxis(R_seq, -3, 0)
    cond_obs_scan = jnp.moveaxis(conditional_obs_seq, -2, 0)

    (m_final, P_final, total_ll, _), history = lax.scan(
        step,
        (m0, P0, jnp.array(0.0), jnp.array(0, dtype=jnp.int32)),
        (Y_scan, U_scan, R_seq_scan, cond_obs_scan)
    )
    
    m_hist, P_hist, m_pred_hist, P_pred_hist, mu_r_hist, P_r_hist = history

    # Prepend initial state to history
    m_hist = jnp.concatenate([m0[jnp.newaxis, ...], m_hist], axis=0)
    P_hist = jnp.concatenate([P0[jnp.newaxis, ...], P_hist], axis=0)
    
    m_pred_hist = jnp.concatenate([m0[jnp.newaxis, ...], m_pred_hist], axis=0)
    P_pred_hist = jnp.concatenate([P0[jnp.newaxis, ...], P_pred_hist], axis=0)

    return m_hist, P_hist, m_pred_hist, P_pred_hist, mu_r_hist, P_r_hist, total_ll


@partial(jax.jit, static_argnames=['processor'])
def kf_rts_smoother( processor, trans_config: TransformationConfig, F: jnp.ndarray, Q: jnp.ndarray, 
                     m_filt: jnp.ndarray, P_filt: jnp.ndarray, 
                     m_pred: jnp.ndarray, P_pred: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Stateless, optimized RTS smoother function.
    
    Args:
        processor: Nonlinearity processor instance
        trans_config: Transformation configuration for transition nonlinearities
        F: State transition matrix (..., n, n)
        Q: Process noise covariance (..., n, n)
        m_filt: Filtered state means (..., T, n)
        P_filt: Filtered state covariances (..., T, n, n)
        m_pred: Predicted state means (..., T, n)
        P_pred: Predicted state covariances (..., T, n, n)
        
    Returns:
        Tuple of (smoothed_means, smoothed_covariances)
        - smoothed_means: (..., T, n)
        - smoothed_covariances: (..., T, n, n)
    """
    T = m_filt.shape[-2]

    def step(carry, inputs):
        m_next, P_next = carry
        m_f, P_f, m_p, P_p = inputs
        # Use optimized utility functions with correct notation
        # F and Q are now correctly in scope
        # J = rts_smoother_gain(P_p, P_f, F)
        J = rts_smoother_gain(P_p, P_f, F) 
        mu_smooth, P_smooth = rts_smooth(m_next, P_next, m_f, P_f, m_p, P_p, J)
        
        P_smooth = symmetrize(P_smooth)

        mu_smooth, P_smooth = processor.process_transition_nonlinearities(mu_smooth, P_smooth, trans_config)
        
        return (mu_smooth, P_smooth), (mu_smooth, P_smooth)

    # Initialize with the last filtered estimate
    init = (m_filt[..., -1, :], P_filt[..., -1, :, :])

    # Run backward scan with reversed arrays
    (_, _), (M_rev, P_rev) = lax.scan(
        step,
        init,
        (
            m_filt[..., :-1, :][..., ::-1, :],
            P_filt[..., :-1, :, :][..., ::-1, :, :],
            m_pred[..., 1:, :][..., ::-1, :],
            P_pred[..., 1:, :, :][..., ::-1, :, :]
        )
    )

    # Combine results: smoothed estimates for times 0 to T-2, plus last filtered estimate
    mu_smooth_T = jnp.concatenate([M_rev[..., ::-1, :], m_filt[..., -1:, :]], axis=-2)
    P_smooth_T = jnp.concatenate([P_rev[..., ::-1, :, :], P_filt[..., -1:, :, :]], axis=-3)
    
    return mu_smooth_T, P_smooth_T