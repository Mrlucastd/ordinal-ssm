"""
JAX-based utility functions for Kalman filtering.

This module provides optimized, JIT-compiled utility functions for common
linear algebra operations used in Kalman filtering.
"""

import jax
import jax.numpy as jnp
from jax.scipy import linalg
from jax.scipy.stats import norm
from typing import Optional, Union
from functools import partial

# New optimized utility functions for Kalman filter
@jax.jit
def std_normal_pdf_cdf(z):
    log_pdf = -0.5 * z**2 - 0.5 * jnp.log(2 * jnp.pi)
    pdf = jnp.exp(log_pdf)
    cdf = 0.5 * jax.scipy.special.erfc(-z / jnp.sqrt(2))
    return pdf, cdf

@jax.jit
def symmetrize(P: jnp.ndarray) -> jnp.ndarray:
    """
    Ensure covariance matrix is symmetric by averaging with its transpose.
    Supports batch dimensions.
    
    Args:
        P: Covariance matrix or batch of covariance matrices
        
    Returns:
        Symmetrized covariance matrix
    """
    return 0.5 * (P + P.swapaxes(-1, -2))


@jax.jit
def kalman_gain(PHt: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
    """
    Compute Kalman gain using Cholesky decomposition for numerical stability.
    Supports batch dimensions.
    
    Args:
        P_pred: Predicted state covariance
        H: Observation matrix
        S: Predicted observation covariance (innovation covariance)
        
    Returns:
        Kalman gain matrix
    """
    # Solve for K = P_pred H^T S^{-1} with Cholesky
    L = linalg.cholesky(S, lower=True)
    K = linalg.cho_solve((L, True), PHt.swapaxes(-1, -2)).swapaxes(-1, -2)
    return K


@jax.jit
def predict(F: jnp.ndarray, Q: jnp.ndarray, m: jnp.ndarray, P: jnp.ndarray, 
           B: Optional[jnp.ndarray] = None, u: Optional[jnp.ndarray] = None) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Predict state mean and covariance. Supports batch dimensions.
    
    Args:
        F: State transition matrix
        Q: Process noise covariance
        m: Current state mean
        P: Current state covariance
        B: Control matrix (optional)
        u: Control input (optional)
        
    Returns:
        Tuple of (predicted_mean, predicted_covariance)
    """
    if B is None or u is None:
        m_pred = F @ m
    else:
        m_pred = F @ m + B @ u
    P_pred = F @ P @ F.swapaxes(-1, -2) + Q
    return m_pred, P_pred


@jax.jit
def obs_moments(H: jnp.ndarray, R: jnp.ndarray, m_pred: jnp.ndarray, P_pred: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute predicted observation mean and covariance. Supports batch dimensions.
    
    Args:
        H: Observation matrix
        R: Observation noise covariance
        m_pred: Predicted state mean
        P_pred: Predicted state covariance
        
    Returns:
        Tuple of (predicted_observation_mean, predicted_observation_covariance)
    """
    mu = H @ m_pred
    S = H @ P_pred @ H.swapaxes(-1, -2) + R
    cov_xy = P_pred @ H.swapaxes(-1, -2)
    return mu, S, cov_xy


@jax.jit
def obs_moments_conditional(
    H: jnp.ndarray, 
    R: jnp.ndarray, 
    m_pred: jnp.ndarray, 
    P_pred: jnp.ndarray,
    conditional_obs: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute observation moments for the conditional/ratio observation model.

    The observation model is  y_A = x_A / S + v_A / S,  where S = y_AB.
    This simplifies to scaling the Gate-A noise by 1/S²:
        E[y_A]   = H @ m_pred         (mu_phi)
        Var[y_A] = H @ P_pred @ H^T + diag(σ_A²) / S²
        Cov(x,y) = P_pred @ H^T       (exact, unchanged)
    """
    # --- Prior state moments through H ---
    mu_phi = H @ m_pred                          # E[φ(g_A)]
    P_phi  = H @ P_pred @ H.swapaxes(-1, -2)    # Var[φ(g_A)]
    cov_xy = P_pred @ H.swapaxes(-1, -2)        # Cov(state, obs)

    # --- Scale Gate-A noise by 1/S² ---
    S       = conditional_obs
    ratio = jnp.where(S > 0, 1/S, 1.0)
    inv_S_sq = ratio ** 2

    sigma_A_sq = jnp.diagonal(R)                # σ_A² from Gate-A inspector
    R_scaled   = jnp.diag(sigma_A_sq) * inv_S_sq

    mu = mu_phi
    Sy = P_phi + R_scaled

    return mu, Sy, cov_xy


@jax.jit
def conditional_cov(
    mu_phi: jnp.ndarray,
    R: jnp.ndarray,
    conditional_obs: jnp.ndarray,
    conditional_var: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute Cov(z, v_A) with sign correctly handled based on σ_A vs σ_AB.
    """
    S = conditional_obs
    inv_S = jnp.where(S > 0, 1.0 / S, 0.0)

    sigma_A_sq = jnp.diagonal(R)
    sigma_A = jnp.sqrt(sigma_A_sq)
    sigma_AB = jnp.sqrt(conditional_var)

    # --- Sign factor ---
    sign_factor = jnp.where(sigma_AB <= sigma_A, -1.0, 1.0)

    # Cov(φ·ε_A, v_A) = ±σ_A·σ_AB / S * μ_φ
    Cov_phi_eps_vA = sign_factor * (sigma_A * sigma_AB) * inv_S * mu_phi

    # Cov(v_A/S, v_A) = σ_A² / S
    Cov_vA_S_vA = sigma_A_sq * inv_S

    return Cov_vA_S_vA + Cov_phi_eps_vA

@jax.jit
def nan_mask(y: jnp.ndarray) -> jnp.ndarray:
    """
    Scalar mask indicating if ANY component is NaN. Returns 0.0 for NaN, 1.0 for valid.
    
    Args:
        y: Observation vector
        
    Returns:
        Scalar mask in {0, 1}
    """
    return 1.0 - jnp.asarray(jnp.any(jnp.isnan(y)), y.dtype)


@jax.jit
def blend(mask: jnp.ndarray, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """
    Blend two arrays using scalar mask. mask=1 returns b, mask=0 returns a.
    
    Args:
        mask: Scalar mask in {0, 1}
        a: First array (used when mask=0)
        b: Second array (used when mask=1)
        
    Returns:
        Blended array: mask*b + (1-mask)*a
    """
    return jnp.where(mask > 0.5, b, a)

@jax.jit
def update(m_pred: jnp.ndarray, P_pred: jnp.ndarray, K: jnp.ndarray, 
                        y_t: jnp.ndarray, mu: jnp.ndarray, S: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Update state mean and covariance using numerically stable Joseph form.
    
    The Joseph form ensures numerical stability and guarantees positive 
    semi-definite covariance, unlike the standard form P = P - K @ S @ K^T
    which can become negative due to numerical errors when K is large.
    
    Joseph form: P = (I - K @ H) @ P @ (I - K @ H)^T + K @ R @ K^T
    Simplified: Since S = H @ P @ H^T + R and K @ S @ K^T = K @ (H @ P @ H^T + R) @ K^T,
                we use: P = P - K @ S / 2 @ K^T (symmetrized) as intermediate,
                then enforce positive definiteness.
    
    For robustness, we use: P = P - K @ H @ P (simplified Joseph), 
    where K @ H is implicitly K @ S @ K^T / P_pred structure.
    
    Args:
        m_pred: Predicted state mean
        P_pred: Predicted state covariance
        K: Kalman gain
        y_t: Current observation
        mu: Predicted observation mean
        S: Predicted observation covariance (innovation covariance)
        
    Returns:
        Tuple of (updated_mean, updated_covariance)
    """
    innov = jnp.nan_to_num(y_t - mu)
    m_upd = m_pred + K @ innov
    
    # Standard form (can become negative with large K)
    # P_upd = P_pred - K @ S @ K.swapaxes(-1, -2)
    
    # Joseph form using the identity:
    # P_upd = (I - K @ H) @ P_pred @ (I - K @ H)^T + K @ R @ K^T
    # We can derive: K @ H @ P_pred = K @ S - K @ R (since S = H @ P_pred @ H^T + R)
    # But we don't have H and R separately here.
    
    # Alternative numerically stable approach:
    # Use the standard form but clamp diagonal to be non-negative
    P_upd = P_pred - K @ S @ K.swapaxes(-1, -2)
    
    # Enforce positive semi-definiteness by clamping diagonal
    # This is a pragmatic fix that prevents numerical blow-up
    diag = jnp.diagonal(P_upd, axis1=-2, axis2=-1)
    diag_clamped = jnp.maximum(diag, 1e-15)
    
    # Use at indexing for batch-compatible diagonal update
    n = P_upd.shape[-1]
    idx = jnp.arange(n)
    
    # Handle both 2D and higher-dimensional cases
    if P_upd.ndim == 2:
        P_upd = P_upd.at[idx, idx].set(diag_clamped)
    else:
        # For batched case, we need to use advanced indexing
        P_upd = P_upd.at[..., idx, idx].set(diag_clamped)
    
    return m_upd, P_upd


@jax.jit
def rts_smoother_gain(P_pred: jnp.ndarray, P_filt: jnp.ndarray, F: jnp.ndarray) -> jnp.ndarray:
    """
    Compute RTS smoother gain using Pseudo-Inverse for numerical stability.
    Supports batch dimensions.
    
    Args:
        P_pred: Predicted state covariance from forward filter (P_{t+1|t})
        P_filt: Filtered state covariance from forward filter (P_{t|t})
        F: State transition matrix
        
    Returns:
        RTS smoother gain matrix
    """
    # P_pred is already P_{t+1|t} (predicted covariance for next step)
    
    # Use pseudo-inverse to handle rank-deficient matrices (structural singularities)
    # This avoids adding bias (jitter) to small variances
    # rcond=1e-15 allow us to catch the numerical noise but keep the small variances
    P_inv = jnp.linalg.pinv(P_pred, rcond=1e-15)
    
    # J = P_filt @ F^T @ P_{t+1|t}^{-1}
    cross_cov = P_filt @ F.swapaxes(-1, -2)
    J = cross_cov @ P_inv
    return J



@jax.jit
def rts_smooth(m_smooth_next: jnp.ndarray, P_smooth_next: jnp.ndarray, 
                         m_filt: jnp.ndarray, P_filt: jnp.ndarray,
                         m_pred: jnp.ndarray, P_pred: jnp.ndarray,
                         J: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    RTS smoothing step using pre-computed smoother gain. Supports batch dimensions.
    
    Args:
        m_smooth_next: Smoothed state mean at next time step
        P_smooth_next: Smoothed state covariance at next time step
        m_filt: Filtered state mean at current time step
        P_filt: Filtered state covariance at current time step
        m_pred: Predicted state mean at current time step
        P_pred: Predicted state covariance at current time step
        J: Pre-computed RTS smoother gain (J = P_filt @ F.T @ P_pred^{-1})
        
    Returns:
        Tuple of (smoothed_mean, smoothed_covariance)
    """
    # Smoothed mean
    mu_smooth = m_filt + J @ (m_smooth_next - m_pred)
    
    # Smoothed covariance (standard RTS form)
    P_smooth = P_filt + J @ (P_smooth_next - P_pred) @ J.swapaxes(-1, -2)
    
    # Simple diagonal clamping for numerical stability
    # This is simpler and less likely to cause issues than eigenvalue decomposition
    EPS = 1e-12
    diag = jnp.diagonal(P_smooth, axis1=-2, axis2=-1)
    diag_clamped = jnp.maximum(diag, EPS)
    
    n_dim = P_smooth.shape[-1]
    idx = jnp.arange(n_dim)
    
    if P_smooth.ndim == 2:
        P_smooth = P_smooth.at[idx, idx].set(diag_clamped)
    else:
        P_smooth = P_smooth.at[..., idx, idx].set(diag_clamped)
    
    return mu_smooth, P_smooth


@jax.jit
def log_likelihood_step(y_t: jnp.ndarray, mu_y: jnp.ndarray, Sy: jnp.ndarray, 
                       H: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the log-likelihood for a single time step using Cholesky decomposition.
    
    Includes regularization (jitter) to prevent Cholesky failures on 
    numerically singular covariance matrices.
    """
    # 1. Constants
    EPS = 1e-10 # Jitter for float32. Use 1e-8 for float64
    obs_dim = H.shape[-2]
    
    # 2. Handle NaN observations
    # If any component is NaN, mask is 0.0, else 1.0
    mask = nan_mask(y_t)
    
    # 3. Safe Inputs (Pre-computation masking)
    # Replace NaNs in y_t with expected mean (innov -> 0)
    y_t_safe = jnp.where(mask > 0.5, y_t, mu_y)
    
    # Replace Sy with Identity if masked, OR if Sy is valid but we want to 
    # ensure gradients flow smoothly, we usually just use Identity for the 
    # masked case.
    # CRITICAL: Add jitter to Sy to ensure Positive Definiteness
    Sy_stabilized = Sy + EPS * jnp.eye(obs_dim)
    Sy_safe = jnp.where(mask > 0.5, Sy_stabilized, jnp.eye(obs_dim))
    
    # 4. Cholesky Decomposition
    # If Sy_safe is not PD, this returns NaNs. The jitter prevents this.
    L_S = linalg.cholesky(Sy_safe, lower=True)
    
    # 5. Innovation & Mahalanobis Distance
    innov = y_t_safe - mu_y
    # Solve L_S * alpha = innov
    alpha = linalg.solve_triangular(L_S, innov, lower=True)
    mahalanobis = jnp.sum(alpha * alpha, axis=-1)
    
    # 6. Log Determinant
    # 2 * sum(log(diag(L)))
    log_det_S = 2.0 * jnp.sum(jnp.log(jnp.diag(L_S)), axis=-1)
    
    # 7. Compute Constant Term
    log_2pi = jnp.log(2.0 * jnp.pi)
    
    # 8. Final Calculation
    # If masked (mask=0), Sy_safe=I, so log_det=0. innov=0, so mahal=0.
    # The formula gives -0.5 * (0 + 0 + k*log(2pi)).
    # This is NOT zero. So we must multiply the *entire result* by mask.
    ll_step = -0.5 * (log_det_S + mahalanobis + obs_dim * log_2pi)
    
    return mask * ll_step


@partial(jax.jit, static_argnums=(3,))
def stabilize_observation_moments(
    P_y: jnp.ndarray,           # (obs_dim, obs_dim) observation covariance
    cov_xy: jnp.ndarray,        # (state_dim, obs_dim) cross-covariance
    P_x_diag: jnp.ndarray,      # (state_dim,) diagonal of state covariance
    correlation_limit: float = 0.999
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Enforces valid correlations and positive updated variance by clipping cov_xy.
    
    Three constraints are applied:
    1. Valid correlation: |cov_xy[i,j]| <= correlation_limit * sqrt(var_x[i] * var_y[j])
    2. Positive variance update: P_upd = var_x - K²*var_y > 0
       Since K = cov_xy/var_y, this means: var_x - (cov_xy/var_y)² * var_y > 0
       Simplifying: |cov_xy| < sqrt(var_x * var_y)
    3. Minimum variance floor: ensure P_upd >= min_variance_ratio * var_x
       This gives: |cov_xy| <= sqrt(var_x * var_y * (1 - min_variance_ratio))
    
    Args:
        P_y: (obs_dim, obs_dim) observation covariance matrix (unchanged)
        cov_xy: (state_dim, obs_dim) cross-covariance between states and observations  
        P_x_diag: (state_dim,) diagonal of state covariance matrix
        correlation_limit: Maximum allowed correlation (default 0.999)
        
    Returns:
        Tuple of (P_y, stabilized_cov_xy) - P_y unchanged, cov_xy clipped if needed
    """
    # Extract observation variances (diagonal of P_y)
    var_y = jnp.diagonal(P_y)  # (obs_dim,)
    
    # Safe variances (avoid sqrt of negative or zero)
    safe_var_x = jnp.maximum(P_x_diag, 1e-15)  # (state_dim,)
    safe_var_y = jnp.maximum(var_y, 1e-15)     # (obs_dim,)
    
    # Compute standard deviations
    std_x = jnp.sqrt(safe_var_x)  # (state_dim,)
    std_y = jnp.sqrt(safe_var_y)  # (obs_dim,)
    
    # Constraint 1: Valid correlation
    # |cov_xy[i,j]| <= rho * std_x[i] * std_y[j]
    max_cov_correlation = correlation_limit * (std_x[:, None] * std_y[None, :])
    
    # Constraint 2: Positive variance update with safety margin
    # P_upd = var_x - K² * var_y >= min_variance_ratio * var_x
    # K = cov_xy / var_y
    # var_x - (cov_xy/var_y)² * var_y >= min_variance_ratio * var_x
    # (1 - min_variance_ratio) * var_x >= cov_xy² / var_y
    # |cov_xy| <= sqrt((1 - min_variance_ratio) * var_x * var_y)
    min_variance_ratio = 0.10  # Ensure at least 10% of original variance remains
    max_cov_positive_update = jnp.sqrt((1.0 - min_variance_ratio) * safe_var_x[:, None] * safe_var_y[None, :])
    
    # Take the stricter (minimum) constraint
    max_cov_magnitude = jnp.minimum(max_cov_correlation, max_cov_positive_update)
    
    # Clip cov_xy to respect constraints
    cov_xy_stabilized = jnp.clip(cov_xy, -max_cov_magnitude, max_cov_magnitude)
    
    return P_y, cov_xy_stabilized

@partial(jax.jit, static_argnums=(2,))
def stabilize_covariance_block(
    cov: jnp.ndarray, 
    updated_indices: jnp.ndarray, 
    correlation_limit: float = 0.999999
) -> jnp.ndarray:
    """
    Enforces valid correlations for specific rows/cols in a covariance matrix.
    Optimized for JAX JIT compilation.
    
    Args:
        cov: (n, n) full covariance matrix.
        updated_indices: (batch,) Indices of the states just transformed.
        correlation_limit: Static float (0.999) for max correlation.
    """
    # 1. Extract Variances
    # We access the diagonal just once
    all_vars = jnp.diagonal(cov)
    new_vars = all_vars[updated_indices]

    # 2. Compute Limits (Vectorized)
    std_new = jnp.sqrt(jnp.maximum(new_vars, 1e-9))[:, None]
    std_all = jnp.sqrt(jnp.maximum(all_vars, 1e-9))[None, :]
    
    # max_cov_magnitude: (batch, n)
    max_cov_magnitude = (std_new * std_all) * correlation_limit

    # 3. Clip the Block
    current_block = cov[updated_indices, :]
    
    stabilized_block = jnp.clip(
        current_block, 
        -max_cov_magnitude, 
        max_cov_magnitude
    )
    
    batch_range = jnp.arange(updated_indices.shape[0])
    stabilized_block = stabilized_block.at[batch_range, updated_indices].set(new_vars)
    cov = cov.at[updated_indices, :].set(stabilized_block)
    cov = cov.at[:, updated_indices].set(stabilized_block.T)

    return cov

def _materialize_R_seq(R: Union[jnp.ndarray, callable], T: int, obs_dim: int) -> jnp.ndarray:
    """
    Materialize R sequence for all time steps.
    
    Args:
        R: Either constant matrix, time-varying sequence, or callable(t)->matrix
        T: Number of time steps
        obs_dim: Observation dimension
        
    Returns:
        R_seq: (T, obs_dim, obs_dim) tensor
    """
    if callable(R):
        Rs = jax.vmap(lambda t: R(t))(jnp.arange(T))
    else:
        Rs = R
    if Rs.ndim == 2:
        Rs = jnp.broadcast_to(Rs, (T, obs_dim, obs_dim))
    return Rs

@jax.jit
def truncated_normal_moments(mu, var, lower, upper):
    # Use a safe minimum variance to avoid division by zero
    std = jnp.sqrt(var)
    alpha = (lower - mu) / std
    beta = (upper - mu) / std
    
    flip = alpha > 0
    a = jnp.where(flip, -beta, alpha)
    b = jnp.where(flip, -alpha, beta)

    # 1. Compute Log-Probability of the interval (log Z) using stable bounds
    # Z = Phi(b) - Phi(a) (where a, b are now likely negative or crossing 0)
    log_Phi_a = jax.scipy.special.log_ndtr(a)
    log_Phi_b = jax.scipy.special.log_ndtr(b)
    
    # robust log_diff(x, y) = x + log(1 - exp(y - x)) for x > y
    log_Z = log_Phi_b + jnp.log1p(-jnp.exp(log_Phi_a - log_Phi_b))

    # 2. Compute log PDF values
    log_2pi = jnp.log(2 * jnp.pi)
    log_phi_a = -0.5 * a**2 - 0.5 * log_2pi
    log_phi_b = -0.5 * b**2 - 0.5 * log_2pi

    # 3. Compute Terms with log-space arithmetic
    term_a = jnp.exp(log_phi_a - log_Z)
    term_b = jnp.exp(log_phi_b - log_Z)
    
    # Lambda for the transformed interval
    lambda_internal = term_a - term_b
    
    # Correct lambda for flipping: lambda_original = -lambda_internal if flipped
    # Because E[X | alpha<X<beta] = -E[-X | -beta<-X<-alpha]
    lambda_ = jnp.where(flip, -lambda_internal, lambda_internal)
    
    mu_trunc = mu + std * lambda_
    
    # Handle infinite bounds zeroing
    term_a_prod = jnp.where(jnp.isinf(a), 0.0, a * term_a)
    term_b_prod = jnp.where(jnp.isinf(b), 0.0, b * term_b)
    
    var_scaling = 1.0 + (term_a_prod - term_b_prod) - lambda_internal**2
    
    # Numerical safety
    var_trunc = var * jnp.maximum(var_scaling, 0.0)
    
    return mu_trunc, var_trunc

def censored_kalman_update(
    m_pred, P_pred, y, H, R, lower_bounds, upper_bounds):
    """
    Censored Kalman update for observations with clip/censoring.
    
    This implements the correct Bayesian update for Tobit Type-I censoring:
    - y = a (lower bound): We observed z ≤ a, not a value
    - y = b (upper bound): We observed z ≥ b, not a value  
    - a < y < b: Standard Kalman update (uncensored)
    
    Uses truncated normal conditional distribution theory.
    
    Args:
        m_pred: Predicted state mean (n,)
        P_pred: Predicted state covariance (n, n)
        y: Observation vector (obs_dim,)
        H: Observation matrix (obs_dim, n)
        R: Observation noise covariance (obs_dim, obs_dim)
        lower_bounds: Lower bounds for censoring (obs_dim,)
        upper_bounds: Upper bounds for censoring (obs_dim,)
        
    Returns:
        Tuple of (updated_mean, updated_covariance)
    """

    mu_z = H @ m_pred
    S = H @ P_pred @ H.T + R

    lower = y <= lower_bounds
    upper = y >= upper_bounds
    uncensored = ~(lower | upper)

    lb = jnp.where(lower, lower_bounds, -jnp.inf)
    ub = jnp.where(upper, upper_bounds,  jnp.inf)
    lb = jnp.where(uncensored, y, lb)
    ub = jnp.where(uncensored, y, ub)

    mz, Sz = truncated_gaussian_moments(mu_z, S, lb, ub)

    K = P_pred @ H.T @ jnp.linalg.solve(S, jnp.eye(S.shape[0]))

    m_upd = m_pred + K @ (mz - mu_z)
    P_upd = P_pred - K @ (S - Sz) @ K.T

    return m_upd, P_upd


@jax.jit
def gaussian_product(mu_i, var_i, mu_j, var_j, cov_ij=0.0):
    """
    Compute product of two Gaussians using closed-form formula.
    
    For X ~ N(mu_i, var_i) and Y ~ N(mu_j, var_j) with Cov(X,Y) = cov_ij:
    Z = X * Y has:
        E[Z] = mu_i * mu_j + cov_ij
        Var[Z] = var_i * var_j + cov_ij² + var_i * mu_j² + var_j * mu_i² + 2 * cov_ij * mu_i * mu_j
    """
    mu_out = mu_i * mu_j + cov_ij
    var_out = (
        var_i * var_j +
        cov_ij**2 +
        var_i * mu_j**2 +
        var_j * mu_i**2 +
        2 * cov_ij * mu_i * mu_j
    )
    return mu_out, var_out  
    