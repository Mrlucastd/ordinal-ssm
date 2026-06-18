import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from typing import List, Union, Dict, Any, Tuple
from .agvi import agvi_batch_jax
from .utils import truncated_normal_moments

@dataclass
class InspectorState:
    """
    Storage for AGVI estimates of multiple inspectors.
    All arrays are JAX arrays of shape (N_inspectors, 1).
    """
    means: jnp.ndarray
    vars: jnp.ndarray
    update_counts: np.ndarray 
    ids: List[Any]
    id_to_idx: Dict[Any, int]

class InspectorManager:
    """
    Manages a bank of AGVI estimators for a set of inspectors.
    Handles mapping from discrete Inspector IDs to continuous JAX array indices.
    """
    # Precision bounds for inspector standard deviation [0.025, 0.15]
    # Stored as variance limits: [0.000625, 0.0225]
    MIN_VAR = 0.025**2
    MAX_VAR = 0.15**2
    
    def __init__(self, inspector_ids: List[Any], init_std: float = 0.1, init_var: float = 0.01):
        """
        Initialize the manager with a list of inspector IDs.
        
        Args:
            inspector_ids: List of unique identifiers for inspectors (strings, ints, etc.)
            init_std: Initial guess for standard deviation (stored as variance = sigma^2).
            init_var: Initial uncertainty (variance of the variance).
        """
        self.n_inspectors = len(inspector_ids)
        self.ids = inspector_ids
        
        # Create mapping from ID to Index for fast lookup
        self.id_to_idx = {uid: i for i, uid in enumerate(self.ids)}
        
        # Initialize JAX arrays
        # Shape (N, 1)
        # Use initial guess mu=init_std**2 and uncertainty=init_var
        # This matches the user's required initialization logic
        init_means = jnp.ones((self.n_inspectors, 1)) * (init_std**2)
        mu0, var0 = truncated_normal_moments(init_means, init_var, self.MIN_VAR, self.MAX_VAR)
        
        self.state = InspectorState(
            means=mu0,
            vars=var0,
            update_counts=np.zeros(self.n_inspectors, dtype=int),
            ids=self.ids,
            id_to_idx=self.id_to_idx
        )
        
    def _get_indices(self, active_ids: Union[List[Any], np.ndarray]) -> jnp.ndarray:
        """Helper to convert external IDs to internal indices."""
        # Handle numpy or list input
        if isinstance(active_ids, np.ndarray):
             active_ids = active_ids.tolist()
             
        indices = [self.state.id_to_idx[uid] for uid in active_ids]
        return jnp.array(indices, dtype=jnp.int32)

    def get_R_seq(self, active_ids: Union[List[Any], np.ndarray], T: int) -> jnp.ndarray:
        """
        Constructs the Priors for R (Observation Variance) for a time series.
        
        Args:
            active_ids: List or Array of Inspector IDs active at each time step (length T).
            T: Length of time series.
            
        Returns:
            R_seq: JAX array of shape (T, 1, 1).
        """
        indices = self._get_indices(active_ids)
        
        # Look up current estimates
        # indices shape (T,) -> means shape (T, 1)
        current_vars_seq = self.state.means[indices, 0] 
        
        # Reshape to (T, 1, 1) for Kalman Filter
        return current_vars_seq.reshape(T, 1, 1)
    
    def update_batch(self, 
                     mr_flat: jnp.ndarray, 
                     pr_flat: jnp.ndarray, 
                     active_ids: Union[List[Any], np.ndarray],
                     valid_mask: jnp.ndarray):
        """
        Performs a batched, vectorized AGVI update for all inspectors involved in the series.
        
        Args:
            mr_flat: Posterior Means of R from KF (T,)
            pr_flat: Posterior Variances of R from KF (T,)
            active_ids: List/Array of Inspector IDs at each step (T,)
            valid_mask: Boolean mask of valid updates (T,) - False if observation was NaN
        """
        T = len(valid_mask)
        indices = self._get_indices(active_ids) # (T,)
        
        # 1. Construct Dense Update Matrix (T, N_inspectors)
        # Initialize with NaNs (no update)
        batch_means_in = jnp.full((T, self.n_inspectors), jnp.nan)
        batch_vars_in  = jnp.full((T, self.n_inspectors), jnp.nan)
        
        # 2. Scatter valid updates into the matrix
        row_indices = jnp.arange(T)
        
        # Filter only valid steps
        valid_rows = row_indices[valid_mask]
        valid_cols = indices[valid_mask]
        
        # Ensure 1D shapes for assignment
        valid_means = mr_flat[valid_mask]
        valid_vars  = pr_flat[valid_mask]
        
        if len(valid_rows) > 0:
            batch_means_in = batch_means_in.at[valid_rows, valid_cols].set(valid_means)
            batch_vars_in  = batch_vars_in.at[valid_rows, valid_cols].set(valid_vars)
            
            # Update counts
            # valid_cols is a JAX array of indices that received an update
            # We convert to numpy to update the numpy counter state
            v_cols_np = np.array(valid_cols)
            counts = np.bincount(v_cols_np, minlength=self.n_inspectors)
            self.state.update_counts += counts
            
        # 3. Current State flattened for AGVI
        curr_mu_all = self.state.means[:, 0]
        curr_var_all = self.state.vars[:, 0]
        
        # 4. Run Vectorized AGVI
        mu_new_all, var_new_all = agvi_batch_jax(
            batch_means_in, 
            batch_vars_in, 
            curr_mu_all, 
            curr_var_all, 
            return_history=False
        )
        
        # 5. Update Internal State
        # Apply truncation to the estimated variance distribution
        mu_trunc, var_trunc = truncated_normal_moments(
            mu_new_all, var_new_all, self.MIN_VAR, self.MAX_VAR
        )
        self.state.means = mu_trunc[:, None]
        self.state.vars  = var_new_all[:, None]
        
    def get_R_batch(self, batch_inspector_ids: List[np.ndarray], T: int) -> jnp.ndarray:
        """
        Constructs the Priors for R for a whole batch of sequences.
        Handles invalid/padding IDs by checking against id_to_idx.
        
        Args:
            batch_inspector_ids: List of shape (B,) where each element is an array of IDs of length T.
            T: Length of time series.
            
        Returns:
            R_batch: JAX array of shape (B, T, 1, 1).
        """
        B = len(batch_inspector_ids)
        batch_indices = []
        
        # We need a fallback safe index for padding/invalid IDs
        # We assume at least one ID exists if manager initialized
        safe_idx = 0 
        
        for b in range(B):
            insp_seq = batch_inspector_ids[b]
            seq_indices = []
            for t in range(T):
                uid = insp_seq[t]
                if uid in self.state.id_to_idx:
                    seq_indices.append(self.state.id_to_idx[uid])
                else:
                    seq_indices.append(safe_idx)
            batch_indices.append(seq_indices)
            
        # (B, T)
        indices_arr = jnp.array(batch_indices, dtype=jnp.int32)
        
        # Look up
        # (B, T) -> (B, T, 1)
        R_vals = self.state.means[indices_arr, 0]
        
        return R_vals.reshape(B, T, 1, 1)

    def update_batch_from_results(self, 
                                  mr_post_batch: jnp.ndarray, 
                                  pr_post_batch: jnp.ndarray, 
                                  batch_inspector_ids: List[np.ndarray], 
                                  valid_mask_batch: jnp.ndarray):
        """
        Handles flattening and masking for batched KF results update.
        
        Args:
            mr_post_batch: (B, T, 1)
            pr_post_batch: (B, T, 1, 1) or (B, T, 1)
            batch_inspector_ids: List (B,) of (T,) ID arrays
            valid_mask_batch: (B, T) bool array (from Y nans)
        """
        # Flatten Results
        mr_flat = mr_post_batch.reshape(-1)
        pr_flat = pr_post_batch.reshape(-1)
        
        valid_mask_flat = np.array(valid_mask_batch.reshape(-1))
        
        # Flatten IDs and Refine Mask for Invalid IDs
        flat_ids = []
        
        B = len(batch_inspector_ids)
        T = valid_mask_batch.shape[1]
        
        counter = 0
        
        # We need a dummy ID to keep list length consistent with flat arrays
        dummy_id = self.ids[0]
        
        for b in range(B):
            insp_seq = batch_inspector_ids[b]
            for t in range(T):
                uid = insp_seq[t]
                if uid in self.state.id_to_idx:
                    flat_ids.append(uid)
                else:
                    # Invalid/Padding ID
                    flat_ids.append(dummy_id)
                    valid_mask_flat[counter] = False # Invalidate update
                
                counter += 1
                
        # Call core update
        self.update_batch(mr_flat, pr_flat, flat_ids, jnp.array(valid_mask_flat))
