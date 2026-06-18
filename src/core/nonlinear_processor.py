"""
Unified Nonlinear State Processor for JAX-based State-Space Models.

This module provides a bridge between SSMparam's detected nonlinearities
and CloseFormMoments' exact moment matching capabilities, enabling
efficient uncertainty propagation for nonlinear transformations.

Key Features:
- JAX-compatible PyTree structures for JIT compilation
- Optimized batch processing of transformations by type
- Seamless integration with existing SSMparam and CloseFormMoments classes
- Flexible compile-once, run-many workflow for optimal performance
- Simplified implementation - batch methods handle empty arrays gracefully
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import List, Dict, NamedTuple, Tuple, Optional, Union

# Import the base classes we need to extend
from .close_form import CloseFormMoments
from .close_form import CloseFormMoments
# from .state_models import SSMparam  <-- Removed to avoid circular import


class TransformationConfig(NamedTuple):
    """
    A JAX-compatible Pytree to store indices for batch processing.
    This freezes the Python list structure into JAX arrays for JIT compilation.
    """
    # Sin transformation indices
    sin_in: jnp.ndarray
    sin_out: jnp.ndarray
    
    # Exponential transformation indices
    exp_in: jnp.ndarray
    exp_out: jnp.ndarray
    
    # Clip transformation indices and bounds
    clip_in: jnp.ndarray
    clip_out: jnp.ndarray
    clip_low: jnp.ndarray
    clip_high: jnp.ndarray
    
    # Metadata for validation
    n_sin: int
    n_exp: int
    n_clip: int
    total_transformations: int
    
    # Flag to indicate if observation nonlinearities are present
    has_observation_nonlinearities: bool


class NonlinearStateProcessor(CloseFormMoments):
    """
    Extends CloseFormMoments to seamlessly integrate with SSMparam's
    nonlinearity detection and provide optimized batch processing.
    
    This class bridges the gap between SSMparam's symbolic analysis
    and CloseFormMoments' numerical moment matching, enabling efficient
    uncertainty propagation for nonlinear state-space models.
    
    Example Usage:
        ```python
        # Setup Phase (run once)
        ssm_model = SSMparam(...)
        processor = NonlinearStateProcessor()
        
        trans_config = processor.compile_config(
            ssm_model.get_nonlinearities()['transition']
        )
        obs_config = processor.compile_config(
            ssm_model.get_nonlinearities()['observation']
        )
        
        # Runtime Phase (JIT-compiled, run many times)
        mu, cov = processor.process_transition_nonlinearities(mu, cov, trans_config)
        m_y, P_y, cov_xy = processor.process_observation_nonlinearities(
            m_y, P_y, cov_xy, obs_config
        )
        ```
    """
    
    def __init__(self):
        """Initialize the processor with inherited CloseFormMoments capabilities."""
        super().__init__()
    
    def compile_config(self, nonlinear_map_list: List[Dict], is_observation: bool = False) -> TransformationConfig:
        """
        Convert SSMparam's nonlinearity list into JAX-compatible TransformationConfig.
        
        This method should be called once before the main filtering loop.
        It converts Python lists into JAX arrays that can be used in JIT-compiled code.
        
        Args:
            nonlinear_map_list: List of nonlinearity dictionaries from SSMparam.get_nonlinearities()
            is_observation: Whether this is for observation (True) or transition (False)
            
        Returns:
            TransformationConfig: JAX-compatible PyTree with all transformation indices
        """
        # Temporary storage for each transformation type
        sin_i, sin_o = [], []
        exp_i, exp_o = [], []
        clip_i, clip_o, clip_l, clip_h = [], [], [], []

        if nonlinear_map_list is None:
            nonlinear_map_list = []

        # Process each nonlinearity entry
        for entry in nonlinear_map_list:
            t_type = entry['transformation']
            
            # Extract input and output indices
            # SSMparam uses different keys for transition vs observation nonlinearities
            idx_in = entry['input_state_index']
            idx_out = entry.get('output_state_index', entry.get('output_obs_index'))

            # Skip invalid entries
            if idx_in is None or idx_out is None:
                continue

            # Categorize by transformation type
            if t_type == 'sin':
                sin_i.append(idx_in)
                sin_o.append(idx_out)
            elif t_type == 'exp':
                exp_i.append(idx_in)
                exp_o.append(idx_out)
            elif t_type == 'clip':
                clip_i.append(idx_in)
                clip_o.append(idx_out)
                clip_l.append(entry.get('lower_bound', -jnp.inf))
                clip_h.append(entry.get('upper_bound', jnp.inf))
            else:
                # Unknown transformation type - skip but could raise warning
                continue

        # Helper function to create safe JAX arrays
        def safe_array(arr, dtype=jnp.int32):
            """Create JAX array, ensuring non-empty for JAX compatibility."""
            return jnp.array(arr, dtype=dtype) if len(arr) > 0 else jnp.empty((0,), dtype=dtype)

        # Create the TransformationConfig
        config = TransformationConfig(
            sin_in=safe_array(sin_i),
            sin_out=safe_array(sin_o),
            exp_in=safe_array(exp_i),
            exp_out=safe_array(exp_o),
            clip_in=safe_array(clip_i),
            clip_out=safe_array(clip_o),
            clip_low=safe_array(clip_l, jnp.float32),
            clip_high=safe_array(clip_h, jnp.float32),
            n_sin=len(sin_i),
            n_exp=len(exp_i),
            n_clip=len(clip_i),
            total_transformations=len(sin_i) + len(exp_i) + len(clip_i),
            has_observation_nonlinearities=is_observation and (len(sin_i) + len(exp_i) + len(clip_i) > 0)
        )
        
        return config

    @partial(jax.jit, static_argnums=(0,))
    def process_transition_nonlinearities(self, 
                                         mu: jnp.ndarray, 
                                         cov: jnp.ndarray, 
                                         config: TransformationConfig) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Apply state transition nonlinearities to the state mean and covariance.
        ...
        """
        # --- FIX: Ensure inputs are correct dimensionality ---
        mu = jnp.atleast_1d(mu)
        cov = jnp.atleast_2d(cov)
        # ---------------------------------------------------

        # 1. Apply SIN transformations (only if present)
        if len(config.sin_in) > 0:
            mu, cov, _ = self.apply_batch_sin_transforms(mu, cov, config.sin_in, config.sin_out, None)

        # 2. Apply EXP transformations (only if present)
        if len(config.exp_in) > 0:
            mu, cov, _ = self.apply_batch_exp_transforms(mu, cov, config.exp_in, config.exp_out, None)

        # 3. Apply CLIP transformations (only if present)
        if len(config.clip_in) > 0:
            mu, cov, _ = self.apply_batch_clip_transforms(
                mu, cov, 
                config.clip_in, config.clip_out, 
                config.clip_low, config.clip_high,
                None
            )
        
        return mu, cov

    @partial(jax.jit, static_argnums=(0,))
    def process_observation_nonlinearities(self, 
                                      mu: jnp.ndarray, 
                                      cov: jnp.ndarray,
                                      cov_xy: jnp.ndarray,
                                      config: TransformationConfig) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Apply observation nonlinearities to the observation moments.
        
        This function processes transformations for observations where
        cov_xy scaling is needed for observation cross-covariance.
        
        Args:
            mu: Observation mean vector (M,)
            cov: Observation covariance matrix (M, M)
            cov_xy: Cross-covariance between observations and states (M, N)
            config: TransformationConfig from compile_config()
            
        Returns:
            Tuple of (updated_mu, updated_cov, updated_cov_xy)
        """
        # --- FIX: Ensure inputs are correct dimensionality ---
        mu = jnp.atleast_1d(mu)
        cov = jnp.atleast_2d(cov)
        cov_xy = jnp.atleast_2d(cov_xy)
        # ---------------------------------------------------
        
        # Process transformations in optimal order for numerical stability
        
        # 1. Apply SIN transformations (only if present)
        if len(config.sin_in) > 0:
            mu, cov, cov_xy = self.apply_batch_sin_transforms(mu, cov, config.sin_in, config.sin_out, cov_xy)

        # 2. Apply EXP transformations (only if present)
        if len(config.exp_in) > 0:
            mu, cov, cov_xy = self.apply_batch_exp_transforms(mu, cov, config.exp_in, config.exp_out, cov_xy)

        # 3. Apply CLIP transformations (only if present)
        if len(config.clip_in) > 0:
            mu, cov, cov_xy = self.apply_batch_clip_transforms(
                mu, cov, 
                config.clip_in, config.clip_out, 
                config.clip_low, config.clip_high,
                cov_xy
            )
        
        return mu, cov, cov_xy

    #@partial(jax.jit, static_argnums=(0,)) # Removed JIT to handle conditional logic with tracers
    def process_nonlinearities(self, 
                               mu: jnp.ndarray, 
                               cov: jnp.ndarray, 
                               config: TransformationConfig,
                               cov_xy: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]]:
        """
        Legacy unified function that processes both transition and observation nonlinearities.
        
        This function maintains backward compatibility with the original interface.
        
        Args:
            mu: State or observation mean vector (N,)
            cov: State or observation covariance matrix (N, N)
            config: TransformationConfig from compile_config()
            cov_xy: Optional cross-covariance matrix
            
        Returns:
            Tuple of (updated_mu, updated_cov, updated_cov_xy)
        """
        # Use jax.lax.cond to handle the branching within JIT
        # We need to define the two branches as functions
        
        def obs_branch(operands):
            mu, cov, config, cov_xy = operands
            # If cov_xy is None (which shouldn't happen in this branch due to logic below, 
            # but we need to handle types), we create a zero one.
            # However, jax.lax.cond requires operands to be consistent.
            # The issue is cov_xy can be None in the input.
            
            # Let's simplify: if cov_xy is None, we make a zero one.
            cov_xy_safe = jnp.zeros((mu.shape[0], cov.shape[0])) if cov_xy is None else cov_xy
            return self.process_observation_nonlinearities(mu, cov, cov_xy_safe, config)

        def trans_branch(operands):
            mu, cov, config, cov_xy = operands
            mu_new, cov_new = self.process_transition_nonlinearities(mu, cov, config)
            return mu_new, cov_new, None

        
        is_obs = config.has_observation_nonlinearities
        if cov_xy is not None:
             is_obs = True
        
        if cov_xy is not None or config.has_observation_nonlinearities:
            # Use observation processing
            if cov_xy is None:
                # Create dummy cov_xy for observation processing
                cov_xy = jnp.zeros((mu.shape[0], cov.shape[0]))
            return self.process_observation_nonlinearities(mu, cov, cov_xy, config)
        else:
            # Use transition processing
            mu, cov = self.process_transition_nonlinearities(mu, cov, config)
            return mu, cov, None
    
    def process_nonlinearities_with_mask(self,
                                         mu: jnp.ndarray,
                                         cov: jnp.ndarray,
                                         config: TransformationConfig,
                                         apply_sin: bool = True,
                                         apply_exp: bool = True,
                                         apply_clip: bool = True,
                                         cov_xy: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]]:
        """
        Process nonlinearities with optional masking for selective application.
        
        This method provides fine-grained control over which transformation types
        are applied, useful for debugging or specialized algorithms.
        
        Args:
            mu: State mean vector (N,)
            cov: State covariance matrix (N, N)
            config: TransformationConfig from compile_config()
            apply_sin: Whether to apply sin transformations
            apply_exp: Whether to apply exp transformations
            apply_clip: Whether to apply clip transformations
            cov_xy: Optional cross-covariance matrix
            
        Returns:
            Tuple of (updated_mu, updated_cov, updated_cov_xy)
        """
        # Choose processing method based on whether cov_xy is provided
        if cov_xy is not None or config.has_observation_nonlinearities:
            # Use observation processing methods
            if apply_sin and len(config.sin_in) > 0:
                mu, cov, cov_xy = self.apply_batch_sin_transforms(mu, cov, config.sin_in, config.sin_out, cov_xy)
            
            if apply_exp and len(config.exp_in) > 0:
                mu, cov, cov_xy = self.apply_batch_exp_transforms(mu, cov, config.exp_in, config.exp_out, cov_xy)
            
            if apply_clip and len(config.clip_in) > 0:
                mu, cov, cov_xy = self.apply_batch_clip_transforms(
                    mu, cov, 
                    config.clip_in, config.clip_out, 
                    config.clip_low, config.clip_high,
                    cov_xy
                )
            
            return mu, cov, cov_xy
        else:
            # Use transition processing methods (no cov_xy)
            if apply_sin and len(config.sin_in) > 0:
                mu, cov, _ = self.apply_batch_sin_transforms(mu, cov, config.sin_in, config.sin_out, None)
            
            if apply_exp and len(config.exp_in) > 0:
                mu, cov, _ = self.apply_batch_exp_transforms(mu, cov, config.exp_in, config.exp_out, None)
            
            if apply_clip and len(config.clip_in) > 0:
                mu, cov, _ = self.apply_batch_clip_transforms(
                    mu, cov, 
                    config.clip_in, config.clip_out, 
                    config.clip_low, config.clip_high,
                    None
                )
            
            return mu, cov, None
    
    def create_configs_from_ssm(self, ssm_model) -> Tuple[TransformationConfig, TransformationConfig]:
        """
        Convenience method to create both transition and observation configs from an SSMparam.
        
        Args:
            ssm_model: An instance of SSMparam with nonlinearities
            
        Returns:
            Tuple of (trans_config, obs_config)
        """
        nonlinearities = ssm_model.get_nonlinearities()
        
        trans_config = self.compile_config(nonlinearities['transition'], is_observation=False)
        obs_config = self.compile_config(nonlinearities['observation'], is_observation=True)
        
        return trans_config, obs_config
    
    def validate_config(self, config: TransformationConfig, state_dim: int) -> bool:
        """
        Validate that a TransformationConfig is compatible with the state dimension.
        
        Args:
            config: TransformationConfig to validate
            state_dim: Expected state dimension
            
        Returns:
            True if valid, False otherwise
        """
        # Check that all indices are within bounds
        all_in_indices = jnp.concatenate([config.sin_in, config.exp_in, config.clip_in])
        all_out_indices = jnp.concatenate([config.sin_out, config.exp_out, config.clip_out])
        
        all_indices = jnp.concatenate([all_in_indices, all_out_indices])
        
        # Check if any index is out of bounds
        max_idx = jnp.max(all_indices) if len(all_indices) > 0 else -1
        min_idx = jnp.min(all_indices) if len(all_indices) > 0 else 0
        
        valid_indices = (min_idx >= 0) & (max_idx < state_dim)
        
        # Check that arrays have consistent lengths
        sin_consistent = len(config.sin_in) == len(config.sin_out)
        exp_consistent = len(config.exp_in) == len(config.exp_out)
        clip_consistent = (len(config.clip_in) == len(config.clip_out) == 
                           len(config.clip_low) == len(config.clip_high))
        
        return bool(valid_indices & sin_consistent & exp_consistent & clip_consistent)
    
    def get_config_summary(self, config: TransformationConfig) -> Dict:
        """
        Get a human-readable summary of the transformation configuration.
        
        Args:
            config: TransformationConfig to summarize
            
        Returns:
            Dictionary with configuration summary
        """
        return {
            'total_transformations': config.total_transformations,
            'sin_transformations': config.n_sin,
            'exp_transformations': config.n_exp,
            'clip_transformations': config.n_clip,
            'has_transformations': config.total_transformations > 0,
            'has_observation_nonlinearities': config.has_observation_nonlinearities
        }


# Helper functions removed to avoid circular dependency with SSMparam
# SSMparam will now inherit from NonlinearStateProcessor directly.


if __name__ == "__main__":
    # Example usage and testing
    
    print("=" * 60)
    print("NONLINEAR STATE PROCESSOR DEMO")
    print("=" * 60)
    
    # Create a simple test case
    processor = NonlinearStateProcessor()
    
    # Test transition configuration creation
    test_transition_nonlinearities = [
        {'transformation': 'sin', 'input_state_index': 0, 'output_state_index': 2},
        {'transformation': 'exp', 'input_state_index': 1, 'output_state_index': 3},
        {'transformation': 'clip', 'input_state_index': 2, 'output_state_index': 4, 
         'lower_bound': -1.0, 'upper_bound': 1.0}
    ]
    
    trans_config = processor.compile_config(test_transition_nonlinearities, is_observation=False)
    trans_summary = processor.get_config_summary(trans_config)
    
    print(f"Transition Configuration Summary: {trans_summary}")
    
    # Test observation configuration creation
    test_observation_nonlinearities = [
        {'transformation': 'sin', 'input_state_index': 0, 'output_obs_index': 0},
        {'transformation': 'exp', 'input_state_index': 1, 'output_obs_index': 1},
        {'transformation': 'clip', 'input_state_index': 2, 'output_obs_index': 2, 
         'lower_bound': -1.0, 'upper_bound': 1.0}
    ]
    
    obs_config = processor.compile_config(test_observation_nonlinearities, is_observation=True)
    obs_summary = processor.get_config_summary(obs_config)
    
    print(f"Observation Configuration Summary: {obs_summary}")
    
    # Test validation
    state_dim = 5
    trans_valid = processor.validate_config(trans_config, state_dim)
    obs_valid = processor.validate_config(obs_config, state_dim)
    print(f"Transition config valid for state_dim={state_dim}: {trans_valid}")
    print(f"Observation config valid for state_dim={state_dim}: {obs_valid}")
    
    # Test transition processing
    mu = jnp.array([0.5, 1.0, 0.0, 2.0, 0.5])
    cov = jnp.eye(5) * 0.1
    
    print(f"\nOriginal mean: {mu}")
    print(f"Original cov diag: {jnp.diag(cov)}")
    
    mu_trans, cov_trans = processor.process_transition_nonlinearities(mu, cov, trans_config)
    
    print(f"\nAfter transition processing:")
    print(f"Mean: {mu_trans}")
    print(f"Cov diag: {jnp.diag(cov_trans)}")
    
    # Test observation processing
    obs_dim = 3
    mu_obs = jnp.array([0.5, 1.0, 0.0])
    cov_obs = jnp.eye(obs_dim) * 0.1
    cov_xy = jnp.zeros((obs_dim, state_dim)) + 0.05  # Some cross-covariance
    
    print(f"\nOriginal obs mean: {mu_obs}")
    print(f"Original obs cov diag: {jnp.diag(cov_obs)}")
    print(f"Original cov_xy: {cov_xy}")
    
    mu_obs_proc, cov_obs_proc, cov_xy_proc = processor.process_observation_nonlinearities(
        mu_obs, cov_obs, cov_xy, obs_config
    )
    
    print(f"\nAfter observation processing:")
    print(f"Mean: {mu_obs_proc}")
    print(f"Cov diag: {jnp.diag(cov_obs_proc)}")
    print(f"Updated cov_xy: {cov_xy_proc}")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("=" * 60)