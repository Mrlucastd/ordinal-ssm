"""
State-space model parameter structures for JAX-based filtering.

This module defines the SSMparam class which serves as a JAX Pytree-compatible
parameter structure for state-space models with a function-based API,
automatic linearization, and intelligent state-lifting.
"""

from typing import List, Callable, Dict, Optional, Tuple
import inspect, textwrap, re
import sympy as sp
import numpy as np
from math import factorial
from dataclasses import dataclass
from dataclasses import dataclass
import jax.numpy as jnp
from .nonlinear_processor import NonlinearStateProcessor


@dataclass
class JaxSSMParams:
    """
    JAX-compatible parameter structure for gradient-based learning.
    
    This dataclass contains the numerical JAX arrays needed for Kalman filtering,
    separated from the symbolic SSMparam structure to enable JIT compilation
    and automatic differentiation.
    
    Attributes:
        F: State transition matrix (state_dim, state_dim)
        Q: Process noise covariance (state_dim, state_dim)
        H: Observation matrix (obs_dim, state_dim)
        B: Control matrix (state_dim, control_dim) or None
        m0: Initial state mean (state_dim,)
        P0: Initial state covariance (state_dim, state_dim)
    """
    F: jnp.ndarray
    Q: jnp.ndarray
    H: jnp.ndarray
    B: Optional[jnp.ndarray]
    m0: jnp.ndarray
    P0: jnp.ndarray


# --- Custom SymPy Clip Function ---------------------------------

class clip(sp.Function):
    """
    Custom symbolic sp.Function for 'clip' to allow for
    symbolic differentiation (for A matrix) while remaining
    detectable for state lifting.
    """
    @classmethod
    def eval(cls, x, lower, upper):
        # Don't evaluate - keep it symbolic
        return None
    
    def fdiff(self, argindex=1):
        # The derivative of clip(x, a, b) w.r.t x is a step function.
        # For a standard Kalman Filter, we linearize at a point,
        # but for a general-purpose A matrix, assuming 0 is the
        # safest approach for auxiliary state lifting.
        # This function returning 0 is KEY for the A matrix.
        return sp.Integer(0)
    
    def _eval_evalf(self, prec=None, **options):
        # Prevent evaluation
        return None

# --- Symbolic Helper Functions ---------------------------------

def _sympify_source(func: Callable) -> str:
    """Extract and sympify function source code for symbolic processing."""
    src = inspect.getsource(func)
    src = textwrap.dedent(src)
    # Rename function to a standard temp name
    src = re.sub(r"def\s+(\w+)\s*\(", r"def _symbolic_func(", src, count=1)
    # Replace JAX/Numpy/Math with SymPy
    src = src.replace("jnp.", "sp.").replace("np.", "sp.").replace("math.", "sp.")
    # Replace sp.clip with our custom symbolic class
    # Use regex to ensure we only replace sp.clip, not other clips
    src = re.sub(r"sp\.clip", "clip", src)
    return src


def _linearize_transition(
    transition_func: Callable,
    primary_state_names: List[str],
    dt_val: float,
    lift_nonlinear: bool = True,
    nonlinear_funcs: Tuple[str] = ("exp", "tanh", "clip", "sin", "cos"),
    tol: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict]:
    """
    Symbolic linearizer that calculates full gradients for dynamics, 
    but masks (zeros out) rows for pure storage states.
    """
    
    # --- Step 0: Define Clip with Gradient = 1 ---
    # We allow the gradient to flow so that dynamics (x = x + clip(v)*dt) 
    # are calculated correctly in the F matrix.
    class trans_clip(sp.Function):
        @classmethod
        def eval(cls, x, lower, upper):
            return None
        
        def fdiff(self, argindex=1):
            # Return 1 so dynamics are captured. 
            # We will manually zero out storage rows later.
            return sp.Integer(1)

    trans_clip.__name__ = "clip"

    # --- Step 1: Parse user function ---
    src = _sympify_source(transition_func)
    
    symbolic_globals = {
        "sp": sp,
        "clip": trans_clip, 
        "exp": sp.exp, "sin": sp.sin, "cos": sp.cos,
        "tanh": sp.tanh, "log": sp.log, "sqrt": sp.sqrt, "Abs": sp.Abs,
        "__builtins__": {"float": float, "int": int}
    }
    exec(src, symbolic_globals)
    symbolic_func = symbolic_globals["_symbolic_func"]

    # --- Step 2: Prepare symbols ---
    primary_state_syms = [sp.symbols(n) for n in primary_state_names]
    dt_sym = sp.symbols("dt")
    noise_val_sym = sp.symbols("noise_val") 

    # --- Step 3: Evaluate transition ---
    # Check if 'noise' argument exists in the original function signature
    sig = inspect.signature(transition_func)
    has_noise_param = 'noise' in sig.parameters

    if has_noise_param:
        symbolic_result = symbolic_func(*primary_state_syms, dt_sym, noise_val_sym)
        symbolic_result_list = list(symbolic_result)
        state_expressions_list = symbolic_result_list[:-1] # Exclude noise return if it exists
    else:
        # Call without noise argument
        symbolic_result = symbolic_func(*primary_state_syms, dt_sym)
        symbolic_result_list = list(symbolic_result)
        # Assume all returns are states if no noise param
        state_expressions_list = symbolic_result_list 

    if len(state_expressions_list) != len(primary_state_syms):
        raise ValueError(f"State count mismatch in {transition_func.__name__}")

    state_expressions_orig = sp.Matrix(tuple(state_expressions_list))

    # --- Step 4 & 5: Detect and lift nonlinearities ---
    auxiliary_state_syms, nonlinear_map = [], {}
    aux_replacements = {}
    pre_lifted_states = {} 
    embedded_funcs = [] 
    
    # *** NEW: Track indices of pure storage states to zero them out later ***
    storage_state_indices = [] 

    if lift_nonlinear:
        # Part A: Find pre-lifted states (Pure Storage)
        for i, expr in enumerate(state_expressions_orig):
            fname = expr.func.__name__ if isinstance(expr, sp.Function) else None
            
            # If the state definition is purely a nonlinear function (e.g., clip_x = clip(x))
            if fname in nonlinear_funcs:
                # Mark this index as a storage state
                storage_state_indices.append(i)
                
                # (Standard lifting logic...)
                state_sym = primary_state_syms[i]
                pre_lifted_states[expr] = state_sym
                aux_replacements[expr] = state_sym

                arg = expr.args[0] if expr.args else "unknown"
                input_idx = None
                if arg in primary_state_syms:
                    input_idx = primary_state_syms.index(arg)
                else:
                    for j, sym in enumerate(primary_state_syms):
                        if sym in arg.free_symbols:
                            input_idx = j
                            break 
                
                nl_entry = {
                    "input_state_index": input_idx,
                    "output_state_index": i,
                    "transformation": fname.lower(),
                }
                if fname == "clip" and len(expr.args) == 3:
                    try:
                        nl_entry["lower_bound"] = float(expr.args[1].evalf())
                        nl_entry["upper_bound"] = float(expr.args[2].evalf())
                    except Exception: pass

                nonlinear_map[len(nonlinear_map)] = nl_entry

        # Part B: Embedded functions (Dynamics)
        for expr in state_expressions_orig:
            if expr in pre_lifted_states: continue
            for sub_expr in sp.preorder_traversal(expr):
                if sub_expr in pre_lifted_states or sub_expr in embedded_funcs:
                    continue
                fname = sub_expr.func.__name__ if isinstance(sub_expr, sp.Function) else None
                if fname in nonlinear_funcs:
                    embedded_funcs.append(sub_expr)

        # Part C: Create new auxiliary states
        for i, sub_expr in enumerate(embedded_funcs, start=1):
            fname = sub_expr.func.__name__
            arg = sub_expr.args[0] if sub_expr.args else "unknown"
            
            input_idx = None
            if arg in primary_state_syms:
                input_idx = primary_state_syms.index(arg)
            else:
                for j, sym in enumerate(primary_state_syms):
                    if sym in arg.free_symbols:
                        input_idx = j
                        break
            
            aux_name = f"{fname}_{str(arg)}"
            if fname == "clip":
                try:
                    # Try to extract bounds if available
                    if len(sub_expr.args) == 3:
                        lower = float(sub_expr.args[1].evalf())
                        upper = float(sub_expr.args[2].evalf())
                        aux_name = f"clip_{str(arg)}_{lower}_{upper}"
                    else:
                        aux_name = f"clip_{str(arg)}_aux"
                except:
                    aux_name = f"clip_aux_{len(auxiliary_state_syms)}"
            
            z = sp.symbols(aux_name)
            auxiliary_state_syms.append(z)
            aux_replacements[sub_expr] = z

            nl_entry = {
                "input_state_index": input_idx,
                "output_state_index": len(primary_state_syms) + len(auxiliary_state_syms) - 1,
                "transformation": fname.lower(),
            }
            if fname == "clip" and len(sub_expr.args) == 3:
                try:
                    nl_entry["lower_bound"] = float(sub_expr.args[1].evalf())
                    nl_entry["upper_bound"] = float(sub_expr.args[2].evalf())
                except: pass
            
            nonlinear_map[len(nonlinear_map)] = nl_entry
            
    # --- Step 6: Build augmented system ---
    if aux_replacements:
        state_expressions_lifted = state_expressions_orig.xreplace(aux_replacements)
    else:
        state_expressions_lifted = state_expressions_orig

    all_state_syms = primary_state_syms + auxiliary_state_syms
    all_state_names = [str(s) for s in all_state_syms]
    
    # Zeros for aux states
    zero_expressions = [sp.Integer(0) for _ in auxiliary_state_syms]
    augmented_expressions_list = list(state_expressions_lifted) + zero_expressions
    augmented_expressions_matrix = sp.Matrix(augmented_expressions_list)

    # --- Step 7: Compute Jacobian (F Matrix) ---
    F_symbolic = augmented_expressions_matrix.jacobian(all_state_syms)
    F_symbolic_subbed = F_symbolic.subs({dt_sym: dt_val})
    
    F_numeric = np.zeros(F_symbolic_subbed.shape, dtype=float)
    for i in range(F_symbolic_subbed.shape[0]):
        for j in range(F_symbolic_subbed.shape[1]):
            expr = F_symbolic_subbed[i, j]
            try:
                if hasattr(expr, 'evalf'):
                    F_numeric[i, j] = float(expr.evalf())
                else:
                    F_numeric[i, j] = float(expr)
            except (TypeError, ValueError, AttributeError):
                F_numeric[i, j] = 0.0

    # *** STEP 7.5: APPLY STORAGE MASK ***
    # We manually zero out the rows corresponding to pure storage states.
    # This allows dynamics (using clip) to have gradients, while storage states stay 0.
    for idx in storage_state_indices:
        F_numeric[idx, :] = 0.0

    # --- Step 8: Detect integration chains for Q ---
    n_primary = len(primary_state_syms)
    n_total = len(all_state_syms)
    Q_numeric = np.zeros((n_total, n_total))

    # Always use unit noise for structural Q calculation
    # Magnitude should be handled at runtime (e.g. process_error in KF)
    noise_val = 1.0

    i = 0
    while i < n_primary:
        chain = [i]
        j = i
        while j + 1 < n_primary:
            expr_j = state_expressions_orig[j]
            sym_j_plus_1 = primary_state_syms[j+1]
            
            try:
                deriv_wrt_dt = sp.diff(expr_j, dt_sym)
                is_linked = sym_j_plus_1 in deriv_wrt_dt.free_symbols
            except Exception:
                is_linked = False
            
            diag_j = np.isclose(F_numeric[j, j], 1.0, rtol=tol)
            diag_j1 = np.isclose(F_numeric[j + 1, j + 1], 1.0, rtol=tol)

            if is_linked and diag_j and diag_j1:
                chain.append(j + 1)
                j += 1
            else:
                break
        
        L = len(chain)
        if L >= 2:
            G_chain = np.array([ (dt_val**k) / factorial(k) for k in reversed(range(L)) ]).reshape((L, 1))
            Q_block = noise_val**2 * (G_chain @ G_chain.T)
            for row_idx, state_idx_i in enumerate(chain):
                for col_idx, state_idx_j in enumerate(chain):
                    Q_numeric[state_idx_i, state_idx_j] += Q_block[row_idx, col_idx]
            i = chain[-1] + 1
        else:
            i += 1

    return F_numeric, Q_numeric, all_state_names, nonlinear_map


def _linearize_observation(
    observation_func: Callable,
    all_state_names: List[str],
    nonlinear_funcs: Tuple[str] = ("exp", "tanh", "clip", "sin", "cos"),
) -> Tuple[np.ndarray, List[str], Dict]:
    """
    Symbolically derive the observation matrix H and detect nonlinearities.
    """
    
    if observation_func is None:
        n_cols = len(all_state_names)
        return np.empty((0, n_cols)), all_state_names, {}

    # --- Step 0: Define a Gradient-Friendly Clip for Observations ---
    # Unlike the global 'clip' class used for Transitions (which returns 0 diff
    # to aid state lifting), this class returns 1 diff to ensure H captures
    # the linear relationship between state and observation.
    class obs_clip(sp.Function):
        @classmethod
        def eval(cls, x, lower, upper):
            return None # Keep symbolic
        
        def fdiff(self, argindex=1):
            # Return 1.0 so the Jacobian (H) sees the connection.
            # Technically this is a Heaviside, but for initialization 
            # and Jacobian shape detection, 1.0 is required.
            return sp.Integer(1)

    # Force the name to be "clip" so the nonlinearity detector recognizes it
    obs_clip.__name__ = "clip"

    # --- Step 1: Parse and symify source ---
    src = _sympify_source(observation_func)
    src = src.replace("_symbolic_func", "_symbolic_obs_func", 1)
    
    # We inject our specific 'obs_clip' into the globals instead of the global 'clip'
    symbolic_globals = {
        "sp": sp, 
        "clip": obs_clip,  # <--- UPDATED: Use the gradient-friendly clip
        "exp": sp.exp, "sin": sp.sin, "cos": sp.cos,
        "tanh": sp.tanh, "log": sp.log, "sqrt": sp.sqrt, "Abs": sp.Abs,
        "__builtins__": {"float": float, "int": int}
    }
    exec(src, symbolic_globals)
    symbolic_func = symbolic_globals["_symbolic_obs_func"]

    # --- Step 2: Get symbols ---
    obs_sig = inspect.signature(observation_func)
    obs_param_names = list(obs_sig.parameters.keys())
    obs_param_syms = [sp.symbols(n) for n in obs_param_names]
    all_state_syms = [sp.symbols(n) for n in all_state_names]

    # --- Step 3: Evaluate observation function ---
    symbolic_result = symbolic_func(*obs_param_syms)
    if not isinstance(symbolic_result, (list, tuple, sp.Matrix)):
        symbolic_result = (symbolic_result,)
    
    obs_expressions = [sp.simplify(expr) for expr in symbolic_result]
    
    # --- Step 4: Detect Nonlinearities ---
    obs_nonlinear_map = {}
    for i, expr in enumerate(obs_expressions):
        for sub_expr in sp.preorder_traversal(expr):
            fname = sub_expr.func.__name__ if isinstance(sub_expr, sp.Function) else None
            if fname in nonlinear_funcs:
                arg = sub_expr.args[0] if sub_expr.args else "unknown"
                input_idx = None
                if arg in all_state_syms:
                    input_idx = all_state_syms.index(arg)
                else:
                    for j, sym in enumerate(all_state_syms):
                        if sym in arg.free_symbols:
                            input_idx = j
                            break
                
                nl_entry = {
                    "input_state_index": input_idx,
                    "output_obs_index": i,
                    "transformation": fname.lower(),
                }
                
                if fname == "clip" and len(sub_expr.args) == 3:
                    try:
                        nl_entry["lower_bound"] = float(sub_expr.args[1].evalf())
                        nl_entry["upper_bound"] = float(sub_expr.args[2].evalf())
                    except Exception: pass
                
                obs_nonlinear_map[len(obs_nonlinear_map)] = nl_entry
                # We only log the outermost nonlinearity
                break 

    # --- Step 5: Build H (Jacobian) ---
    H_rows = []
    for obs_idx, expr in enumerate(obs_expressions):
        coeffs = [sp.diff(expr, s) for s in all_state_syms]
        H_rows.append(coeffs)
    H_symbolic = sp.Matrix(H_rows)

    # --- Step 6: Convert to numeric matrix (linearize at origin) ---
    H_numeric = np.zeros(H_symbolic.shape, dtype=float)
    subs_dict = {s: 0 for s in all_state_syms + obs_param_syms} 

    for i in range(H_symbolic.shape[0]):
        for j in range(H_symbolic.shape[1]):
            expr = H_symbolic[i, j]
            try:
                H_numeric[i, j] = float(expr.evalf(subs=subs_dict))
            except (TypeError, ValueError, AttributeError):
                H_numeric[i, j] = 0.0
            
    return H_numeric, all_state_names, obs_nonlinear_map


# --- Matrix Utility Functions ----------------------------------

def _build_block_diagonal_matrix(matrices: List[np.ndarray]) -> np.ndarray:
    """Build block-diagonal matrix from list of matrices."""
    if not matrices:
        return np.array([[]])
    
    total_rows = sum(m.shape[0] for m in matrices)
    total_cols = sum(m.shape[1] for m in matrices)
    
    result = np.zeros((total_rows, total_cols))
    
    row_offset = 0
    col_offset = 0
    for matrix in matrices:
        rows, cols = matrix.shape
        if rows > 0 and cols > 0:
            result[row_offset:row_offset+rows, col_offset:col_offset+cols] = matrix
        row_offset += rows
        col_offset += cols
        
    return result


def _stack_vectors(vectors: List[jnp.ndarray]) -> jnp.ndarray:
    """Stack list of vectors into single vector."""
    if not vectors:
        return jnp.array([])
    return jnp.concatenate(vectors)


# --- Main SSMparam Class ---------------------------------------

class SSMparam(NonlinearStateProcessor):
    """
    State-space model parameters with function-based API and automatic linearization.

    This class accepts lists of transition and observation functions and automatically
    builds block-diagonal matrices for multi-model systems with nonlinear dynamics.
    
    Attributes:
        initial_mean: Stacked initial state mean vector (jnp.ndarray)
        initial_covariance: Block-diagonal initial state covariance (jnp.ndarray)
        transition: List of original transition functions
        observation: List of original observation functions
        transition_matrix: Block-diagonal state transition matrix F (jnp.ndarray)
        transition_covariance: Block-diagonal process noise covariance Q (jnp.ndarray)
        observation_matrix: Block-diagonal observation matrix H (jnp.ndarray)
        state_names: Global list of all state variable names (including auxiliary)
        model_info: List of dictionaries with per-model information
    """

    def __init__(self, 
                 transition: List[Callable],
                 observation: List[Callable],
                 dt_val: float = 1.0,
                 **kwargs):
        """
        Create SSMparam from functions with automatic linearization.
        
        Args:
            transition: List of transition functions per model.
            observation: List of observation functions.
                         - If len(obs) == len(trans), lists are paired 1-to-1.
                         - If len(obs) == 1 and len(trans) > 1, 
                           the single obs is paired with the first model,
                           and all other models get 'None'.
            dt_val: Time step value for linearization.
            **kwargs: Additional arguments passed to linearization.
        """
        super().__init__()  # Initialize NonlinearStateProcessor (CloseFormMoments)
        
        n_models = len(transition)
        
        # The observation list must be explicit and match the other lists
        # If len(observation) == 1, we treat it as global, so we don't enforce length match with models
        if len(observation) != 1 and not (len(transition) == len(observation)):
             raise ValueError(
                "Transition and Observation lists must have the same length. "
                "The 'observation' list must have length 1 (global) or {n_models} (per-model). "
                f"Got lengths {len(transition)}, and {len(observation)}."
            )
        
        models_data = []
        all_state_names = []

        # --- Process Transition Models ---
        self.primary_dims = [] # Track primary dimensions for padding

        for i, trans_func in enumerate(transition):
            sig = inspect.signature(trans_func)
            param_names = list(sig.parameters.keys())
            
            # Identify primary states from signature (excluding dt, noise)
            primary_state_param_names = [name for name in param_names if name not in ['dt', 'noise']]
            num_primary_states = len(primary_state_param_names)
            self.primary_dims.append(num_primary_states)
            
            primary_state_names_i = primary_state_param_names
            
            F_i, Q_i, states_i, nonlinear_i = _linearize_transition(
                trans_func, primary_state_names_i, dt_val, **kwargs
            )
            
            # Detect storage states (primary states that are purely nonlinear outputs)
            # These do NOT need to be initialized by the user; they will be padded.
            # We assume they are at the end of the parameter list for correct padding behavior.
            num_primary_total = len(primary_state_param_names)
            num_storage_states = 0
            if nonlinear_i:
                for nl in nonlinear_i.values():
                    # If the output of a nonlinearity is a primary state index
                    if nl['output_state_index'] < num_primary_total:
                        num_storage_states += 1
            
            # The number of states the USER must provide
            num_init_required = num_primary_total - num_storage_states
            self.primary_dims.append(num_init_required)
            
            models_data.append({
                'F': F_i, 'Q': Q_i, 'states': states_i, 
                'nonlinear': nonlinear_i, 'model_idx': i,
                'transition_func': trans_func,
                'n_primary': num_init_required # Used for validation in pad_initial_state
            })
            all_state_names.extend(states_i)
        
        # --- Assemble Block-Diagonal F, Q ---
        F_block = _build_block_diagonal_matrix([model['F'] for model in models_data])
        Q_block = _build_block_diagonal_matrix([model['Q'] for model in models_data])
        
        # Initial mean/cov are now None
        self.initial_mean = None
        self.initial_covariance = None

        # --- Process Observation Models ---
        self.global_obs_nonlinear = {}
        
        if len(observation) == 1:
            # Global Observation Mode
            H, _, obs_nonlinear = _linearize_observation(
                observation[0], all_state_names, **kwargs
            )
            self.observation_matrix = jnp.array(H)
            self.global_obs_nonlinear = obs_nonlinear
            
        else:
            # Block-Diagonal Observation Mode
            H_matrices = []
            for i, obs_func in enumerate(observation):
                model_state_names = models_data[i]['states']
                
                H_i, _, obs_nonlinear_i = _linearize_observation(
                    obs_func, model_state_names, **kwargs
                )
                
                H_matrices.append(H_i)
                models_data[i]['obs_nonlinear'] = obs_nonlinear_i
                models_data[i]['n_obs'] = H_i.shape[0]
                
            H_block = _build_block_diagonal_matrix(H_matrices)
            self.observation_matrix = jnp.array(H_block)
        
        # --- Set Instance Attributes ---
        self.transition = transition
        self.observation = observation
        self.transition_matrix = jnp.array(F_block)
        self.transition_covariance = jnp.array(Q_block)
        # self.observation_matrix is already set above
        self.state_names = all_state_names
        self.model_info = models_data
        self.control_matrix = None
        
        # --- Compile Nonlinear Transformation Configs ---
        nonlinearities = self.get_nonlinearities()
        self.trans_config = self.compile_config(nonlinearities['transition'], is_observation=False)
        self.obs_config = self.compile_config(nonlinearities['observation'], is_observation=True)
        
    # --- Public Getters ---

    def is_linear(self) -> bool:
        """
        Check if the model is linear *for filtering*.
        The system is considered nonlinear if there are
        nonlinear observations or transition, which require an EKF/UKF-style H update.
        """
        if self.global_obs_nonlinear:
            return False
            
        for model in self.model_info:
            # Check for transition nonlinearities (stored in 'nonlinear')
            # Check for observation nonlinearities (stored in 'obs_nonlinear')
            if model.get('nonlinear') or model.get('obs_nonlinear'):
                return False
        return True
    
    def get_state_dim(self) -> int:
        """Get total state dimension."""
        return self.transition_matrix.shape[0]
    
    def get_obs_dim(self) -> int:
        """Get observation dimension."""
        return self.observation_matrix.shape[0]
    
    def get_model_count(self) -> int:
        """Get number of models."""
        return len(self.model_info)
    
    def get_jax_params(self) -> JaxSSMParams:
        """
        Extract JAX-compatible parameters for gradient-based learning.
        
        This method separates the numerical JAX arrays from the symbolic SSMparam
        structure, enabling JIT compilation and automatic differentiation.
        
        Returns:
            JaxSSMParams: Dataclass containing JAX arrays for filtering
        """
        return JaxSSMParams(
            F=self.transition_matrix,
            Q=self.transition_covariance,
            H=self.observation_matrix,
            B=self.control_matrix,
            m0=self.initial_mean,
            P0=self.initial_covariance
        )

    def get_nonlinearities(self) -> Dict:
        """
        Get all detected nonlinearities, aggregated into a single global map.
        All indices are adjusted to be absolute w.r.t. the final stacked vectors.
        
        Returns:
            A dict {"transition": [...], "observation": [...]}.
            Each value is a single list of nonlinearity info dicts.
        """
        global_transition_nl = []
        global_observation_nl = []
        
        state_offset = 0
        obs_offset = 0
        
        # Iterate over each model to build the global map
        for i, model in enumerate(self.model_info):
            
            # --- Process Transition Nonlinearities ---
            # model['nonlinear'] is a dict like {0: {...}, 1: {...}}
            if model['nonlinear']:
                for nl_dict in model['nonlinear'].values():
                    # Copy the dict to avoid modifying the original
                    global_nl_entry = nl_dict.copy()
                    
                    # Adjust indices by adding the current state_offset
                    if 'input_state_index' in global_nl_entry and global_nl_entry['input_state_index'] is not None:
                        global_nl_entry['input_state_index'] += state_offset
                    
                    if 'output_state_index' in global_nl_entry and global_nl_entry['output_state_index'] is not None:
                        global_nl_entry['output_state_index'] += state_offset
                    
                    global_transition_nl.append(global_nl_entry)
            
            # --- Process Observation Nonlinearities (Per-Model) ---
            if model.get('obs_nonlinear'):
                for nl_dict in model['obs_nonlinear'].values():
                    global_nl_entry = nl_dict.copy()
                    
                    # Adjust indices
                    if 'input_state_index' in global_nl_entry and global_nl_entry['input_state_index'] is not None:
                        global_nl_entry['input_state_index'] += state_offset
                    
                    if 'output_obs_index' in global_nl_entry and global_nl_entry['output_obs_index'] is not None:
                        global_nl_entry['output_obs_index'] += obs_offset
                        
                    global_observation_nl.append(global_nl_entry)

            # --- Update offsets for the next model ---
            # The state offset increases by the total number of states in this model
            state_offset += len(model['states'])
            
            # The obs offset increases by the number of observations in this model
            obs_offset += model.get('n_obs', 0)
            
        # --- Process Global Observation Nonlinearities ---
        if self.global_obs_nonlinear:
            for nl_dict in self.global_obs_nonlinear.values():
                # Indices are already global (w.r.t all_state_names)
                global_observation_nl.append(nl_dict.copy())

        return {
            "transition": global_transition_nl,
            "observation": global_observation_nl
        }

    def pad_initial_state(self, 
                       initial_mean: List[jnp.ndarray], 
                       initial_covariance: List[jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Pad user-provided primary initial states with zeros for auxiliary states.
        
        Args:
            initial_mean: List of primary state means per model (n_models, n_primary_i)
            initial_covariance: List of primary state covariances per model (n_models, n_primary_i, n_primary_i)
        
        Returns:
            Tuple of (m0_padded, P0_padded):
            - m0_padded: Stacked padded mean vector (total_state_dim,)
            - P0_padded: Block-diagonal padded covariance matrix (total_state_dim, total_state_dim)
        """
        n_models = len(self.model_info)
        
        if len(initial_mean) != n_models or len(initial_covariance) != n_models:
             raise ValueError(
                f"Input lists must have length {n_models} (number of models). "
                f"Got lengths {len(initial_mean)} and {len(initial_covariance)}."
            )
            
        padded_means = []
        padded_covs = []
        
        for i, model in enumerate(self.model_info):
            m_i = initial_mean[i]
            P_i = initial_covariance[i]
            
            n_primary = model['n_primary']
            F_dim = model['F'].shape[0]

            # Validate primary dimensions
            if m_i.shape[0] != n_primary:
                raise ValueError(f"Model {i}: Expected mean dim {n_primary}, got {m_i.shape[0]}")
            
            if P_i.shape != (n_primary, n_primary):
                 raise ValueError(f"Model {i}: Expected cov shape ({n_primary}, {n_primary}), got {P_i.shape}")

            dim_diff = F_dim - n_primary
            
            # Pad Mean
            if dim_diff > 0:
                m_pad = jnp.concatenate([m_i, jnp.zeros(dim_diff)])
            else:
                m_pad = m_i
            padded_means.append(m_pad)
            
            # Pad Covariance
            if dim_diff > 0:
                P_pad = jnp.zeros((F_dim, F_dim))
                P_pad = P_pad.at[:n_primary, :n_primary].set(P_i)
                padded_covs.append(P_pad)
            else:
                padded_covs.append(P_i)

        # Assemble global block-diagonal matrices
        m0_global = _stack_vectors(padded_means)
        
        # Use JAX-compatible block_diag for covariance
        from jax.scipy.linalg import block_diag
        if len(padded_covs) == 0:
            P0_global = jnp.zeros((0, 0))
        else:
            P0_global = block_diag(*padded_covs)
        
        return m0_global, P0_global
