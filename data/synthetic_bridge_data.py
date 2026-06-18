import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from typing import List, Dict, Tuple, Any
from functools import partial

# Configuration JAX pour la précision 64-bit
jax.config.update("jax_enable_x64", True)

@partial(jax.jit, static_argnames=['T', 'G', 'min_obs', 'max_obs', 'gap_range', 'num_inspector_ids'])
def _core_simulation(
    key: jnp.ndarray,
    init_means: jnp.ndarray,
    init_covs: jnp.ndarray,
    Q_base: jnp.ndarray,
    directions: jnp.ndarray,
    inspector_sigma_range: Tuple[float, float],
    T: int,
    G: int,
    min_obs: int,
    max_obs: int,
    gap_range: Tuple[int, int],
    dt: float,
    num_inspector_ids: int
):
    """
    Cœur de la simulation JAX.
    Retourne les états vrais et les observations brutes (avec padding/NaN).
    """
    N_total = init_means.shape[0]
    
    # --- 1. Échantillonnage des états initiaux ---
    key, key_init = random.split(key)
    
    def sample_init(k, m, c):
        return random.multivariate_normal(k, m, c)
    
    # (N, G, 3)
    flat_means = init_means.reshape(-1, 3)
    flat_covs = init_covs.reshape(-1, 3, 3)
    keys_init = random.split(key_init, N_total * G)
    
    x0_flat = jax.vmap(sample_init)(keys_init, flat_means, flat_covs)
    x0 = x0_flat.reshape(N_total, G, 3)
    
    # --- 2. Simulation des trajectoires (Dynamique) ---
    Q_sym = 0.5 * (Q_base + Q_base.T)
    eigvals, eigvecs = jnp.linalg.eigh(Q_sym)
    eigvals_clamped = jnp.clip(eigvals, a_min=1e-12)
    sqrt_eigs = jnp.sqrt(eigvals_clamped)
    
    L_robust = eigvecs * sqrt_eigs[None, :]
    
    key, key_noise = random.split(key)
    z_noise = random.normal(key_noise, (T, N_total, G, 3))
    w_noise = jnp.einsum('...i,ji->...j', z_noise, L_robust)
    
    def step_fn(state_prev, w_curr):
        g = state_prev[..., 0]
        v = state_prev[..., 1]
        a = state_prev[..., 2]
        
        dir_broadcast = directions[jnp.newaxis, :] 
        
        v_term = jnp.where(dir_broadcast < 0, jnp.minimum(v, 0.0), jnp.maximum(v, 0.0))
        a_term = jnp.where(dir_broadcast < 0, jnp.minimum(a, 0.0), jnp.maximum(a, 0.0))
        
        g_new = g + v_term * dt + 0.5 * a_term * (dt**2) + w_curr[..., 0]
        v_new = v + a_term * dt                        + w_curr[..., 1]
        a_new = a                                      + w_curr[..., 2]
        
        state_new = jnp.stack([g_new, v_new, a_new], axis=-1)
        return state_new, state_new

    final_state, traj_rest = jax.lax.scan(step_fn, x0, w_noise[1:])
    
    true_states_time_first = jnp.concatenate([x0[jnp.newaxis, ...], traj_rest], axis=0)
    true_states = jnp.transpose(true_states_time_first, (1, 2, 3, 0)) # (N, G, 3, T)
    
    # --- 3. Génération des observations ---
    key, key_inspectors = random.split(key)
    
    insp_sigmas = random.uniform(
        key_inspectors, 
        (num_inspector_ids,), 
        minval=inspector_sigma_range[0], 
        maxval=inspector_sigma_range[1]
    )
    
    key, k1, k2, k3, k4 = random.split(key, 5)
    n_obs = random.randint(k1, (N_total,), min_obs, max_obs + 1)
    
    def generate_schedule(k, n_i):
        k_gap, k_start = random.split(k)
        gaps = random.randint(k_gap, (max_obs,), gap_range[0], gap_range[1] + 1)
        
        idx = jnp.arange(max_obs)
        gap_mask = idx < (n_i - 1)
        valid_gaps = jnp.where(gap_mask, gaps, 0)
        
        rel_times = jnp.concatenate([jnp.array([0]), jnp.cumsum(valid_gaps)])[:max_obs] 
        span = rel_times[n_i-1]
        
        # max_start = jnp.maximum(0, (T - 1) - span)
        # Bias observations to start early (first 1/4 of time) to capture non-zero Gate A values
        max_start_early = jnp.minimum(T // 4, (T - 1) - span)
        max_start = jnp.maximum(0, max_start_early)
        start_offset = random.randint(k_start, (), 0, max_start + 1)
        
        abs_times = rel_times + start_offset
        valid_mask = idx < n_i
        return abs_times, valid_mask
        
    keys_sched = random.split(k3, N_total)
    obs_times, obs_valids = jax.vmap(generate_schedule)(keys_sched, n_obs)
    
    insp_indices_small = random.randint(k4, (N_total, max_obs), 0, num_inspector_ids).astype(jnp.int32)
    
    def fill_row(times, mask, insp_idx):
        row_mask = jnp.zeros((T,), dtype=bool)
        row_insp = jnp.full((T,), -1, dtype=jnp.int32)
        
        scatter_indices = jnp.where(mask, times, T).astype(jnp.int32)
        
        row_mask = row_mask.at[scatter_indices].set(True, mode='drop')
        row_insp = row_insp.at[scatter_indices].set(insp_idx, mode='drop')
        return row_mask, row_insp
        
    observation_mask, inspector_indices = jax.vmap(fill_row)(obs_times, obs_valids, insp_indices_small)
    
    safe_insp_indices = jnp.maximum(inspector_indices, 0)
    sigmas_looked_up = insp_sigmas[safe_insp_indices]
    inspector_sigmas_grid = jnp.where(observation_mask, sigmas_looked_up, 0.0)
    
    g_true = true_states[:, :, 0, :] # (N, G, T) where indices are 0:'a', 1:'ab', 2:'d'
    
    # 1. Calcul des agrégats de grades "vrais"
    a_true = jnp.clip(g_true[:, 0, :], 0, 1) * jnp.clip(g_true[:, 1, :], 0, 1)
    b_true = jnp.clip(1 - g_true[:, 0, :], 0, 1) * jnp.clip(g_true[:, 1, :], 0, 1)
    c_true = jnp.clip(1 - g_true[:, 2, :], 0, 1) * (1.0 - jnp.clip(g_true[:, 1, :], 0, 1))
    d_true = jnp.clip(g_true[:, 2, :], 0, 1) * (1.0 - jnp.clip(g_true[:, 1, :], 0, 1))
    ab_true = a_true + b_true

    key, k_obs_noise = random.split(key)
    # base_noise: (N, G, T) -> G=3 offre 3 bruits indépendants pour v_a, v_ab, v_d
    base_noise = random.normal(k_obs_noise, (N_total, G, T))
    
    sigma_expanded = inspector_sigmas_grid[:, jnp.newaxis, :]
    obs_noise_all = base_noise * sigma_expanded
    
    v_a = obs_noise_all[:, 0, :]
    v_ab = obs_noise_all[:, 1, :]
    v_d = obs_noise_all[:, 2, :]

    # 2. Sequential noise clipping to ensure y_a, y_b, y_c, y_d ∈ [0,1] with sum=1
    # 2. Sequential noise and hierarchical application
    # Extract latent states for readability
    g_a_val = g_true[:, 0, :]
    g_ab_val = g_true[:, 1, :]
    g_d_val = g_true[:, 2, :]

    # 1.5 Apply the "Perfect Condition" physical constraints
    # If the latent state is beyond the "perfect" threshold, the inspector makes no mistakes.
    # User rules: > 1 for A and AB means perfect. < 0 for D means perfect (all C).
    
    v_ab_active = jnp.where(g_ab_val >= 1.0, 0.0, v_ab)  # No top-level confusion if perfectly AB
    v_a_active  = jnp.where(g_a_val >= 1.0, 0.0, v_a)    # No internal A/B confusion if perfectly A
    v_d_active  = jnp.where(g_d_val <= 0.0, 0.0, v_d)    # No internal C/D confusion if perfectly C

    # 2. Sequential noise clipping and hierarchical application
    cd_true = c_true + d_true  # = 1 - ab_true
    
    # Step 1: Clip the top-level parent error using the ACTIVE noise
    v_ab_clipped = jnp.clip(v_ab_active, -ab_true, cd_true)
    
    # Calculate true proportions safely to avoid division by zero
    prop_a = jnp.where(ab_true > 0, a_true / ab_true, 0.5)
    prop_d = jnp.where(cd_true > 0, d_true / cd_true, 0.5)

    # Step 2: Apply the Professor's hierarchical generative model
    # Because we zeroed the noises above, if the bridge is perfect, 
    # these naturally resolve to a_true, b_true, etc., without breaking the sum!
    y_a_raw = a_true + v_a_active + v_ab_clipped * prop_a
    y_b_raw = b_true - v_a_active + v_ab_clipped * (1.0 - prop_a)
    
    y_c_raw = c_true - v_d_active - v_ab_clipped * (1.0 - prop_d)
    y_d_raw = d_true + v_d_active - v_ab_clipped * prop_d

    # Step 3: Secure clipping to ensure bounds [0,1] 
    # We clip one grade per pair, and subtract to get the other to strictly preserve the parent sums
    ab_observed = ab_true + v_ab_clipped
    cd_observed = cd_true - v_ab_clipped
    
    y_a = jnp.clip(y_a_raw, 0.0, ab_observed)
    y_b = ab_observed - y_a
    
    y_d = jnp.clip(y_d_raw, 0.0, cd_observed)
    y_c = cd_observed - y_d

    y_ab = y_a + y_b
    y_cd = y_c + y_d
    
    # 3. Dérivation des gates observés avec division sécurisée
    g_a_obs = jnp.where(y_ab > 0, y_a / y_ab, 0.0)
    g_ab_obs = y_ab
    g_d_obs = jnp.where(y_cd > 0, y_d / y_cd, 0.0)
    
    y_all = jnp.stack([g_a_obs, g_ab_obs, g_d_obs], axis=1) # (N, G, T)
    
    mask_expanded = observation_mask[:, jnp.newaxis, :]
    observations = jnp.where(mask_expanded, y_all, jnp.nan)
    
    # Stack true grades (N, 4, T)
    true_grades = jnp.stack([a_true, b_true, c_true, d_true], axis=1)
    
    # Stack observed grades (N, 4, T)
    obs_grades = jnp.stack([y_a, y_b, y_c, y_d], axis=1)
    
    return true_states, observations, observation_mask, inspector_indices, inspector_sigmas_grid, true_grades, obs_grades

def generate_bridge_network_data(
    num_bridges: int,
    categories: List[str],
    num_elements_per_category: int,
    num_inspector_ids: int,
    inspector_sigma_range: Tuple[float, float],
    init_conditions: Dict[str, Dict[str, Dict[str, Any]]],
    sigma_w: float = 5e-5,
    year_start: int = 2008,
    year_end: int = 2100,
    min_obs: int = 3,
    max_obs: int = 10,
    gap_range: Tuple[int, int] = (2, 4),
    seed: int = 0
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Génère des données synthétiques de dégradation.
    Retourne (obs_dict, true_dict).
    """
    
    years = jnp.arange(year_start, year_end + 1)
    T = int(years.shape[0])
    dt = 1.0
    gate_labels = ['a', 'ab', 'd']
    G = len(gate_labels)
    directions = jnp.array([-1.0, -1.0, 1.0])
    
    Q_base = jnp.array([
        [dt**4 / 4, dt**3 / 2, dt**2 / 2],
        [dt**3 / 2, dt**2, dt],
        [dt**2 / 2, dt, 1],
    ], dtype=jnp.float64) * (sigma_w**2)
    
    meta_bridges = []
    meta_cats = []
    meta_elements = []
    means_list = []
    covs_list = []
    
    for b in range(num_bridges):
        b_id = f"b{b+1:04d}"
        for cat in categories:
            for e in range(num_elements_per_category):
                e_id = f"e{e+1:04d}"
                meta_bridges.append(b_id)
                meta_cats.append(cat)
                meta_elements.append(e_id)
                
                g_means = []
                g_covs = []
                for g in gate_labels:
                    m = jnp.array(init_conditions[cat][g]['mean'])
                    c = jnp.array(init_conditions[cat][g]['cov'])
                    g_means.append(m)
                    g_covs.append(c)
                
                means_list.append(jnp.stack(g_means))
                covs_list.append(jnp.stack(g_covs))
    
    init_means = jnp.stack(means_list)
    init_covs = jnp.stack(covs_list)
    
    key = random.PRNGKey(seed)
    
    true_states, observations, obs_mask, insp_idx, insp_sigmas, true_grades, obs_grades = _core_simulation(
        key, init_means, init_covs, Q_base, directions, inspector_sigma_range,
        T, G, min_obs, max_obs, gap_range, dt, num_inspector_ids
    )
    
    inspector_ids_list = [f"I{i+1:04d}" for i in range(num_inspector_ids)]
    
    # true_grades and obs_grades are now returned directly from _core_simulation
    # true_grades: (N, 4, T) - true A, B, C, D values
    # obs_grades: (N, 4, T) - observed y_a, y_b, y_c, y_d values
    
    # --- Construction de true_dict (séries complètes) ---
    true_dict = {
        'states': true_states,       # (N, G, 3, T)
        'grades': true_grades,       # (N, 4, T)
        'times': years,              # (T,)
        'metadata': {
            'bridge_ids': meta_bridges,     # List[str], length N
            'categories': meta_cats,        # List[str], length N
            'element_ids': meta_elements,   # List[str], length N
        }
    }
    
    # --- Construction de obs_dict (observations denses) ---
    # On récupère les indices où il y a une observation
    # obs_mask is (N, T)
    n_idx, t_idx = jnp.nonzero(obs_mask)
    
    # Helper pour extraire les valeurs aplaties
    def flatten_on_mask(matrix_nt_vals):
        # matrix_nt_vals: (N, T, ...)
        return matrix_nt_vals[n_idx, t_idx, ...]

    # Extraction des données
    # observations: (N, G, T) -> (N, T, G)
    obs_g_flat = flatten_on_mask(jnp.transpose(observations, (0, 2, 1))) # (N_obs, G)
    obs_grades_flat = flatten_on_mask(jnp.transpose(obs_grades, (0, 2, 1))) # (N_obs, 4)
    
    # Métadonnées pour chaque observation
    # On convertit les listes en array numpy pour indexation facile
    meta_bridges_arr = np.array(meta_bridges)
    meta_cats_arr = np.array(meta_cats)
    meta_elements_arr = np.array(meta_elements)
    
    # Les indices n_idx sont des indices JAX, on les convertit en numpy pour l'indexation de listes
    n_idx_np = np.array(n_idx)
    
    obs_bridge_ids = meta_bridges_arr[n_idx_np]
    obs_categories = meta_cats_arr[n_idx_np]
    obs_element_ids = meta_elements_arr[n_idx_np]
    obs_years = years[t_idx] # t_idx est jax array, years jax array -> ok
    
    # Inspecteurs et Sigmas
    # inspector_indices: (N, T)
    # inspector_sigmas_grid: (N, T)
    insp_idx_flat = insp_idx[n_idx, t_idx]
    sigma_flat = insp_sigmas[n_idx, t_idx]
    
    # Conversion des IDs inspecteurs
    insp_idx_np = np.array(insp_idx_flat)
    inspector_ids_arr = np.array(inspector_ids_list)
    obs_inspector_ids = inspector_ids_arr[insp_idx_np]

    obs_dict = {
        'g': obs_g_flat,             # (N_obs, G)
        'grades': obs_grades_flat,   # (N_obs, 4)
        'years': obs_years,          # (N_obs,)
        'bridge_ids': obs_bridge_ids, # (N_obs,)
        'categories': obs_categories, # (N_obs,)
        'element_ids': obs_element_ids, # (N_obs,)
        'inspector_ids': obs_inspector_ids, # (N_obs,)
        'sigmas': sigma_flat,         # (N_obs,)
        'gate_labels': gate_labels,
        'grade_labels': ['A', 'B', 'C', 'D']
    }

    return obs_dict, true_dict

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import time

    print("Génération des données synthétiques...")
    
    init_cond = {
        'poutre': {
            'a':  {'mean': [1., -0.065, -0.0035], 'cov': np.diag([0.05, 0.005, 0.0005])**2},
            'ab': {'mean': [1.2, -0.00, -0.0035], 'cov': np.diag([0.05, 0.005, 0.0005])**2},
            'd':  {'mean': [-0.3,  0.00,  0.0008], 'cov': np.diag([0.1,  0.00001,   0.00001 ])**2},
        }
    }
    
    start_time = time.time()
    obs_dict, true_dict = generate_bridge_network_data(
        num_bridges=10, 
        categories=['poutre'],
        num_elements_per_category=10,
        num_inspector_ids=10,
        inspector_sigma_range=(0.01, 0.1),
        init_conditions=init_cond,
        year_end=2060,
        max_obs=7,
        sigma_w=1e-5,
        seed=42
    )
    print(f"Temps d'exécution: {time.time() - start_time:.4f}s")
    
    print("Obs Dict Keys:", obs_dict.keys())
    print("True Dict Keys:", true_dict.keys())
    print("Shape Observatios (g):", obs_dict['g'].shape)
    
def plot_element(obs_dict, true_dict, b_idx, filename):
    b_id = true_dict['metadata']['bridge_ids'][b_idx]
    c_id = true_dict['metadata']['categories'][b_idx]
    e_id = true_dict['metadata']['element_ids'][b_idx]
    
    times = true_dict['times']
    g_true = true_dict['states'][b_idx, :, 0, :] # (G, T)
    grades_true = true_dict['grades'][b_idx, :, :] # (4, T)
    
    mask_obs = (obs_dict['bridge_ids'] == b_id) & \
               (obs_dict['categories'] == c_id) & \
               (obs_dict['element_ids'] == e_id)
               
    obs_years = obs_dict['years'][mask_obs]
    obs_g = obs_dict['g'][mask_obs]
    obs_grades = obs_dict['grades'][mask_obs]
    obs_sigmas = obs_dict['sigmas'][mask_obs]
    obs_inspectors = obs_dict['inspector_ids'][mask_obs]
    
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 1)
    
    # Plot 1: Gates
    ax1 = fig.add_subplot(gs[0])
    gate_colors = ['green', 'blue', 'red']
    gate_labels = obs_dict['gate_labels']
    
    for i, g_name in enumerate(gate_labels):
        ax1.plot(times, g_true[i], label=f"True {g_name}", color=gate_colors[i], alpha=0.7)
        if len(obs_years) > 0:
            ax1.errorbar(obs_years, obs_g[:, i], yerr=obs_sigmas, fmt='o', color=gate_colors[i], label=f"Obs {g_name}", markersize=4)
            for k, insp in enumerate(obs_inspectors):
                if i == 0:
                    ax1.annotate(insp, (obs_years[k], obs_g[k, i]), xytext=(0, 5), textcoords='offset points', fontsize=7, alpha=0.6)

    ax1.set_title(f"Gates Evolution: {b_id} | {c_id} | {e_id}")
    ax1.set_ylim(-0.2, 1.3)
    ax1.legend(loc='upper right', fontsize='small', ncol=3)
    ax1.grid(True, alpha=0.2)
    
    # Plot 2: Grades
    ax2 = fig.add_subplot(gs[1])
    grade_colors = ['green', 'blue', 'orange', 'red']
    grade_labels = obs_dict['grade_labels']
    
    for i, grade_name in enumerate(grade_labels):
        ax2.plot(times, grades_true[i], label=f"True {grade_name}", color=grade_colors[i], linestyle='-', alpha=0.6)
        if len(obs_years) > 0:
            ax2.scatter(obs_years, obs_grades[:, i], label=f"Obs {grade_name}", color=grade_colors[i], marker='x', s=20)

    ax2.set_title(f"Grades (A, B, C, D) Evolution")
    ax2.set_ylim(-0.1, 1.1)
    ax2.legend(loc='upper right', fontsize='small', ncol=4)
    ax2.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import time

    print("Génération des données synthétiques...")
    
    init_cond = {
        'poutre': {
            'a':  {'mean': [1., -0.0065, -0.00035], 'cov': np.diag([0.05, 0.005, 0.0005])**2},
            'ab': {'mean': [1.2, -0.00, -0.0035], 'cov': np.diag([0.05, 0.005, 0.0005])**2},
            'd':  {'mean': [-0.3,  0.00,  0.0008], 'cov': np.diag([0.1,  0.00001,   0.00001 ])**2},
        },
        'dalle': {
            'a':  {'mean': [1.1, -0.0004, -0.002], 'cov': np.diag([0.03, 0.003, 0.0003])**2},
            'ab': {'mean': [1.5, -0.001, -0.0001], 'cov': np.diag([0.03, 0.003, 0.0003])**2},
            'd':  {'mean': [-0.5,  0.0001,  0.00005], 'cov': np.diag([0.05, 0.001, 0.0001])**2},
        },
        'culee': {
            'a':  {'mean': [1.0, -0.02, -0.005], 'cov': np.diag([0.02, 0.002, 0.0002])**2},
            'ab': {'mean': [1.5, -0.05, -0.008], 'cov': np.diag([0.02, 0.002, 0.0002])**2},
            'd':  {'mean': [-0.45,  0.001,  0.001], 'cov': np.diag([0.1, 0.005, 0.0005])**2},
        }
    }
    
    start_time = time.time()
    obs_dict, true_dict = generate_bridge_network_data(
        num_bridges=5, 
        categories=['poutre', 'dalle', 'culee'],
        num_elements_per_category=2,
        num_inspector_ids=5,
        inspector_sigma_range=(0.02, 0.08),
        init_conditions=init_cond,
        year_end=2070,
        max_obs=8,
        sigma_w=2e-5,
        seed=42
    )
    print(f"Temps d'exécution: {time.time() - start_time:.4f}s")
    
    # --- Plotting ---
    print("Création des graphiques pour différents éléments...")
    
    # On plotte le premier élément de chaque catégorie
    n_elements = len(true_dict['metadata']['bridge_ids'])
    seen_categories = set()
    
    for i in range(n_elements):
        cat = true_dict['metadata']['categories'][i]
        if cat not in seen_categories:
            filename = f"synthetic_plot_{cat}_element0.png"
            plot_element(obs_dict, true_dict, i, filename)
            seen_categories.add(cat)
            
    # On plotte aussi un autre élément au hasard pour montrer la variabilité intra-catégorie
    if n_elements > 5:
        plot_element(obs_dict, true_dict, 5, "synthetic_plot_variability.png")

