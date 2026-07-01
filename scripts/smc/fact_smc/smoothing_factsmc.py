import json
import os

# Configure JAX to avoid slow compilation warnings and improve performance
# See: https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_FLAGS', '--xla_gpu_autotune_level=0')

import jax
import jax.numpy as jnp
from cuthberto_carlos.data_types import ResultData
from scripts.smc.fact_smc.model_factsmc import build_factsmc_model
from scripts.smc.process_data import process_data_pl
from tqdm import tqdm
import cuthbert
from jax import tree
from cuthbert.smc.backward_sampler import build_smoother
from cuthbertlib.smc.smoothing import exact_sampling
import cuthbertlib

MAX_GOALS = 8

def print_jax_device_info():
    """Print JAX device and backend information."""
    print("=" * 60)
    print("JAX Device Information:")
    print(f"  Platform: {jax.default_backend()}")
    print(f"  Devices: {jax.devices()}")
    print(f"  Local devices: {jax.local_devices()}")
    print(f"  Process count: {jax.process_count()}")
    print(f"  # of devices: {jax.device_count()}")
    print(f"  # of local devices: {jax.local_device_count()}")
    print("=" * 60)

INIT_MEAN = jnp.array([0.0, 0.0])
INIT_SD = 1.0
INIT_CORR = 0.0
INIT_COV = jnp.array([[INIT_SD**2, INIT_CORR * INIT_SD**2],
                    [INIT_CORR * INIT_SD**2, INIT_SD**2]])
INIT_CHOL_COV = jnp.linalg.cholesky(INIT_COV)
INITIAL_KAPPA = 1e-2
INITIAL_ALPHA = 0.2
INITIAL_BETA = -4.0
NEUTRAL_SCALE = 1.5
FRIENDLY_SCALE = 2.0

# def factSMC_particleEM(
#     params: dict[str, jax.Array],
#     jax_data: ResultData,
#     num_teams: int,
#     N: int,
#     num_steps: int = 10,
#     learning_rate: float = 0.01,
#     key: jax.Array = jax.random.PRNGKey(0)
# ) -> tuple[dict[str, jax.Array], float, list]:
#     """
#     Train model parameters using gradient ascent on log-likelihood.
    
#     This is a practical alternative to full Particle EM with smoothing.
#     Uses JAX automatic differentiation to compute gradients through the filter.
    
#     Args:
#         params: Initial model parameters
#         jax_data: Training data as ResultData
#         num_teams: Number of teams
#         N: Number of particles
#         num_steps: Number of gradient steps
#         learning_rate: Step size for parameter updates
#         key: JAX random key
        
#     Returns:
#         new_params: Updated parameters after training
#         final_log_likelihood: Final log-likelihood
#         history: Training history
#     """
#     import optax
    
#     # Define the loss function (negative log-likelihood)
#     def loss_fn(params):
#         smc_filter, factorializer = build_factsmc_model(
#             N=N, num_teams=num_teams, params=params
#         )
        
#         init_state, local_states, final_state = cuthbert.factorial.filter(
#             filter_obj=smc_filter,
#             factorializer=factorializer,
#             model_inputs=jax_data,
#             output_factorial=False,
#             key=key
#         )
        
#         # Return negative log-likelihood for minimization
#         return -final_state.log_normalizing_constant
    
#     # Setup optimizer
#     optimizer = optax.adam(learning_rate)
#     opt_state = optimizer.init(params)
    
#     # JIT compile the loss and gradient computation
#     value_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    
#     history = []
#     current_params = params
    
#     print(f"\n{'='*60}")
#     print(f"Training Parameters (Gradient-based)")
#     print(f"{'='*60}")
#     print(f"Steps: {num_steps}, Learning rate: {learning_rate}")
#     print(f"Particles: {N}, Teams: {num_teams}")
    
#     for step in range(num_steps):
#         loss_value, grads = value_and_grad(current_params)
#         log_likelihood = float(-loss_value)
        
#         history.append({
#             'step': step,
#             'loss': float(loss_value),
#             'log_likelihood': log_likelihood
#         })
        
#         if step % 5 == 0 or step == num_steps - 1:
#             print(f"\nStep {step:3d}: loss={loss_value:.4f}, log_likelihood={log_likelihood:.4f}")
#             print(f"  kappa={float(current_params['kappa']):.4f}, "
#                   f"alpha={float(current_params['alpha']):.4f}, "
#                   f"beta={float(current_params['beta']):.4f}")
#             print(f"  friendly_scale={float(current_params.get('friendly_scale', 2.0)):.4f}, "
#                   f"neutral_scale={float(current_params.get('neutral_scale', 1.5)):.4f}")
#             init_mean = current_params.get('init_mean', INIT_MEAN)
#             init_cov = current_params.get('init_cov', INIT_COV)
#             print(f"  init_mean=[{float(init_mean[0]):.4f}, {float(init_mean[1]):.4f}], "
#                   f"init_cov_diag=[{float(init_cov[0,0]):.4f}, {float(init_cov[1,1]):.4f}]")
        
#         # Update parameters
#         updates, opt_state = optimizer.update(grads, opt_state, current_params)
#         current_params = optax.apply_updates(current_params, updates)
        
#         # Ensure parameters stay positive where needed
#         current_params = {
#             **current_params,
#             'kappa': jnp.maximum(current_params['kappa'], 1e-6),
#             'friendly_scale': jnp.maximum(current_params.get('friendly_scale', 2.0), 1.0),
#             'neutral_scale': jnp.maximum(current_params.get('neutral_scale', 1.5), 1.0),
#         }
    
#     print(f"\n{'='*60}")
#     print(f"Training Complete")
#     print(f"{'='*60}")
#     print(f"Final log-likelihood: {history[-1]['log_likelihood']:.4f}")
#     print(f"Final params:")
#     print(f"  kappa={float(current_params['kappa']):.4f}")
#     print(f"  alpha={float(current_params['alpha']):.4f}")
#     print(f"  beta={float(current_params['beta']):.4f}")
#     print(f"  friendly_scale={float(current_params.get('friendly_scale', 2.0)):.4f}")
#     print(f"  neutral_scale={float(current_params.get('neutral_scale', 1.5)):.4f}")
#     init_mean = current_params.get('init_mean', INIT_MEAN)
#     init_cov = current_params.get('init_cov', INIT_COV)
#     print(f"  init_mean=[{float(init_mean[0]):.4f}, {float(init_mean[1]):.4f}]")
#     print(f"  init_cov=[[{float(init_cov[0,0]):.4f}, {float(init_cov[0,1]):.4f}], [{float(init_cov[1,0]):.4f}, {float(init_cov[1,1]):.4f}]]")
    
#     # Save results to JSON
#     import json
#     from pathlib import Path
    
#     output_dir = Path(__file__).resolve().parent / "output"
#     output_dir.mkdir(exist_ok=True)
#     output_path = output_dir / "factsmc_params.json"
    
#     # Convert params to JSON-serializable format
#     def to_jsonable(val):
#         if isinstance(val, jax.Array):
#             return float(val) if val.shape == () else val.tolist()
#         return val
    
#     results = {
#         "final_loss": float(history[-1]['loss']),
#         "final_log_normalizing_constant": float(history[-1]['log_likelihood']),
#         "steps": num_steps,
#         "learning_rate": learning_rate,
#         "particles": N,
#         "num_teams": num_teams,
#         "params": {
#             "kappa": to_jsonable(current_params['kappa']),
#             "alpha": to_jsonable(current_params['alpha']),
#             "beta": to_jsonable(current_params['beta']),
#             "friendly_scale": to_jsonable(current_params.get('friendly_scale', 2.0)),
#             "neutral_scale": to_jsonable(current_params.get('neutral_scale', 1.5)),
#             "init_mean": to_jsonable(current_params.get('init_mean', INIT_MEAN)),
#             "init_cov": to_jsonable(current_params.get('init_cov', INIT_COV)),
#         },
#         "history": [
#             {
#                 "step": h['step'],
#                 "loss": h['loss'],
#                 "log_likelihood": h['log_likelihood']
#             } for h in history
#         ]
#     }
    
#     with open(output_path, 'w') as f:
#         json.dump(results, f, indent=2)
    
#     print(f"\nSaved results to {output_path}")
    
#     return current_params, history[-1]['log_likelihood'], history

def extract_team_sequences(
    local_filter_states,
    jax_data: ResultData,
    num_teams: int
):
    """
    Extract per-team state sequences from joint filter states.

    local_filter_states: sequence of joint filter states (4D: home+away)
    jax_data: ResultData with home_team_id, away_team_id, timestamps
    
    Returns:
        team_sequences: dict[team_id] -> {
            'filter_states': list of filter states for this team,
            'timestamps': array of timestamps,
            'is_home': boolean array,
            'opponent_ids': array of opponent team IDs,
            'match_indices': indices into original data
        }
    """
    from cuthbert.smc.particle_filter import ParticleFilterState
    
    # For each team, collect all matches they played
    team_matches = {t: [] for t in range(num_teams)}
    
    num_matches = len(jax_data.match_index)
    
    for t in range(num_matches):
        home_id = int(jax_data.home_team_id[t])
        away_id = int(jax_data.away_team_id[t])
        
        # Extract the joint filter state at time t
        # local_filter_states has shape (T, 2, N, 2) for particles
        # and (T, 2, N) for log_weights
        joint_particles_t = local_filter_states.particles[t]  # (2, N, 2)
        joint_log_weights_t = local_filter_states.log_weights[t]  # (2, N)
        joint_log_norm_t = local_filter_states.log_normalizing_constant[t]  # scalar
        
        # Store match info for both teams
        team_matches[home_id].append({
            'time_idx': t,
            'timestamp': jax_data.timestamp[t],
            'is_home': True,
            'opponent': away_id,
            'particles': joint_particles_t[0],  # Home team: first slot, shape (N, 2)
            'log_weights': joint_log_weights_t[0],  # Home team weights, shape (N,)
            'log_normalizing_constant': joint_log_norm_t,
        })
        team_matches[away_id].append({
            'time_idx': t,
            'timestamp': jax_data.timestamp[t],
            'is_home': False,
            'opponent': home_id,
            'particles': joint_particles_t[1],  # Away team: second slot, shape (N, 2)
            'log_weights': joint_log_weights_t[1],  # Away team weights, shape (N,)
            'log_normalizing_constant': joint_log_norm_t,
        })
    
    team_sequences = {}
    # Extract single-factor states from joint states
    for team_id in team_matches:
        matches = sorted(team_matches[team_id], key=lambda m: m['timestamp'])
        
        # For each match, create a single-factor filter state
        single_factor_states = []
        for match in matches:
            # Create single-factor filter state
            # ParticleFilterState requires: key, particles, log_weights, ancestor_indices, model_inputs, log_normalizing_constant
            
            # Create dummy ancestor indices (identity mapping) since smoother needs them
            n_particles = match['particles'].shape[0]
            ancestor_indices = jnp.arange(n_particles)
            
            single_factor_state = ParticleFilterState(
                key=None,  # Not used in smoothing
                particles=match['particles'],  # (N, 2)
                log_weights=match['log_weights'],  # (N,)
                ancestor_indices=ancestor_indices,
                model_inputs=None,
                log_normalizing_constant=match['log_normalizing_constant'],
            )
            single_factor_states.append(single_factor_state)
        
        team_sequences[team_id] = {
            'filter_states': single_factor_states,
            'timestamps': jnp.array([m['timestamp'] for m in matches]),
            'is_home': jnp.array([m['is_home'] for m in matches]),
            'opponents': jnp.array([m['opponent'] for m in matches]),
        }
    
    return team_sequences

def build_single_factor_smoother(
        params : dict[str, jax.Array],
        n_smoother_particles : int = 500):
    """
    Build backward sampler for single-factor (single team) smoothing.
    """
    
    def single_factor_log_potential(state_prev, state, model_inputs):
        """
        Log potential for single factor. 
        For smoothing, this is just the transition density (no observation).
        """
        # OU transition log-density
        dt = model_inputs['time_delta']
        phi = jnp.exp(-params['kappa'] * dt)
        mean = params['init_mean'] + phi * (state_prev - params['init_mean'])
        cov = params['init_cov'] - phi**2 * params['init_cov']
        cov = cov + jnp.eye(2) * 1e-8
        
        diff = state - mean
        log_det = jnp.linalg.slogdet(cov)[1]
        mahalanobis = diff @ jnp.linalg.solve(cov, diff)
        return -0.5 * (2 * jnp.log(2 * jnp.pi) + log_det + mahalanobis)
    
    def backward_sampling_fn(
            key : jax.Array, 
            x0_all : jax.Array, 
            x1_all : jax.Array,
            log_weight_x0_all : jax.Array,
            log_density : jax.Array,
            x1_ancestor_indices : jax.Array):
        return exact_sampling.simulate(
            key, x0_all, x1_all, log_weight_x0_all,
            log_density, x1_ancestor_indices
        )
    
    # Use multinomial resampling which supports different input/output sizes
    # Note: systematic resampling has numba issues with Python 3.13
    smoother = build_smoother(
        log_potential=single_factor_log_potential,
        backward_sampling_fn=backward_sampling_fn,
        resampling_fn=cuthbertlib.resampling.multinomial.resampling,
        n_smoother_particles=n_smoother_particles
    )
    
    return smoother

def smooth_single_factor_manual(
    smoother,
    team_filter_states,
    team_model_inputs,
    key: jax.Array = jax.random.PRNGKey(0)
):
    """
    Smooth a single team's trajectory using manual backward pass.
    
    team_filter_states: sequence of filter states for this team
    team_model_inputs: sequence of model inputs (with time_delta, etc.)
    """
    T = len(team_filter_states)
    if T == 0:
        return []
    
    # Initialize from final filter state
    smoother_state = smoother.convert_filter_to_smoother_state(
        team_filter_states[-1],
        model_inputs=team_model_inputs[-1],
        key=key
    )
    
    smoothed_states = [smoother_state]
    
    # Backward pass
    for t in range(T - 2, -1, -1):
        key, subkey = jax.random.split(key)
        prepare_state = smoother.smoother_prepare(
            team_filter_states[t],
            team_model_inputs[t],
            key=subkey
        )
        smoother_state = smoother.smoother_combine(
            prepare_state, smoother_state
        )
        smoothed_states.append(smoother_state)
    
    # Reverse to get chronological order
    return smoothed_states[::-1]

def e_step_factSMC(
    params: dict[str, jax.Array],
    jax_data: ResultData,
    num_teams: int,
    N_filter: int,
    N_smoother: int,
    key: jax.Array = jax.random.PRNGKey(0)
):
    """
    E-step: Run forward filter and backward smoothing for all teams.
    
    Returns:
        all_team_smoothed: dict mapping team_id to list of smoothed states
        log_likelihood: final log normalizing constant from filter
        team_sequences: team match sequences for M-step
    """
    # Forward filter
    smc_filter, factorializer = build_factsmc_model(
        N=N_filter, num_teams=num_teams, params=params
    )
    
    init_state, local_filter_states, final_factorial_state = (
        cuthbert.factorial.filter(
            filter_obj=smc_filter,
            factorializer=factorializer,
            model_inputs=jax_data,
            output_factorial=False,
            key=key
        )
    )
    print(f"Final log-likelihood: {final_factorial_state.log_normalizing_constant:.4f}")
    
    # ============ Extract Team Sequences ============
    team_sequences = extract_team_sequences(
        local_filter_states, jax_data, num_teams
    )
    
    # ============ Smooth Each Team ============
    smoother = build_single_factor_smoother(params, N_smoother)
    
    all_team_smoothed = {}
    for team_id in tqdm(range(num_teams), desc="Smoothing teams"):
        seq = team_sequences[team_id]
        
        if len(seq['filter_states']) == 0:
            # Team has no matches
            all_team_smoothed[team_id] = []
            continue
        
        # Build model inputs for this team's matches
        timestamps = seq['timestamps']
        time_deltas = jnp.diff(timestamps, prepend=0)
        
        team_model_inputs = [
            {'time_delta': float(time_deltas[i]), 'timestamp': float(timestamps[i])}
            for i in range(len(timestamps))
        ]
        
        key, subkey = jax.random.split(key)
        smoothed = smooth_single_factor_manual(
            smoother=smoother,
            team_filter_states=seq['filter_states'],
            team_model_inputs=team_model_inputs,
            key=subkey
        )
        
        all_team_smoothed[team_id] = smoothed
    
    return all_team_smoothed, float(final_factorial_state.log_normalizing_constant), team_sequences

# =========== M-STEP ============

def compute_ou_log_likelihood_single_transition(state_prev, state, time_delta, kappa, params):
    """
    Log p(state | state_prev) under OU dynamics.
    """
    phi = jnp.exp(-kappa * time_delta)
    mean = params['init_mean'] + phi * (state_prev - params['init_mean'])
    cov = params['init_cov'] - phi**2 * params['init_cov']
    cov = cov + jnp.eye(2) * 1e-8
    
    # Multivariate normal log PDF
    diff = state - mean
    log_det = jnp.linalg.slogdet(cov)[1]
    mahalanobis = diff @ jnp.linalg.solve(cov, diff)
    return -0.5 * (2 * jnp.log(2 * jnp.pi) + log_det + mahalanobis)

def expected_log_likelihood_kappa(kappa, all_team_transitions, params):
    """
    Compute expected log-likelihood under smoothed distributions.
    
    all_team_transitions: list of (state_prev_particles, state_particles, 
                                   log_weights_prev, log_weights, time_delta)
    """
    total_loglik = 0.0
    
    for state_prev, state, log_w_prev, log_w, dt in all_team_transitions:
        # Compute log p(state | state_prev, kappa) for all particle pairs
        # This is O(N^2) - use vmap for efficiency
        def log_trans_for_pair(sprev, s):
            return compute_ou_log_likelihood_single_transition(sprev, s, dt, kappa, params)
        
        # Vectorize over particles
        log_trans = jax.vmap(lambda sp: jax.vmap(lambda s: log_trans_for_pair(sp, s))(state))(state_prev)
        
        # Expected value under smoothed distribution: sum over weighted particles
        weights_prev = jnp.exp(log_w_prev - jax.scipy.special.logsumexp(log_w_prev))
        weights = jnp.exp(log_w - jax.scipy.special.logsumexp(log_w))
        
        expected_trans = jnp.sum(log_trans * weights_prev[:, None] * weights[None, :])
        total_loglik += expected_trans
    
    return -total_loglik  # Return negative for minimization

# Optimize kappa - using sampling for tractability
def make_kappa_loss_fn(params, all_team_transitions, max_transitions=100):
    """Create a JIT-compiled loss function for kappa using a sample of transitions.
    
    Uses chunked computation to avoid slow XLA compilation.
    """
    
    # Sample transitions if too many
    n_trans = len(all_team_transitions)
    if n_trans > max_transitions:
        # Use a deterministic sample for reproducibility
        indices = jnp.linspace(0, n_trans - 1, max_transitions).astype(int)
        sampled_transitions = [all_team_transitions[int(i)] for i in indices]
    else:
        sampled_transitions = all_team_transitions
    
    # Stack all transitions for efficient computation
    state_prev_list = jnp.stack([t[0] for t in sampled_transitions])  # (T, N, 2)
    state_list = jnp.stack([t[1] for t in sampled_transitions])      # (T, N, 2)
    log_w_prev_list = jnp.stack([t[2] for t in sampled_transitions])   # (T, N)
    log_w_list = jnp.stack([t[3] for t in sampled_transitions])        # (T, N)
    dt_list = jnp.array([t[4] for t in sampled_transitions])          # (T,)
    
    # Pre-compute weights (softmax normalization)
    weights_prev_list = jax.nn.softmax(log_w_prev_list, axis=-1)  # (T, N)
    weights_list = jax.nn.softmax(log_w_list, axis=-1)            # (T, N)
    
    @jax.jit
    def compute_transition_loss(kappa, state_prev, state, weights_prev, weights, dt):
        """Compute expected log-likelihood for a single transition using chunked computation."""
        # Compute OU parameters
        phi = jnp.exp(-kappa * dt)
        mean = params['init_mean'] + phi * (state_prev - params['init_mean'])  # (N, 2)
        cov = params['init_cov'] - phi**2 * params['init_cov']
        cov = cov + jnp.eye(2) * 1e-8
        
        n_particles = state_prev.shape[0]
        
        # Process in chunks to avoid large computation graphs
        chunk_size = min(32, n_particles)
        total_expected = 0.0
        
        for i in range(0, n_particles, chunk_size):
            chunk_end = min(i + chunk_size, n_particles)
            mean_chunk = mean[i:chunk_end]  # (chunk, 2)
            
            # Expand for broadcasting: (chunk, 1, 2) and (1, N, 2)
            mean_expanded = mean_chunk[:, None, :]  # (chunk, 1, 2)
            state_expanded = state[None, :, :]       # (1, N, 2)
            
            # Compute diff for all pairs in chunk
            diff = state_expanded - mean_expanded  # (chunk, N, 2)
            
            # Mahalanobis distance
            inv_cov = jnp.linalg.solve(cov, jnp.eye(2))
            mahalanobis = jnp.sum(diff @ inv_cov * diff, axis=-1)  # (chunk, N)
            
            log_det = jnp.linalg.slogdet(cov)[1]
            log_trans_chunk = -0.5 * (2 * jnp.log(2 * jnp.pi) + log_det + mahalanobis)
            
            # Expected value for this chunk
            expected_chunk = jnp.sum(log_trans_chunk * weights_prev[i:chunk_end][:, None] * weights[None, :])
            total_expected += expected_chunk
        
        return total_expected
    
    def loss_fn(kappa):
        """Compute negative expected log-likelihood for OU transitions."""
        total_loss = 0.0
        n_valid = 0.0
        
        # Process transitions in batches
        batch_size = min(50, len(sampled_transitions))
        for batch_start in range(0, len(sampled_transitions), batch_size):
            batch_end = min(batch_start + batch_size, len(sampled_transitions))
            for i in range(batch_start, batch_end):
                dt = dt_list[i]
                
                expected = compute_transition_loss(
                    kappa,
                    state_prev_list[i], state_list[i],
                    weights_prev_list[i], weights_list[i],
                    dt
                )
                
                # Only count transitions with dt >= 1.0
                valid_mask = jnp.where(dt >= 1.0, 1.0, 0.0)
                total_loss -= expected * valid_mask
                n_valid = n_valid + valid_mask
        
        # Return average loss (avoid division by zero)
        return total_loss / jnp.maximum(n_valid, 1.0)
    
    return loss_fn


def _sample_transitions_stratified(team_transitions: dict, max_transitions: int) -> list:
    """
    Sample transitions using stratified sampling.
    
    Each team contributes equally (up to their available transitions),
    ensuring representation from all teams.
    
    Args:
        team_transitions: dict[team_id] -> list of transitions
        max_transitions: maximum total transitions to sample
        
    Returns:
        list of sampled transitions
    """
    if max_transitions == -1:
        # Use all transitions
        return [t for transitions in team_transitions.values() for t in transitions]
    
    # Count teams with transitions
    active_teams = {tid: trans for tid, trans in team_transitions.items() if len(trans) > 0}
    n_teams = len(active_teams)
    
    if n_teams == 0:
        return []
    
    # Calculate per-team allocation (stratified)
    # Ensure at least 1 per team if possible, otherwise distribute evenly
    if max_transitions >= n_teams:
        # Can sample at least 1 from each team
        per_team = max_transitions // n_teams
        remainder = max_transitions % n_teams
    else:
        # Fewer transitions than teams - sample 1 from each team until we hit the limit
        per_team = 1
        remainder = 0
    
    sampled = []
    team_ids = list(active_teams.keys())
    
    for i, team_id in enumerate(team_ids):
        transitions = active_teams[team_id]
        # Add 1 extra for the first 'remainder' teams
        n_take = min(per_team + (1 if i < remainder else 0), len(transitions))
        if n_take > 0:
            sampled.extend(transitions[-n_take:])
        
        # Stop if we've reached the limit
        if len(sampled) >= max_transitions:
            break
    
    return sampled[:max_transitions]


def update_kappa(params, team_transitions: dict, learning_rate=0.1, max_steps=5, max_transitions=100):
    """Update kappa using gradient descent with stratified sampling."""
    import optax
    
    # Flatten transitions for counting
    all_transitions = [t for transitions in team_transitions.values() for t in transitions]
    
    if len(all_transitions) == 0:
        return params['kappa']
    
    # Use stratified sampling
    sampled_transitions = _sample_transitions_stratified(team_transitions, max_transitions)
    
    if max_transitions == -1:
        print(f"  Optimizing kappa (using all {len(all_transitions)} transitions)...")
    else:
        print(f"  Optimizing kappa (using {len(sampled_transitions)} transitions via stratified sampling from {len(all_transitions)} total)...")
    
    # Create JIT-compiled loss function
    loss_fn = make_kappa_loss_fn(params, sampled_transitions)
    
    # Use optax for optimization
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params['kappa'])
    
    current_kappa = params['kappa']
    best_loss = float('inf')
    best_kappa = current_kappa
    
    for step in range(max_steps):
        loss_val, grads = jax.value_and_grad(loss_fn)(current_kappa)
        
        # Check for NaN
        if jnp.isnan(loss_val) or jnp.any(jnp.isnan(grads)):
            print(f"    NaN detected, stopping optimization")
            break
        
        updates, opt_state = optimizer.update(grads, opt_state)
        current_kappa = optax.apply_updates(current_kappa, updates)
        
        # Ensure kappa stays positive
        current_kappa = jnp.maximum(current_kappa, 1e-6)
        
        loss_float = float(loss_val)
        if loss_float < best_loss:
            best_loss = loss_float
            best_kappa = current_kappa
        
        print(f"    Step {step}: loss={loss_float:.4f}, kappa={float(current_kappa):.6f}")
    
    return best_kappa

def expected_observation_log_likelihood(alpha, beta, match_data, 
                                       home_smoothed, away_smoothed, params):
    """
    Compute E[log p(y | x_home, x_away, alpha, beta)] under smoothed distributions.
    
    home_smoothed: smoothed state for home team
    away_smoothed: smoothed state for away team
    """
    from cuthberto_carlos.bivariate_poisson import loglik_grid
    
    # Extract smoothed particles and weights
    home_particles = home_smoothed.particles  # (N, 2)
    home_weights = jax.nn.softmax(home_smoothed.log_weights)
    
    away_particles = away_smoothed.particles  # (N, 2)
    away_weights = jax.nn.softmax(away_smoothed.log_weights)
    
    home_score = match_data['home_score']
    away_score = match_data['away_score']
    
    # Compute log-likelihood for all particle pairs
    def log_lik_for_pair(home_skill, away_skill):
        # Use the bivariate Poisson log-likelihood from the model
        # loglik_grid returns log probabilities for all score combinations
        log_probs = loglik_grid(
            home_skill, away_skill, alpha, beta, 
            max_goals=MAX_GOALS, scale=1.0
        )
        # Return log probability of observed scores
        return log_probs[home_score, away_score]
    
    # Vectorize over all pairs
    log_liks = jax.vmap(
        lambda h: jax.vmap(lambda a: log_lik_for_pair(h, a))(away_particles)
    )(home_particles)  # (N_home, N_away)
    
    # Expected value: weighted sum over all pairs
    expected_loglik = jnp.sum(
        log_liks * home_weights[:, None] * away_weights[None, :]
    )
    
    return -expected_loglik  # Return negative for minimization

def make_alpha_beta_loss_fn(params, all_matches_smoothed, max_matches=100):
    """Create a JIT-compiled loss function for alpha and beta using sampling.
    
    Uses a batched approach to avoid slow XLA compilation from nested vmap.
    """
    from cuthberto_carlos.bivariate_poisson import loglik_grid
    
    # Compute number of matches to sample (handle int, float percentage, or -1)
    n_matches_total = len(all_matches_smoothed)
    if max_matches == -1:
        n_sample = n_matches_total
    elif 0 < max_matches < 1:
        n_sample = max(1, int(max_matches * n_matches_total))
    else:
        n_sample = int(max_matches)
    
    # Sample matches if too many - take most recent
    if n_matches_total > n_sample:
        sampled_matches = all_matches_smoothed[-n_sample:]
    else:
        sampled_matches = all_matches_smoothed
    
    # Pre-extract data for efficient computation
    home_particles_list = []
    home_weights_list = []
    away_particles_list = []
    away_weights_list = []
    home_scores = []
    away_scores = []
    
    for match_data, home_s, away_s in sampled_matches:
        home_particles_list.append(home_s.particles)
        home_weights_list.append(jax.nn.softmax(home_s.log_weights))
        away_particles_list.append(away_s.particles)
        away_weights_list.append(jax.nn.softmax(away_s.log_weights))
        home_scores.append(match_data['home_score'])
        away_scores.append(match_data['away_score'])
    
    # Stack into arrays for vectorized computation
    home_particles_all = jnp.stack(home_particles_list)  # (M, N, 2)
    home_weights_all = jnp.stack(home_weights_list)      # (M, N)
    away_particles_all = jnp.stack(away_particles_list)  # (M, N, 2)
    away_weights_all = jnp.stack(away_weights_list)      # (M, N)
    home_scores_arr = jnp.array(home_scores)             # (M,)
    away_scores_arr = jnp.array(away_scores)             # (M,)
    
    # JIT-compiled inner function that processes one match
    @jax.jit
    def compute_match_loglik(alpha, beta, home_p, home_w, away_p, away_w, h_score, a_score):
        """Compute expected log-likelihood for a single match using particle sampling."""
        # Instead of NxN computation, sample particle pairs
        # This avoids the slow XLA compilation of nested vmap
        n_particles = home_p.shape[0]
        
        # Expand for broadcasting: (N, 1, 2) and (1, N, 2)
        home_expanded = home_p[:, None, :]  # (N, 1, 2)
        away_expanded = away_p[None, :, :]  # (1, N, 2)
        
        # Compute loglik for all pairs using broadcasting
        # Process in smaller chunks to avoid memory issues
        chunk_size = min(32, n_particles)
        log_liks = []
        
        for i in range(0, n_particles, chunk_size):
            home_chunk = home_expanded[i:i+chunk_size]  # (chunk, 1, 2)
            # Broadcast to (chunk, N, 2)
            home_broadcast = jnp.broadcast_to(home_chunk, (home_chunk.shape[0], n_particles, 2))
            away_broadcast = jnp.broadcast_to(away_expanded, (home_chunk.shape[0], n_particles, 2))
            
            # Flatten for vectorized computation
            home_flat = home_broadcast.reshape(-1, 2)  # (chunk*N, 2)
            away_flat = away_broadcast.reshape(-1, 2)  # (chunk*N, 2)
            
            # Compute loglik for all pairs in chunk
            log_probs_flat = jax.vmap(lambda ha: loglik_grid(ha[0], ha[1], alpha, beta, max_goals=MAX_GOALS, scale=1.0)[h_score, a_score])(jnp.stack([home_flat, away_flat], axis=1))
            log_liks.append(log_probs_flat.reshape(home_chunk.shape[0], n_particles))
        
        log_liks = jnp.concatenate(log_liks, axis=0)  # (N, N)
        
        # Expected value under smoothed distribution
        expected = jnp.sum(log_liks * home_w[:, None] * away_w[None, :])
        return expected
    
    def loss_fn(ab):
        alpha, beta = ab[0], ab[1]
        total = 0.0
        
        # Process matches in batches to reduce compilation time
        batch_size = min(50, len(sampled_matches))
        for batch_start in range(0, len(sampled_matches), batch_size):
            batch_end = min(batch_start + batch_size, len(sampled_matches))
            for i in range(batch_start, batch_end):
                expected = compute_match_loglik(
                    alpha, beta,
                    home_particles_all[i], home_weights_all[i],
                    away_particles_all[i], away_weights_all[i],
                    home_scores_arr[i], away_scores_arr[i]
                )
                total -= expected
        
        # Return average loss
        return total / len(sampled_matches)
    
    return loss_fn


def update_alpha_beta(params, all_matches_smoothed, learning_rate=0.1, max_steps=5, max_matches=100):
    """
    Update alpha and beta using gradient descent with sampling.
    
    Args:
        max_matches: Number of matches to sample (int), percentage (0-1), or -1 for all.
    """
    import optax
    
    if len(all_matches_smoothed) == 0:
        return {'alpha': params['alpha'], 'beta': params['beta']}
    
    # Compute number of matches to sample
    n_matches_total = len(all_matches_smoothed)
    if max_matches == -1:
        n_sample = n_matches_total
        print(f"  Optimizing alpha/beta (using all {n_matches_total} matches)...")
    elif 0 < max_matches < 1:
        n_sample = max(1, int(max_matches * n_matches_total))
        print(f"  Optimizing alpha/beta (using {n_sample} matches = {max_matches*100:.1f}% of {n_matches_total} total)...")
    else:
        n_sample = int(max_matches)
        print(f"  Optimizing alpha/beta (using up to {n_sample} matches from {n_matches_total} total)...")
    
    # Use computed sample size for loss function
    loss_fn = make_alpha_beta_loss_fn(params, all_matches_smoothed, max_matches=n_sample)
    
    # Use optax for optimization
    optimizer = optax.adam(learning_rate)
    current_ab = jnp.array([params['alpha'], params['beta']])
    opt_state = optimizer.init(current_ab)
    
    best_loss = float('inf')
    best_ab = current_ab
    
    for step in range(max_steps):
        loss_val, grads = jax.value_and_grad(loss_fn)(current_ab)
        
        # Check for NaN
        if jnp.isnan(loss_val) or jnp.any(jnp.isnan(grads)):
            print(f"    NaN detected, stopping optimization")
            break
        
        updates, opt_state = optimizer.update(grads, opt_state)
        current_ab = optax.apply_updates(current_ab, updates)
        
        loss_float = float(loss_val)
        if loss_float < best_loss:
            best_loss = loss_float
            best_ab = current_ab
        
        print(f"    Step {step}: loss={loss_float:.4f}, alpha={float(current_ab[0]):.4f}, beta={float(current_ab[1]):.4f}")
    
    return {'alpha': best_ab[0], 'beta': best_ab[1]}

def update_initial_distribution(all_team_initial_states):
    """
    Update initial mean and covariance from smoothed states at t=0.
    
    all_team_initial_states: list of SmootherState for each team's first appearance
    """
    # Collect weighted particles from all teams' initial states
    all_particles = []
    all_weights = []
    
    for state in all_team_initial_states:
        particles = state.particles  # (N, 2)
        weights = jax.nn.softmax(state.log_weights)
        all_particles.append(particles)
        all_weights.append(weights)
    
    # Stack all particles
    all_particles = jnp.concatenate(all_particles, axis=0)  # (F*N, 2)
    all_weights = jnp.concatenate(all_weights, axis=0)
    all_weights = all_weights / jnp.sum(all_weights)  # Normalize
    
    # Compute weighted mean and covariance
    new_init_mean = jnp.sum(all_particles * all_weights[:, None], axis=0)
    
    centered = all_particles - new_init_mean
    new_init_cov = jnp.sum(
        centered[:, :, None] * centered[:, None, :] * all_weights[:, None, None],
        axis=0
    )
    
    return {'init_mean': new_init_mean, 'init_cov': new_init_cov}


def _compute_num_matches(max_matches: float, total: int) -> int:
    """Compute number of matches from max_matches (int or percentage)."""
    if 0 < max_matches < 1:
        # Percentage: use fraction of total
        return max(1, int(max_matches * total))
    elif max_matches == -1:
        # Use all matches
        return total
    else:
        # Integer: use specified number
        return int(max_matches)


def update_friendly_neutral_scales(params, all_matches_smoothed, learning_rate=0.1, max_steps=5, max_matches=100):
    """
    Update friendly_scale and neutral_scale using gradient descent.
    
    These scales affect how much friendly and neutral matches contribute to the likelihood.
    Higher scale = less weight on those matches.
    
    Args:
        max_matches: Number of matches to sample (int), percentage (0-1), or -1 for all.
    
    Note: Uses sampling and JIT compilation for efficiency.
    """
    import optax
    from cuthberto_carlos.bivariate_poisson import loglik_grid
    
    if len(all_matches_smoothed) == 0:
        return {'friendly_scale': params.get('friendly_scale', FRIENDLY_SCALE), 
                'neutral_scale': params.get('neutral_scale', NEUTRAL_SCALE)}
    
    # Compute number of matches to sample
    n_sample = _compute_num_matches(max_matches, len(all_matches_smoothed))
    
    # Sample matches if needed - take most recent
    if n_sample >= len(all_matches_smoothed):
        sampled_matches = all_matches_smoothed
        print(f"  Optimizing scales (using all {len(all_matches_smoothed)} matches)...")
    else:
        sampled_matches = all_matches_smoothed[-n_sample:]
        print(f"  Optimizing scales (using {n_sample} most recent matches from {len(all_matches_smoothed)} total)...")
    
    # Separate sampled matches by type
    friendly_matches = []
    neutral_matches = []
    regular_matches = []
    
    for match_data, home_s, away_s in sampled_matches:
        if match_data.get('is_friendly', False):
            friendly_matches.append((match_data, home_s, away_s))
        elif match_data.get('is_neutral', False):
            neutral_matches.append((match_data, home_s, away_s))
        else:
            regular_matches.append((match_data, home_s, away_s))
    
    print(f"    Breakdown: {len(friendly_matches)} friendly, {len(neutral_matches)} neutral, {len(regular_matches)} regular")
    
    # If no friendly or neutral matches, return current values
    if len(friendly_matches) == 0 and len(neutral_matches) == 0:
        return {'friendly_scale': params.get('friendly_scale', FRIENDLY_SCALE), 
                'neutral_scale': params.get('neutral_scale', NEUTRAL_SCALE)}
    
    # Pre-compute expected log-likelihoods using JIT-compiled function
    alpha = params['alpha']
    beta = params['beta']
    
    def make_batch_loglik_fn(alpha, beta):
        @jax.jit
        def compute_batch_logliks(home_particles, home_weights, away_particles, away_weights, home_score, away_score):
            """JIT-compiled computation for a single match using chunked approach."""
            n_particles = home_particles.shape[0]
            chunk_size = min(32, n_particles)
            total_loglik = 0.0
            
            for i in range(0, n_particles, chunk_size):
                chunk_end = min(i + chunk_size, n_particles)
                home_chunk = home_particles[i:chunk_end]  # (chunk, 2)
                
                # Expand for broadcasting
                home_expanded = home_chunk[:, None, :]  # (chunk, 1, 2)
                away_expanded = away_particles[None, :, :]  # (1, N, 2)
                
                # Broadcast to (chunk, N, 2)
                home_broadcast = jnp.broadcast_to(home_expanded, (home_chunk.shape[0], n_particles, 2))
                away_broadcast = jnp.broadcast_to(away_expanded, (home_chunk.shape[0], n_particles, 2))
                
                # Flatten for vectorized computation
                home_flat = home_broadcast.reshape(-1, 2)
                away_flat = away_broadcast.reshape(-1, 2)
                
                # Compute loglik for all pairs in chunk
                def compute_pair(ha):
                    log_probs = loglik_grid(ha[0], ha[1], alpha, beta, max_goals=MAX_GOALS, scale=1.0)
                    return log_probs[home_score, away_score]
                
                log_probs_flat = jax.vmap(compute_pair)(jnp.stack([home_flat, away_flat], axis=1))
                log_liks_chunk = log_probs_flat.reshape(home_chunk.shape[0], n_particles)
                
                # Accumulate weighted sum
                total_loglik += jnp.sum(log_liks_chunk * home_weights[i:chunk_end][:, None] * away_weights[None, :])
            
            return total_loglik
        return compute_batch_logliks
    
    compute_batch_logliks = make_batch_loglik_fn(alpha, beta)
    
    # Pre-compute expected log-likelihoods
    def compute_match_expected_loglik(match_data, home_s, away_s):
        """Compute expected log-likelihood for a single match."""
        home_weights = jax.nn.softmax(home_s.log_weights)
        away_weights = jax.nn.softmax(away_s.log_weights)
        
        loglik = compute_batch_logliks(
            home_s.particles, home_weights,
            away_s.particles, away_weights,
            match_data['home_score'], match_data['away_score']
        )
        return float(loglik)
    
    # Compute in parallel using multiple CPU cores
    print(f"    Pre-computing log-likelihoods (parallel)...")
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    import os
    import time
    
    # Print system info
    print(f"    JAX devices: {jax.devices()}")
    print(f"    CPU cores available: {os.cpu_count()}")
    
    # Use ThreadPoolExecutor since JAX operations release the GIL
    n_workers = min(8, os.cpu_count() or 4)
    print(f"    Using {n_workers} threads")
    
    start_time = time.time()
    executor = ThreadPoolExecutor(max_workers=n_workers)
    try:
        friendly_logliks = list(executor.map(lambda m: compute_match_expected_loglik(*m), friendly_matches))
        neutral_logliks = list(executor.map(lambda m: compute_match_expected_loglik(*m), neutral_matches))
        regular_logliks = list(executor.map(lambda m: compute_match_expected_loglik(*m), regular_matches))
    finally:
        executor.shutdown(wait=True)
    
    elapsed = time.time() - start_time
    total_matches = len(friendly_matches) + len(neutral_matches) + len(regular_matches)
    print(f"    Computed {total_matches} matches in {elapsed:.2f}s ({total_matches/elapsed:.1f} matches/sec)")
    
    # Convert to JAX arrays
    friendly_logliks = jnp.array(friendly_logliks) if friendly_logliks else jnp.array([0.0])
    neutral_logliks = jnp.array(neutral_logliks) if neutral_logliks else jnp.array([0.0])
    regular_logliks = jnp.array(regular_logliks) if regular_logliks else jnp.array([0.0])
    
    # Simple loss function (no JIT needed - just arithmetic)
    def loss_fn(scales):
        friendly_scale, neutral_scale = scales[0], scales[1]
        
        # Regular matches contribute directly
        regular_loss = -jnp.sum(regular_logliks)
        
        # Friendly and neutral matches are scaled
        friendly_loss = -jnp.sum(friendly_logliks) / friendly_scale if len(friendly_matches) > 0 else 0.0
        neutral_loss = -jnp.sum(neutral_logliks) / neutral_scale if len(neutral_matches) > 0 else 0.0
        
        total_loss = regular_loss + friendly_loss + neutral_loss
        n_matches = len(regular_matches) + len(friendly_matches) + len(neutral_matches)
        
        return total_loss / jnp.maximum(n_matches, 1)
    
    # Use optax for optimization
    optimizer = optax.adam(learning_rate)
    current_scales = jnp.array([
        float(params.get('friendly_scale', FRIENDLY_SCALE)),
        float(params.get('neutral_scale', NEUTRAL_SCALE))
    ])
    opt_state = optimizer.init(current_scales)
    
    best_loss = float('inf')
    best_scales = current_scales
    
    for step in range(max_steps):
        loss_val, grads = jax.value_and_grad(loss_fn)(current_scales)
        
        if jnp.isnan(loss_val) or jnp.any(jnp.isnan(grads)):
            print(f"    NaN detected, stopping optimization")
            break
        
        updates, opt_state = optimizer.update(grads, opt_state)
        current_scales = optax.apply_updates(current_scales, updates)
        
        # Ensure scales stay >= 1.0 (minimum scale)
        current_scales = jnp.maximum(current_scales, 1.0)
        
        loss_float = float(loss_val)
        if loss_float < best_loss:
            best_loss = loss_float
            best_scales = current_scales
        
        print(f"    Step {step}: loss={loss_float:.4f}, friendly_scale={float(current_scales[0]):.4f}, neutral_scale={float(current_scales[1]):.4f}")
    
    return {'friendly_scale': float(best_scales[0]), 'neutral_scale': float(best_scales[1])}


def prepare_m_step_data(all_team_smoothed, team_sequences, jax_data):
    """
    Prepare data for M-step from smoothed states.
    
    Returns:
        team_transitions: dict[team_id] -> list of (state_prev, state, log_w_prev, log_w, dt)
        match_smoothed: list of (match_data, home_smoothed, away_smoothed)
        team_initial_states: list of initial smoothed states per team
    """
    team_transitions = {}  # Organized by team for stratified sampling
    match_smoothed = []
    team_initial_states = []
    
    # Extract team transitions
    for team_id, smoothed_states in all_team_smoothed.items():
        if len(smoothed_states) == 0:
            continue
            
        # Get initial state (first smoothed state)
        team_initial_states.append(smoothed_states[0])
        
        # Extract transitions for this team
        seq = team_sequences[team_id]
        timestamps = seq['timestamps']
        
        team_transitions[team_id] = []
        for i in range(1, len(smoothed_states)):
            dt = float(timestamps[i] - timestamps[i-1])
            team_transitions[team_id].append((
                smoothed_states[i-1].particles,
                smoothed_states[i].particles,
                smoothed_states[i-1].log_weights,
                smoothed_states[i].log_weights,
                dt
            ))
    
    # Extract match-level smoothed states
    num_matches = len(jax_data.match_index)
    for t in range(num_matches):
        home_id = int(jax_data.home_team_id[t])
        away_id = int(jax_data.away_team_id[t])
        
        # Find the smoothed state for this match for each team
        home_seq = team_sequences[home_id]
        away_seq = team_sequences[away_id]
        
        # Find index in team's match sequence
        home_idx = None
        away_idx = None
        
        for i, ts in enumerate(home_seq['timestamps']):
            if ts == jax_data.timestamp[t]:
                home_idx = i
                break
        
        for i, ts in enumerate(away_seq['timestamps']):
            if ts == jax_data.timestamp[t]:
                away_idx = i
                break
        
        if home_idx is not None and away_idx is not None:
            home_smoothed = all_team_smoothed[home_id][home_idx]
            away_smoothed = all_team_smoothed[away_id][away_idx]
            
            match_data = {
                'home_score': int(jax_data.home_score[t]),
                'away_score': int(jax_data.away_score[t]),
                'timestamp': float(jax_data.timestamp[t]),
                'is_friendly': bool(jax_data.friendly[t]),
                'is_neutral': bool(jax_data.neutral[t]),
            }
            
            match_smoothed.append((match_data, home_smoothed, away_smoothed))
    
    return team_transitions, match_smoothed, team_initial_states


def m_step_factSMC(
    params: dict[str, jax.Array],
    all_team_smoothed: dict,
    team_sequences: dict,
    jax_data: ResultData,
    learning_rate: float = 0.05,
    max_transitions: int = 100,
    max_matches: float = 100,
):
    """
    Perform M-step: update all parameters given smoothed states.
    """
    # Prepare data for M-step
    team_transitions, match_smoothed, team_initial_states = prepare_m_step_data(
        all_team_smoothed, team_sequences, jax_data
    )
    
    new_params = params.copy()
    
    print(f"  M-step: {sum(len(t) for t in team_transitions.values())} transitions, {len(match_smoothed)} matches, {len(team_initial_states)} initial states")
    
    # 1. Update kappa (dynamics parameter) - only if we have transitions
    total_transitions = sum(len(t) for t in team_transitions.values())
    if total_transitions > 0:
        try:
            new_kappa = update_kappa(params, team_transitions, learning_rate, max_transitions=max_transitions)
            new_params['kappa'] = new_kappa
        except Exception as e:
            print(f"  Warning: Failed to update kappa: {e}")
            import traceback
            traceback.print_exc()
    
    # 2. Update alpha, beta (observation parameters) - only if we have matches
    if len(match_smoothed) > 0:
        try:
            ab_updates = update_alpha_beta(params, match_smoothed, learning_rate, max_matches=max_matches)
            new_params.update(ab_updates)
        except Exception as e:
            print(f"  Warning: Failed to update alpha/beta: {e}")
            import traceback
            traceback.print_exc()
    
    # 3. Update initial distribution - only if we have initial states
    if len(team_initial_states) > 0:
        try:
            init_updates = update_initial_distribution(team_initial_states)
            new_params.update(init_updates)
        except Exception as e:
            print(f"  Warning: Failed to update initial distribution: {e}")
            import traceback
            traceback.print_exc()
    
    # 4. Update friendly_scale and neutral_scale - only if we have matches
    if len(match_smoothed) > 0:
        try:
            scale_updates = update_friendly_neutral_scales(params, match_smoothed, learning_rate, max_matches=max_matches)
            new_params.update(scale_updates)
        except Exception as e:
            print(f"  Warning: Failed to update friendly/neutral scales: {e}")
            import traceback
            traceback.print_exc()
    
    return new_params

# =========== Smoothing EM Algorithm ============

def factSMC_smoothing(
    initial_params: dict[str, jax.Array],
    jax_data: ResultData,
    num_teams: int,
    N_filter: int,
    N_smoother: int,
    num_iterations: int = 10,
    key: jax.Array = jax.random.PRNGKey(0),
    max_transitions: int = 100,
    max_matches: float = 100,
):
    """
    https://state-space-models.github.io/cuthbert/api_cuthbert/factorial/

    Perform smoothing optimization for factorial SMC. Smoothing is performed for each team since team factors are independent across teams.
    1. E-step
        1. Run forward factorial SMC filter to get the final state and log-likelihood.
        2.Extract per-team sequences of states from the final factorial state.
            - find all matches played
            - extract particles
            - reorganize into continuous time series for each team
        3. build backward sampler with single-factor dynamics
        4. Run backward smoothing to get smoothed states for each team.
    5. M-step - update kappa, alpha, beta, init_mean, init_cov, friendly_scale, neutral_scale
        - Update model parameters to get maximized expected log-likelihood

    can be parallelized across teams.

    Args:
        initial_params (dict[str, jax.Array]): _description_
        jax_data (ResultData): _description_
        num_teams (int): _description_
        N_filter (int): Number of filter particles
        N_smoother (int): Number of smoother particles
        num_iterations (int, optional): Number of EM iterations. Defaults to 10.
        key (jax.Array, optional): Random key. Defaults to jax.random.PRNGKey(0).
        max_transitions (int, optional): Number of recent transitions to use for kappa optimization. Defaults to 100.
        max_matches (float, optional): Number of recent matches to use for optimization. 
            If int >= 1: use that many matches. If 0 < value < 1: use as percentage of total matches. 
            Use -1 for all matches. Defaults to 100.

    Returns:
        dict[str, jax.Array]: Final optimized parameters
    """
    import os
    from pathlib import Path
    
    params = initial_params
    
    # Create output directory
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(exist_ok=True)

    for iteration in tqdm(range(num_iterations), desc="Smoothing EM Iterations"):
        print(f"\n{'='*60}")
        print(f"EM Iteration {iteration + 1}/{num_iterations}")
        print(f"{'='*60}")
        
        # ============ E-STEP ===========
        all_team_smoothed, log_likelihood, team_sequences = e_step_factSMC(
            params=params, 
            jax_data=jax_data, 
            num_teams=num_teams, 
            N_filter=N_filter,
            N_smoother=N_smoother, 
            key=key
        )
        
        print(f"Log-likelihood: {log_likelihood:.4f}")
        
        # =========== M-STEP ===========
        params = m_step_factSMC(
            params, all_team_smoothed, team_sequences, jax_data,
            max_transitions=max_transitions,
            max_matches=max_matches
        )
        
        print(f"Updated parameters:")
        print(f"  kappa={float(params['kappa']):.4f}")
        print(f"  alpha={float(params['alpha']):.4f}")
        print(f"  beta={float(params['beta']):.4f}")
        print(f"  init_mean={params['init_mean']}")
        print(f"  init_cov={params['init_cov']}")
        print(f"  friendly_scale={float(params.get('friendly_scale', FRIENDLY_SCALE)):.4f}")
        print(f"  neutral_scale={float(params.get('neutral_scale', NEUTRAL_SCALE)):.4f}")

        key, _ = jax.random.split(key)

    # Save final parameters to JSON
    json_path = output_dir / "factsmc_smoothing_params.json"
    with open(json_path, "w") as f:
        json.dump({
            'kappa': float(params['kappa']),
            'alpha': float(params['alpha']),
            'beta': float(params['beta']),
            'init_mean': params['init_mean'].tolist(),
            'init_cov': params['init_cov'].tolist(),
            'friendly_scale': float(params.get('friendly_scale', FRIENDLY_SCALE)),
            'neutral_scale': float(params.get('neutral_scale', NEUTRAL_SCALE)),
        }, f, indent=2)
    print(f"Saved final parameters to {json_path}")
    return params

def main():
    # Print JAX device info at start
    print_jax_device_info()
    
    params = {
        'init_mean': INIT_MEAN,
        'init_cov': INIT_COV,
        'kappa': INITIAL_KAPPA,
        'alpha': INITIAL_ALPHA,
        'beta': INITIAL_BETA,
        'friendly_scale': FRIENDLY_SCALE,
        'neutral_scale': NEUTRAL_SCALE,
    }

    # init data
    pl_data, jax_data, pl_data_future, jax_data_future, id_to_name = process_data_pl(
        train_start="2020-01-01",
        train_end="2026-06-10",
        test_end="2026-07-19",
        max_goals=8,
    )
    num_teams = len(id_to_name)
    print(f"Training matches: {len(pl_data)}")
    print(f"Test/future matches: {len(pl_data_future)}")
    print(f"Number of teams: {num_teams}")
    
    # Run smoothing EM
    print("\n" + "="*60)
    print("Starting Factorial SMC Smoothing EM")
    print("="*60)
    
    final_params = factSMC_smoothing(
        initial_params=params,
        jax_data=jax_data,
        num_teams=num_teams,
        N_filter=100,  # Reduced particles for faster testing
        N_smoother=100,
        num_iterations=10,  # Reduced iterations for faster testing
        max_transitions=250,
        max_matches=0.25,  # Use 1% of matches (about 60 matches)
        key=jax.random.PRNGKey(42)
    )
    
    print("\n" + "="*60)
    print("Smoothing EM Complete!")
    print("="*60)
    print(f"Final parameters:")
    print(f"  kappa={float(final_params['kappa']):.4f}")
    print(f"  alpha={float(final_params['alpha']):.4f}")
    print(f"  beta={float(final_params['beta']):.4f}")
    print(f"  init_mean={final_params['init_mean']}")
    print(f"  init_cov={final_params['init_cov']}")
    print(f"  friendly_scale={float(final_params.get('friendly_scale', FRIENDLY_SCALE)):.4f}")
    print(f"  neutral_scale={float(final_params.get('neutral_scale', NEUTRAL_SCALE)):.4f}")


if __name__ == "__main__":
    main()