"""
Factorial SMC

"""

from functools import partial
import os
from turtle import home
from matplotlib import scale
import pandas as pd
from tqdm import tqdm

import cuthbert
import cuthbertlib
from cuthbert.factorial.smc import build_factorializer
from cuthbert.smc.particle_filter import build_filter
from cuthbert.smc import particle_filter
from cuthbertlib.resampling import no_resampling, systematic
from scripts.smc.process_data import process_data_pl

from cuthberto_carlos.data import to_jax_data, download_data
from cuthberto_carlos.data_types import ResultData, DynamicsOnlyData
from cuthberto_carlos.bivariate_poisson import loglik_grid

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, tree

MAX_GOALS = 8

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

def init_sample(
        key: jax.Array,
        model_inputs: ResultData,
        num_teams: int,
        params: dict[str, jax.Array]
):
    """
    Init states for the SMC filter.
    """
    return jax.random.multivariate_normal(key, params['init_mean'], params['init_cov'], shape=(num_teams,))

def compute_ou_dynamics(
    state_team: jax.Array,  # (2,) - single team's (attack, defence)
    time_delta: jax.Array,  # scalar - days since last match
    params: dict[str, jax.Array]
) -> tuple[jax.Array, jax.Array]:
    phi = jnp.exp(-params['kappa'] * time_delta)
    mean = params['init_mean'] + phi * (state_team - params['init_mean'])
    # phi is a scalar, so phi * INIT_COV @ phi.T = phi^2 * INIT_COV
    cov = params['init_cov'] - phi**2 * params['init_cov']
    # When time_delta == 0, phi == 1 and cov becomes a zero matrix.
    # jax.random.multivariate_normal returns NaN for a zero covariance matrix,
    # so add a tiny jitter to the diagonal to keep it positive-definite.
    cov = cov + jnp.eye(2) * 1e-8
    return mean, cov

def propagate_sample(
    key: jax.Array,
    state: jax.Array,
    model_inputs: ResultData,
    params: dict[str, jax.Array]
):
    """
    Propagate sample from a factorized model. tate: (4,) — home(2) + away(2)

    Args:
        key (jax.Array): _description_
        state (jax.Array): 2 x 2 array of attack and defense
        model_inputs (ResultData): _description_
        params (dict[str, jax.Array]): _description_
    """
    key_home, key_away = jax.random.split(key)
    dt_home = jnp.where(model_inputs.home_timestamp_previous == 0, 0, model_inputs.timestamp - model_inputs.home_timestamp_previous)
    mean_home, cov_home = compute_ou_dynamics(
        state_team=state[:2], time_delta=dt_home, params=params)
    state_home = jax.random.multivariate_normal(key_home, mean_home, cov_home)

    dt_away = jnp.where(model_inputs.away_timestamp_previous == 0, 0, model_inputs.timestamp - model_inputs.away_timestamp_previous)
    mean_away, cov_away = compute_ou_dynamics(state_team=state[2:], time_delta=dt_away, params=params)
    state_away = jax.random.multivariate_normal(key_away, mean_away, cov_away)
    return jnp.concatenate([state_home, state_away])

def log_potential(
    state_prev: jax.Array,
    state: jax.Array,
    model_inputs: ResultData,
    params: dict[str, jax.Array]
):
    """
    log potential for factorialized state. state: (4,) — home(2) + away(2)

    Args:
        state_prev (jax.Array): _description_
        state (jax.Array): _description_
        model_inputs (ResultData): _description_

    Returns:
        _type_: _description_
    """
    # lambda_1 = exp(alpha + attack_home - defence_away)
    lambda_1 = jnp.exp(params['alpha'] + state[0] - state[3])
    # lambda_2 = exp(alpha + attack_away - defence_home)
    lambda_2 = jnp.exp(params['alpha'] + state[2] - state[1])
    # lambda_3 = exp(beta)
    lambda_3 = jnp.exp(params['beta'])

    lambda_terms = - (lambda_1 + lambda_2 + lambda_3)
    home_term = model_inputs.home_score * jnp.log(lambda_1) - jax.scipy.special.gammaln(model_inputs.home_score + 1)
    away_term = model_inputs.away_score * jnp.log(lambda_2) - jax.scipy.special.gammaln(model_inputs.away_score + 1)
    
    # Sum over k = 0, 1, ..., min(home_score, away_score).
    # Use a fixed-size range (MAX_GOALS) with masking so shapes are static
    # under JAX tracing/JIT.
    k_max = jnp.minimum(model_inputs.home_score, model_inputs.away_score)
    k_range = jnp.arange(MAX_GOALS + 1)  # 0, 1, ..., MAX_GOALS
    mask = k_range <= k_max

    # log(C(n, k)) = log(n!) - log(k!) - log((n-k)!)
    log_comb_home = jax.scipy.special.gammaln(model_inputs.home_score + 1) - jax.scipy.special.gammaln(k_range + 1) - jax.scipy.special.gammaln(model_inputs.home_score - k_range + 1)
    log_comb_away = jax.scipy.special.gammaln(model_inputs.away_score + 1) - jax.scipy.special.gammaln(k_range + 1) - jax.scipy.special.gammaln(model_inputs.away_score - k_range + 1)
    log_k_factorial = jax.scipy.special.gammaln(k_range + 1)
    log_ratio = k_range * (jnp.log(lambda_3) - jnp.log(lambda_1) - jnp.log(lambda_2))

    terms = log_comb_home + log_comb_away + log_k_factorial + log_ratio
    terms = jnp.where(mask, terms, -jnp.inf)
    sum_terms = jax.scipy.special.logsumexp(terms)

    log_likelihood = lambda_terms + home_term + away_term + sum_terms

    is_friendly = getattr(model_inputs, 'friendly', False)
    is_neutral = getattr(model_inputs, 'neutral', False)
    
    # Use jnp.where for JAX-compatible conditionals
    scale = jnp.where(is_friendly, params['friendly_scale'], 1.0)
    scale = jnp.where(is_neutral, params['neutral_scale'], scale)

    return log_likelihood / scale

# home_team_id: Array
# away_team_id: Array

# def summary(factorial_state, train_data, teams_id_to_name_dict, N, num_teams):
#     # Summary
#     particles = np.array(factorial_state.particles)  # (F, P, 2)
#     weights = jax.nn.softmax(factorial_state.log_weights, axis=-1)  # (F, P)
#     weighted_means = np.sum(particles * weights[..., None], axis=1)  # (F, 2)
#     weighted_vars = np.sum((particles - weighted_means[:, None, :]) ** 2 * weights[..., None], axis=1)  # (F, 2)

#     print(f"\n{'='*60}")
#     print(f"Factorial SMC Filter Summary")
#     print(f"{'='*60}")
#     print(f"Particles per factor:    {N}")
#     print(f"Number of teams (factors): {num_teams}")
#     print(f"Number of matches:        {len(train_data)}")
#     print(f"Log normalizing constant: {float(factorial_state.log_normalizing_constant):.2f}")
#     print(f"Training period:          {train_data['date'].min().date()} to {train_data['date'].max().date()}")

#     # Top 5 teams by attack
#     print(f"\n--- Top 5 Teams by Attack (higher = better) ---")
#     attack_means = weighted_means[:, 0]
#     valid = ~np.isnan(attack_means)
#     top_attack = np.argsort(attack_means[valid])[-5:][::-1]
#     valid_indices = np.where(valid)[0]
#     for idx in top_attack:
#         team_idx = int(valid_indices[idx])
#         name = teams_id_to_name_dict.get(team_idx, f"Team {team_idx}")
#         print(f"  {name}: attack={attack_means[team_idx]:.3f}, defence={weighted_means[team_idx, 1]:.3f}")

#     # Top 5 teams by defence (higher = better, reduces opponent scoring)
#     print(f"\n--- Top 5 Teams by Defence (higher = better) ---")
#     defence_means = weighted_means[:, 1]
#     valid_def = ~np.isnan(defence_means)
#     top_defence = np.argsort(defence_means[valid_def])[-5:][::-1]
#     valid_def_indices = np.where(valid_def)[0]
#     for idx in top_defence:
#         team_idx = int(valid_def_indices[idx])
#         name = teams_id_to_name_dict.get(team_idx, f"Team {team_idx}")
#         print(f"  {name}: attack={weighted_means[team_idx, 0]:.3f}, defence={defence_means[team_idx]:.3f}")

#     # Particle diversity (ESS) for a few teams
#     print(f"\n--- Particle Diversity (ESS) ---")
#     for i in range(min(5, num_teams)):
#         ess = 1.0 / np.sum(weights[i] ** 2)
#         name = teams_id_to_name_dict.get(i, f"Team {i}")
#         print(f"  {name}: ESS = {ess:.1f} / {N}")

#     print(f"{'='*60}")

def build_factsmc_model(
        N: int, 
        num_teams: int,
        params: dict[str, jax.Array]
    ) -> tuple[particle_filter.Filter, cuthbert.factorial.smc.Factorializer]:
    """Build factorial SMC model for bivariate Poisson.
    
    - no reasampling for factorial SMC in build_filter. resampling is part of the `build_factorializer` function. Refer to https://state-space-models.github.io/cuthbert/api_cuthbert/factorial/smc/ for more information.
    """

    smc_filter = particle_filter.build_filter(
        init_sample=partial(
            init_sample,
            num_teams=num_teams,
            params=params
        ),
        propagate_sample=partial(
            propagate_sample,
            params=params
        ),
        log_potential=partial(
            log_potential,
            params=params
        ),
        n_filter_particles=N,
        resampling_fn=cuthbertlib.resampling.no_resampling.resampling, # no resampling for factorial SMC. refer 
    )
    # 
    factorializer = build_factorializer(
        get_factorial_indices=lambda model_inputs: jnp.array(
            [model_inputs.home_team_id, model_inputs.away_team_id]
        ),
        resampling_fn=cuthbertlib.resampling.systematic.resampling,
    )
    return smc_filter, factorializer

def propagate_and_predict_factsmc(
    final_state: cuthbert.factorial.smc.ParticleFilterState,
    factorializer: cuthbert.factorial.smc.Factorializer,
    match_data: ResultData,
    params: dict[str, jax.Array],
    n_samples: int = 1000,
    max_goals: int = MAX_GOALS,
    key: jax.Array = jax.random.PRNGKey(0)
):
    team_ids = jnp.array([match_data.home_team_id, match_data.away_team_id])
    # dynamics_data = DynamicsOnlyData(
    #     team_id=team_ids,
    #     timestamp=jnp.array([match_data.timestamp, match_data.timestamp]),
    #     timestamp_previous=jnp.array([match_data.home_timestamp_previous, match_data.away_timestamp_previous]),
    # )
    # 1. Extract the two teams' states from the final factorial state
    factorial_state_twoteams = jax.vmap(factorializer.extract, in_axes=(None, 0))(
        final_state,
        team_ids,
    )
    particles = factorial_state_twoteams.particles  # (2, N, 2)
    log_weights = factorial_state_twoteams.log_weights  # (2, N)

    # 2. Normalize and Resample particles according to weights
    weights = jax.nn.softmax(log_weights, axis=-1)  # (2, N)

    N = particles.shape[1]
    key_home, key_away, key_sample = jax.random.split(key, 3)
    home_indices = jax.random.choice(key_home, N, shape=(n_samples,), p=weights[0])
    away_indices = jax.random.choice(key_away, N, shape=(n_samples,), p=weights[1])
    
    home_particles = particles[0][home_indices]  # (n_samples, 2)
    away_particles = particles[1][away_indices]  # (n_samples, 2)
    
    home_dt = jnp.where(match_data.home_timestamp_previous == 0, 0, match_data.timestamp - match_data.home_timestamp_previous)
    home_mean, home_cov = compute_ou_dynamics(home_particles, home_dt, params)
    home_next = jax.random.multivariate_normal(key_sample, home_mean, home_cov)
    away_dt = jnp.where(match_data.away_timestamp_previous == 0, 0, match_data.timestamp - match_data.away_timestamp_previous)
    away_mean, away_cov = compute_ou_dynamics(away_particles, away_dt, params)
    away_next = jax.random.multivariate_normal(key_sample, away_mean, away_cov)

    # 3. Compute the bivariate Poisson parameters for each sample
    def compute_probs_for_particles(home_skill, away_skill):
        return loglik_grid(home_skill, away_skill, INITIAL_ALPHA, INITIAL_BETA, max_goals=max_goals, scale=1.0)
    log_probs_per_particle = jax.vmap(compute_probs_for_particles)(home_next, away_next)  # (n_samples, max_goals+1, max_goals+1)
    avg_log_probs = jax.scipy.special.logsumexp(log_probs_per_particle, axis=0) - jnp.log(n_samples)
    probs_grid = jnp.exp(avg_log_probs)

    # 4. compute result probabilities
    goals = jnp.arange(max_goals + 1)
    home_goals = goals[:, None]
    away_goals = goals[None, :]
    
    home_win_mask = home_goals > away_goals
    draw_mask = home_goals == away_goals
    away_win_mask = home_goals < away_goals
    
    p_home_win = jnp.sum(probs_grid * home_win_mask)
    p_draw = jnp.sum(probs_grid * draw_mask)
    p_away_win = jnp.sum(probs_grid * away_win_mask)
    
    probs_results = jnp.array([p_home_win, p_draw, p_away_win])
    
    # 6. Compute expected goals
    home_probs_marginal = jnp.sum(probs_grid, axis=1)  # Sum over away goals
    away_probs_marginal = jnp.sum(probs_grid, axis=0)  # Sum over home goals
    
    expected_home_goals = jnp.sum(goals * home_probs_marginal)
    expected_away_goals = jnp.sum(goals * away_probs_marginal)
    
    return {
        "probs_grid": probs_grid,  # (max_goals+1, max_goals+1) probability grid
        "probs_results": probs_results,  # [p_home, p_draw, p_away]
        "expected_home_goals": expected_home_goals,
        "expected_away_goals": expected_away_goals,
        "home_skills_mean": jnp.mean(home_next, axis=0),
        "away_skills_mean": jnp.mean(away_next, axis=0),
        "home_skills_std": jnp.std(home_next, axis=0),
        "away_skills_std": jnp.std(away_next, axis=0),
    }

def summarize_final_state(factorial_state, train_data, N, num_teams):
    """
    Summarize the final state of the factorial SMC filter.
    """
    particles = np.array(factorial_state.particles)  # (F, P, 2)
    weights = jax.nn.softmax(factorial_state.log_weights, axis=-1)  # (F, P)
    weighted_means = np.sum(particles * weights[..., None], axis=1)  # (F, 2)
    weighted_vars = np.sum((particles - weighted_means[:, None, :]) ** 2 * weights[..., None], axis=1)  # (F, 2)

    print(f"\n{'='*60}")
    print(f"Factorial SMC Filter Summary")
    print(f"{'='*60}")
    print(f"Particles per factor:    {N}")
    print(f"Number of teams (factors): {num_teams}")
    print(f"Number of matches:        {len(train_data)}")
    print(f"Log normalizing constant: {float(factorial_state.log_normalizing_constant):.2f}")
    print(f"Training period:          {train_data['date'].min().date()} to {train_data['date'].max().date()}")

def main():
    N = 10000
    MAX_GOALS = 8

    pl_data, jax_data, pl_data_future, jax_data_future, id_to_name = process_data_pl(
        train_start="2020-01-01",
        train_end="2026-06-10",
        test_end="2026-07-19",
        max_goals=8,
    )
    num_teams = len(id_to_name)
    print(f"Training from {pl_data['date'].min().date()} to {pl_data['date'].max().date()}")
    print(f"Number of teams: {num_teams}")
    print(f"Number of matches: {len(pl_data)}")

    # Build the factorial SMC filter. Same as SMC filter, but propagate sample is only on the 2x2 state.
    params = {
        'init_mean': INIT_MEAN,
        'init_cov': INIT_COV,
        'kappa': INITIAL_KAPPA,
        'alpha': INITIAL_ALPHA,
        'beta': INITIAL_BETA,
        'friendly_scale': FRIENDLY_SCALE,
        'neutral_scale': NEUTRAL_SCALE,
    }

    # ========================= Run Factorial SMC Filter =========================
    # smc_filter, factorializer = build_factsmc_model(N=N, num_teams=num_teams, params=new_params)
    # init_state, local_states, final_state = cuthbert.factorial.filter(
    #     filter_obj=smc_filter,
    #     factorializer=factorializer,
    #     model_inputs=jax_data,
    #     output_factorial=False,
    #     key=jax.random.PRNGKey(0)
    # )
    
    # ========================= Manual Filtering (for debugging) =========================
    # keys = jax.random.split(jax.random.PRNGKey(0), len(jax_data.match_index))
    # # map data into
    # init_model_inputs = tree.map(lambda x: x[0], jax_data)
    # factorial_state = smc_filter.init_prepare(init_model_inputs, key=keys[0])
    # factorial_state = factorializer.factorialize_init_state(factorial_state, init_model_inputs)

    # for t in tqdm(range(1, len(jax_data.match_index)), desc="Filtering"):
    #     model_inputs_t = tree.map(lambda x: x[t], jax_data)
    #     local_state = factorializer.extract_and_join(factorial_state, model_inputs_t)
    #     prep_state = smc_filter.filter_prepare(model_inputs_t, key=keys[t])
    #     filtered_joint = smc_filter.filter_combine(local_state, prep_state)
    #     factorial_state = factorializer.marginalize_and_insert(
    #         filtered_joint, factorial_state, model_inputs_t
    #     )

    # ====================================================
    # summary(factorial_state, train_data, N, num_teams)


    # output_dir = os.path.join(os.path.dirname(__file__), "output")
    # os.makedirs(output_dir, exist_ok=True)
    # output_path = os.path.join(output_dir, "fact_smc_filter_latest.npz")
    # np.savez(
    #     output_path,
    #     particles=np.array(factorial_state.particles),
    #     log_weights=np.array(factorial_state.log_weights),
    #     log_normalizing_constant=np.array(factorial_state.log_normalizing_constant),
    # )
    # print(f"\nFinal filter state saved to {output_path}")

if __name__ == "__main__":
    main()