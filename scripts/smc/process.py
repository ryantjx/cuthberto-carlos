from datetime import datetime

import numpy as np
import pandas as pd
from jax import numpy as jnp

from cuthberto_carlos.data import DATA_URL
from cuthberto_carlos.data_types import ResultData

ORIGIN_DATE = "1872-11-30"  # Date of the first international football match


def process_data(
    start_date: str = "2000-01-01",
    end_date: str = datetime.now().strftime("%Y-%m-%d"),
    max_goals: int = 8,
    future_matches: bool = False,
) -> tuple[pd.DataFrame, ResultData, dict[int, str], dict[str, int]]:
    """Download and preprocess historical international football data.

    Args:
        start_date: Only keep matches on or after this date (inclusive).
        end_date: Only keep matches on or before this date (inclusive).
        max_goals: Removes matches where either team scored more than this many goals.
        future_matches: Whether to include matches with missing scores (future fixtures).
            Defaults to False.

    Returns:
        A tuple containing:
        - A DataFrame with match data and derived columns
        - A ResultData NamedTuple containing JAX arrays for the match data
        - A dictionary mapping team IDs to names
        - A dictionary mapping team names to IDs
    """
    origin_timestamp = pd.to_datetime(ORIGIN_DATE)

    data_all = pd.read_csv(DATA_URL)
    data_all["date"] = pd.to_datetime(data_all["date"])

    # Patch the Senegal vs Morocco AFCON 2026 final to 1-0 Senegal (not the defaulted 3-0)
    mask = (
        (data_all["home_team"] == "Morocco")
        & (data_all["away_team"] == "Senegal")
        & (data_all["date"] == "2026-01-18")
    )
    data_all.loc[mask, ["home_score", "away_score"]] = [0, 1]

    # Filter by date range
    if start_date is not None:
        data_all = data_all[data_all["date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        data_all = data_all[data_all["date"] <= pd.to_datetime(end_date)]

    # Drop future matches (missing scores) unless requested
    if not future_matches:
        data_all = data_all[
            data_all["home_score"].notna() & data_all["away_score"].notna()
        ]

    # Convert scores to int (NaN → -1 since int doesn't support NaN)
    data_all["home_score"] = data_all["home_score"].fillna(-1).astype(int)
    data_all["away_score"] = data_all["away_score"].fillna(-1).astype(int)

    # Remove matches with too many goals
    data_all = data_all[
        (data_all["home_score"] <= max_goals) & (data_all["away_score"] <= max_goals)
    ]

    # Derived columns
    data_all["timestamp_days"] = (data_all["date"] - origin_timestamp).dt.days
    data_all["friendly"] = data_all["tournament"].str.contains(
        "Friendly", case=False, na=False
    )

    # Build team dictionaries and IDs
    team_names = sorted(set(data_all["home_team"]) | set(data_all["away_team"]))
    teams_name_to_id = {name: i for i, name in enumerate(team_names)}
    teams_id_to_name = {i: name for i, name in enumerate(team_names)}
    data_all["home_team_id"] = data_all["home_team"].map(teams_name_to_id)
    data_all["away_team_id"] = data_all["away_team"].map(teams_name_to_id)

    # Extract previous timestamps for home and away teams
    num_matches = len(data_all)
    timestamps = data_all["timestamp_days"].to_numpy()
    team_ids = np.concatenate(
        [data_all["home_team_id"].to_numpy(), data_all["away_team_id"].to_numpy()]
    )
    match_positions_by_team = np.concatenate(
        [np.arange(num_matches), np.arange(num_matches)]
    )
    timestamps_by_team = np.concatenate([timestamps, timestamps])
    is_home_team = np.concatenate(
        [np.ones(num_matches, dtype=bool), np.zeros(num_matches, dtype=bool)]
    )
    order = np.lexsort((match_positions_by_team, timestamps_by_team, team_ids))
    previous_timestamps = np.zeros(2 * num_matches, dtype=timestamps.dtype)
    same_team_as_previous = team_ids[order][1:] == team_ids[order][:-1]
    previous_timestamps[order[1:]] = np.where(
        same_team_as_previous,
        timestamps_by_team[order[:-1]],
        0,
    )
    data_all["home_timestamp_previous"] = previous_timestamps[is_home_team]
    data_all["away_timestamp_previous"] = previous_timestamps[~is_home_team]

    jax_data = ResultData(
        match_index=jnp.array(data_all.index.values),
        home_team_id=jnp.array(data_all["home_team_id"].values),
        away_team_id=jnp.array(data_all["away_team_id"].values),
        home_score=jnp.array(data_all["home_score"].values),
        away_score=jnp.array(data_all["away_score"].values),
        neutral=jnp.array(data_all["neutral"].values),
        friendly=jnp.array(data_all["friendly"].values),
        timestamp=jnp.array(data_all["timestamp_days"].values),
        home_timestamp_previous=jnp.array(data_all["home_timestamp_previous"].values),
        away_timestamp_previous=jnp.array(data_all["away_timestamp_previous"].values),
    )
    return data_all, jax_data, teams_id_to_name, teams_name_to_id
