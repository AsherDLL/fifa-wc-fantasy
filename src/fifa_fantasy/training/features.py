"""Engineer training features compatible with the WC inference table.

At inference (WC), the per-(player, round) feature table has these
LightGBM-relevant columns:

    position, price_millions, is_home,
    strength_diff, rank_diff,
    squad_top_n_avg_price, opp_squad_top_n_avg_price,
    squad_rank_points,     opp_squad_rank_points,
    ownership_fraction

For training (EPL), we synthesize matching columns:

    strength_diff  = own_team_strength - opp_team_strength    (FPL strength 1-5)
    rank_diff      = None (no FIFA ranking for clubs; LightGBM treats as NaN)
    squad_*_price  = team-aggregated FPL prices per gameweek
    opp_*          = same, on opponent
    ownership_fraction = computed from the FPL selected count later if we need it

Target is `total_points` per gameweek.

The intent is not perfect parity, only structural parity: the model
learns shape from features that have the same meaning across data sets.
A row is dropped if the player played 0 minutes that gameweek (no
useful signal for fantasy-points prediction beyond DNP).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fifa_fantasy.external.football_data import (
    compute_club_elo_history, load_matches,
)
from fifa_fantasy.external.mapping import to_fd_club

TOP_N = 11

# Trailing window (in matches) for the lagged-form feature. Three matches
# is a "current form" window: long enough to smooth a single blank, short
# enough to react to a hot streak. The window is shared with WC inference
# (features/build.py) and WC label extraction (training/wc.py) so the GBM
# sees one feature with one meaning across data sets.
FORM_WINDOW = 3

def add_lagged_start_rate(player_gameweek: pd.DataFrame,
                          window: int = FORM_WINDOW) -> pd.DataFrame:
    """Attach `start_rate_lag`: trailing fraction of prior matches started.

    This is the minutes / rotation-risk signal. A benched starter scores
    near zero regardless of form or matchup, so the models need to know how
    reliably a player actually takes the pitch. On EPL we have a real
    `starts` flag per gameweek; the WC side derives the analogous signal
    from participation (round_points > 0) because the FIFA API does not
    expose per-round minutes.

    Leak-free (`.shift(1)`), grouped by (season, player_id), NaN before a
    player's first match. Idempotent like add_lagged_form.
    """
    if "start_rate_lag" in player_gameweek.columns:
        return player_gameweek
    df = player_gameweek.copy()
    if "season" not in df.columns:
        df["season"] = "single"
    df["season"] = df["season"].fillna("single")
    # `starts` is 1/0 on the EPL feed. Fall back to minutes>0 if absent.
    if "starts" in df.columns:
        started = pd.to_numeric(df["starts"], errors="coerce").fillna(0.0).clip(0, 1)
    else:
        started = (pd.to_numeric(df.get("minutes"), errors="coerce").fillna(0.0) > 0).astype(float)
    df["_started"] = started
    df = df.sort_values(_lag_sort_keys(df)).reset_index(drop=True)
    grp = df.groupby(["season", "player_id"], sort=False)["_started"]
    df["start_rate_lag"] = grp.transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    return df.drop(columns=["_started"])


def add_team_gc_form(player_gameweek: pd.DataFrame,
                     window: int = FORM_WINDOW) -> pd.DataFrame:
    """Attach `team_gc_form`: trailing mean goals conceded by the player's team.

    Personal scoring form is noise for goalkeepers (their points come from
    clean sheets and saves, not from scoring). The correct recency signal
    for GK and DEF is how leaky their team has actually been. We estimate
    per-(team, gameweek) goals conceded from the full-match players'
    `goals_conceded` (a 90-minute player's value equals the team's), then
    take a trailing mean per team, shifted so the current match is excluded.

    Leak-free, grouped by (season, team_id). NaN before a team's first
    match. Idempotent.
    """
    if "team_gc_form" in player_gameweek.columns:
        return player_gameweek
    df = player_gameweek.copy()
    if "season" not in df.columns:
        df["season"] = "single"
    df["season"] = df["season"].fillna("single")
    if "goals_conceded" not in df.columns or "team_id" not in df.columns:
        df["team_gc_form"] = np.nan
        return df
    # Per-(season, team, gameweek) team goals conceded: median over the
    # players who played a near-full match, whose personal goals_conceded
    # equals the team's. Falls back to the max when nobody hit 60 minutes.
    mins = pd.to_numeric(df.get("minutes"), errors="coerce").fillna(0.0)
    gc = pd.to_numeric(df["goals_conceded"], errors="coerce")
    full = df.assign(_gc=gc, _min=mins)
    full60 = full[full["_min"] >= 60]
    team_gc = (
        full60.groupby(["season", "team_id", "gameweek"])["_gc"].median()
        .rename("team_gc").reset_index()
    )
    if team_gc.empty:
        df["team_gc_form"] = np.nan
        return df
    team_gc = team_gc.sort_values(["season", "team_id", "gameweek"]).reset_index(drop=True)
    g = team_gc.groupby(["season", "team_id"], sort=False)["team_gc"]
    team_gc["team_gc_form"] = g.transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    return df.merge(
        team_gc[["season", "team_id", "gameweek", "team_gc_form"]],
        on=["season", "team_id", "gameweek"], how="left",
    )


def add_lagged_form(player_gameweek: pd.DataFrame,
                    window: int = FORM_WINDOW) -> pd.DataFrame:
    """Attach `form_lag`: trailing mean of prior-match fantasy points.

    Grouped by (season, player_id) because FPL `player_id` (the element id)
    is only unique within a season; without the season key a player's form
    would bleed across season boundaries. Sorted by gameweek so the shift
    is a true "previous match" shift.

    Leak-free: `.shift(1)` drops the current gameweek from the window, so a
    row's own label is never part of its own feature. The first match of a
    (season, player) gets NaN.

    Idempotent: if `form_lag` already exists (the caller computed it on the
    full multi-season history before a train/holdout split, which is the
    correct order), this is a no-op so the split's per-subset recompute
    does not clobber it with a truncated window.
    """
    if "form_lag" in player_gameweek.columns:
        return player_gameweek
    df = player_gameweek.copy()
    if "season" not in df.columns:
        df["season"] = "single"
    df["season"] = df["season"].fillna("single")
    df = df.sort_values(_lag_sort_keys(df)).reset_index(drop=True)
    grp = df.groupby(["season", "player_id"], sort=False)["total_points"]
    df["form_lag"] = grp.transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    return df


def _lag_sort_keys(df: pd.DataFrame) -> list[str]:
    """Sort keys for the pre-shift ordering of lagged features.

    `gameweek` alone under-determines double gameweeks (a player's two
    matches share the gameweek number); adding kickoff_time when present
    makes within-gameweek order explicit instead of resting on the input
    file's row order surviving a stable sort.
    """
    keys = ["season", "player_id", "gameweek"]
    if "kickoff_time" in df.columns:
        keys.append("kickoff_time")
    return keys


def _team_top_n_avg_price(per_team_gw: pd.DataFrame, top_n: int = TOP_N) -> pd.Series:
    return (
        per_team_gw.sort_values("price_millions", ascending=False)
        .groupby(["team_id", "gameweek"], sort=False)
        .head(top_n)
        .groupby(["team_id", "gameweek"], sort=False)["price_millions"]
        .mean()
        .rename("top_n_avg_price")
    )


def build_training_table(player_gameweek: pd.DataFrame) -> pd.DataFrame:
    """Augment the scraped FPL table with feature columns matching inference."""
    # Compute lagged form before the DNP drop so the trailing window counts
    # real calendar matches (a benched 0-point week is legitimate form). If
    # the caller already computed it on the full pre-split history, this is
    # a no-op (see add_lagged_form docstring).
    df = add_lagged_form(player_gameweek)
    df = add_lagged_start_rate(df)
    df = add_team_gc_form(df)
    df = df[df["minutes"] > 0].copy()  # drop DNPs; same as inference filter

    # Per-(team, gameweek) top-N avg price.
    top = _team_top_n_avg_price(df).reset_index()
    own = top.rename(columns={"top_n_avg_price": "squad_top_n_avg_price"})
    df = df.merge(own, on=["team_id", "gameweek"], how="left")

    opp = top.rename(columns={
        "team_id": "opponent_team_id",
        "top_n_avg_price": "opp_squad_top_n_avg_price",
    })
    df = df.merge(opp, on=["opponent_team_id", "gameweek"], how="left")

    df["strength_diff"] = (
        df["squad_top_n_avg_price"] - df["opp_squad_top_n_avg_price"]
    )

    # Ownership: vaastav 2022-23 / 2023-24 dumps lack ownership; the
    # live-API 2024-25 file may have it as a pre-divided fraction. Anything
    # missing gets a neutral 0.10. EPL absolute ownership is in the same
    # ballpark as WC absolute ownership, so this is reasonable.
    if "ownership_fraction" in df.columns:
        df["ownership_fraction"] = pd.to_numeric(
            df["ownership_fraction"], errors="coerce"
        ).fillna(0.10)
    else:
        df["ownership_fraction"] = 0.10

    df = _attach_team_elo_diff(df)

    df["target"] = df["total_points"].astype(int)
    return df.reset_index(drop=True)


def _attach_team_elo_diff(df: pd.DataFrame) -> pd.DataFrame:
    """Add `team_elo_diff` (own club Elo - opponent club Elo) at match time.

    Uses football-data.co.uk match history to roll Elo forward and joins
    each (club, kickoff) row to the latest pre-match snapshot. Missing
    data leaves NaN; LightGBM tolerates that, and on the inference side
    the same column gets populated from country_elo_diff so the GBM sees
    one feature.
    """
    matches = load_matches()
    if matches.empty or "team_name" not in df.columns or "opponent_team_name" not in df.columns:
        df["team_elo_diff"] = pd.NA
        return df
    history = compute_club_elo_history(matches)
    # Map FPL team names to football-data club names.
    df = df.copy()
    df["_fd_club"] = df["team_name"].map(to_fd_club)
    df["_fd_opp"] = df["opponent_team_name"].map(to_fd_club)
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True, errors="coerce")
    # Strip tz on both sides for asof merge.
    df["_kickoff_naive"] = df["kickoff_time"].dt.tz_localize(None)

    history = history.sort_values(["club", "date"]).reset_index(drop=True)
    history["date"] = pd.to_datetime(history["date"])

    hist_sorted = (
        history[["club", "date", "elo_before"]]
        .rename(columns={"date": "_kickoff_naive"})
        .sort_values("_kickoff_naive")
        .reset_index(drop=True)
    )

    def lookup(side_col: str, out_col: str) -> pd.Series:
        merge_left = (
            df[["_kickoff_naive", side_col]]
            .rename(columns={side_col: "club"})
            .sort_values("_kickoff_naive")
            .reset_index()
        )
        merged = pd.merge_asof(
            merge_left, hist_sorted,
            on="_kickoff_naive", by="club", direction="backward",
        )
        merged = merged.sort_values("index").set_index("index")
        return merged["elo_before"].rename(out_col)

    df["_own_elo"] = lookup("_fd_club", "_own_elo").reindex(df.index)
    df["_opp_elo"] = lookup("_fd_opp", "_opp_elo").reindex(df.index)
    df["team_elo_diff"] = pd.to_numeric(df["_own_elo"], errors="coerce") - \
                          pd.to_numeric(df["_opp_elo"], errors="coerce")
    df = df.drop(columns=["_fd_club", "_fd_opp", "_kickoff_naive", "_own_elo", "_opp_elo"])
    return df
