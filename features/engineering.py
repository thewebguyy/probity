"""
features/engineering.py
-----------------------
Feature engineering for the Dixon-Coles model.

Core features (as described in the roadmap):
  1. Exponentially weighted rolling xG (attack proxy)
  2. Exponentially weighted rolling xGA (defensive proxy)
  3. Home advantage (fitted as model parameter, not a feature per se)
  4. Rest days differential (days since last match per team)
  5. Score-state adjusted xG (optional, currently approximated from goals)

All features are built from the matches table into a clean DataFrame
that the model layer ingests.

EWM weight function:
  w = exp(-λ × days_since_match)
  λ = settings.LAMBDA_DECAY (default 0.0065 ≈ half-life ~107 days)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from core.config import settings
from core.database import SyncSessionLocal
from core.models import Match, MatchStatus, Team

logger = logging.getLogger(__name__)


def load_match_dataframe(
    league: str = "NL1",
    window_days: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load finished matches from the DB into a flat DataFrame.

    Returns columns:
        match_id, match_date, season,
        home_team, away_team,
        home_goals, away_goals,
        home_xg, away_xg
    """
    window_days = window_days or settings.MODEL_CALIBRATION_WINDOW_DAYS
    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)

    with SyncSessionLocal() as session:
        rows = (
            session.query(
                Match.match_id,
                Match.match_date,
                Match.season,
                Match.home_goals,
                Match.away_goals,
                Match.home_xg,
                Match.away_xg,
                Match.home_team_id,
                Match.away_team_id,
            )
            .filter(
                Match.league == league,
                Match.status == MatchStatus.FINISHED,
                Match.match_date >= cutoff,
                Match.home_goals.isnot(None),
                Match.away_goals.isnot(None),
            )
            .order_by(Match.match_date.asc())
            .all()
        )

        # Batch load team names
        team_ids = set()
        for r in rows:
            team_ids.add(r.home_team_id)
            team_ids.add(r.away_team_id)
        teams = session.query(Team).filter(Team.team_id.in_(team_ids)).all()
        id_to_name = {t.team_id: t.name for t in teams}

    if not rows:
        logger.warning("No finished matches found in window of %d days", window_days)
        return pd.DataFrame()

    records = []
    for r in rows:
        records.append(
            {
                "match_id": r.match_id,
                "match_date": pd.Timestamp(r.match_date).tz_localize(None)
                if r.match_date.tzinfo is None
                else pd.Timestamp(r.match_date).tz_convert(None),
                "season": r.season,
                "home_team": id_to_name.get(r.home_team_id, str(r.home_team_id)),
                "away_team": id_to_name.get(r.away_team_id, str(r.away_team_id)),
                "home_goals": r.home_goals,
                "away_goals": r.away_goals,
                # Fall back to goals if xG absent
                "home_xg": r.home_xg if r.home_xg is not None else float(r.home_goals),
                "away_xg": r.away_xg if r.away_xg is not None else float(r.away_goals),
            }
        )

    df = pd.DataFrame(records).sort_values("match_date").reset_index(drop=True)
    logger.info("Loaded %d finished matches for feature engineering", len(df))
    return df


def compute_decay_weights(
    df: pd.DataFrame,
    reference_date: Optional[datetime] = None,
    lam: Optional[float] = None,
) -> np.ndarray:
    """
    Compute exponential decay weights for each match row.
    w_i = exp(-λ × days_since_match_i)
    """
    lam = lam or settings.LAMBDA_DECAY
    ref = reference_date or datetime.now()
    if isinstance(ref, datetime) and ref.tzinfo is not None:
        ref = ref.replace(tzinfo=None)

    days_ago = (ref - df["match_date"]).dt.total_seconds() / 86400.0
    weights = np.exp(-lam * days_ago.clip(lower=0).values)
    return weights.astype(np.float64)


def compute_team_rolling_features(
    df: pd.DataFrame,
    lam: Optional[float] = None,
) -> pd.DataFrame:
    """
    For each match row compute team-level EWM attack/defense features.
    This is informational — the actual MLE uses all history simultaneously.

    Returns df with added columns:
        home_ewm_xg_for, home_ewm_xg_against
        away_ewm_xg_for, away_ewm_xg_against
        home_rest_days, away_rest_days
    """
    lam = lam or settings.LAMBDA_DECAY

    # Build per-team history lookups
    # For each team, track all past matches in sorted order
    team_matches: dict[str, list[dict]] = {}
    for _, row in df.iterrows():
        for side, opp_side in [("home", "away"), ("away", "home")]:
            team = row[f"{side}_team"]
            if team not in team_matches:
                team_matches[team] = []
            team_matches[team].append(
                {
                    "date": row["match_date"],
                    "xg_for": row[f"{side}_xg"],
                    "xg_against": row[f"{opp_side}_xg"],
                }
            )

    def _ewm_stats(team: str, cutoff_date, n_lookback: int = 20):
        history = [m for m in team_matches.get(team, []) if m["date"] < cutoff_date]
        if not history:
            return np.nan, np.nan, None
        history = sorted(history, key=lambda x: x["date"])[-n_lookback:]
        last_date = history[-1]["date"]
        days = np.array(
            [(cutoff_date - m["date"]).total_seconds() / 86400 for m in history]
        )
        w = np.exp(-lam * days)
        w /= w.sum()
        xgf = np.dot(w, [m["xg_for"] for m in history])
        xga = np.dot(w, [m["xg_against"] for m in history])
        return xgf, xga, last_date

    rows = []
    for _, row in df.iterrows():
        cd = row["match_date"]
        h_xgf, h_xga, h_last = _ewm_stats(row["home_team"], cd)
        a_xgf, a_xga, a_last = _ewm_stats(row["away_team"], cd)
        h_rest = (cd - h_last).days if h_last else np.nan
        a_rest = (cd - a_last).days if a_last else np.nan
        rows.append(
            {
                "home_ewm_xg_for": h_xgf,
                "home_ewm_xg_against": h_xga,
                "away_ewm_xg_for": a_xgf,
                "away_ewm_xg_against": a_xga,
                "home_rest_days": h_rest,
                "away_rest_days": a_rest,
                "rest_days_diff": (h_rest - a_rest)
                if (not np.isnan(h_rest) and not np.isnan(a_rest))
                else np.nan,
            }
        )

    feature_df = pd.DataFrame(rows, index=df.index)
    return pd.concat([df, feature_df], axis=1)


def get_all_teams(df: pd.DataFrame) -> list[str]:
    """Return sorted unique team list from match dataframe."""
    teams = set(df["home_team"].tolist()) | set(df["away_team"].tolist())
    return sorted(teams)
