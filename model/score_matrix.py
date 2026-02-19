"""
model/score_matrix.py
---------------------
Builds the full score probability matrix from Dixon-Coles parameters
and derives market probabilities:

  - Asian Handicap (AH)
  - Over/Under (OU)
  - 1X2 (H2H)

Then converts to fair odds and stores in fair_odds table.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from scipy.stats import poisson

from core.config import settings
from model.dixon_coles import DCParams, _tau

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Score matrix
# ─────────────────────────────────────────────────────────────────────────────


def build_score_matrix(
    lam_h: float,
    lam_a: float,
    rho: float,
    max_goals: int = None,
) -> np.ndarray:
    """
    Build a (max_goals+1) × (max_goals+1) matrix where
    M[i][j] = P(home scores i, away scores j).

    Applies Dixon-Coles τ correction to (0,0), (1,0), (0,1), (1,1).
    """
    max_goals = max_goals or settings.MAX_GOALS
    g = max_goals + 1

    home_pmf = poisson.pmf(np.arange(g), lam_h)
    away_pmf = poisson.pmf(np.arange(g), lam_a)

    matrix = np.outer(home_pmf, away_pmf)

    # Apply DC corrections to low-score cells
    for i in range(min(2, g)):
        for j in range(min(2, g)):
            matrix[i, j] *= _tau(i, j, lam_h, lam_a, rho)

    # Renormalize (tau correction may shift probability mass slightly)
    total = matrix.sum()
    if total > 0:
        matrix /= total

    return matrix


# ─────────────────────────────────────────────────────────────────────────────
# Market probability derivations
# ─────────────────────────────────────────────────────────────────────────────


def h2h_probs(matrix: np.ndarray) -> tuple[float, float, float]:
    """
    Returns (P_home_win, P_draw, P_away_win) from score matrix.
    """
    g = matrix.shape[0]
    p_home = float(np.sum(np.tril(matrix, -1)))   # home_goals > away_goals
    p_draw = float(np.trace(matrix))
    p_away = float(np.sum(np.triu(matrix, 1)))    # away_goals > home_goals

    # Sanity normalize
    total = p_home + p_draw + p_away
    return p_home / total, p_draw / total, p_away / total


def asian_handicap_probs(
    matrix: np.ndarray,
    line: float,
) -> tuple[float, float]:
    """
    Compute Asian Handicap probabilities for a given line.
    line is applied to home team (positive = home gives goals).

    e.g. line = -0.5  → home team -0.5 (needs to win by 1+ to cover)
         line = 0.0   → pick'em (draw = push/half refund)

    Returns (p_home_cover, p_away_cover).
    For quarter-ball lines we split into two half lines.
    """
    # Handle quarter-ball split
    remainder = (line * 2) % 1
    if remainder != 0:
        # Quarter ball: split 50/50 between floor and ceil half-ball lines
        line_lo = int(line * 2) / 2.0
        line_hi = line_lo + 0.5
        ph_lo, pa_lo = _ah_half_ball(matrix, line_lo)
        ph_hi, pa_hi = _ah_half_ball(matrix, line_hi)
        return (ph_lo + ph_hi) / 2, (pa_lo + pa_hi) / 2
    else:
        return _ah_half_ball(matrix, line)


def _ah_half_ball(matrix: np.ndarray, line: float) -> tuple[float, float]:
    """
    Asian Handicap calculation for a pure half-ball or whole-number line.
    line = handicap on home team (negative means home favoured).
    """
    g = matrix.shape[0]
    p_home = 0.0
    p_away = 0.0

    for i in range(g):
        for j in range(g):
            margin = (i - j) + line   # adjusted home margin
            prob = matrix[i, j]
            if margin > 0:
                p_home += prob
            elif margin < 0:
                p_away += prob
            # margin == 0 → push (void), not counted

    total = p_home + p_away
    if total == 0:
        return 0.5, 0.5
    return p_home / total, p_away / total


def over_under_probs(
    matrix: np.ndarray,
    total: float,
) -> tuple[float, float]:
    """
    Over/Under probability for a given total goals line.
    Returns (p_over, p_under). Exact = push (split half lines).
    """
    g = matrix.shape[0]
    p_over = 0.0
    p_under = 0.0

    for i in range(g):
        for j in range(g):
            goals = i + j
            prob = matrix[i, j]
            if goals > total:
                p_over += prob
            elif goals < total:
                p_under += prob
            # goals == total → push for whole numbers

    # For half-ball totals, push never occurs
    t = p_over + p_under
    if t == 0:
        return 0.5, 0.5
    return p_over / t, p_under / t


# ─────────────────────────────────────────────────────────────────────────────
# Fair odds computation + DB storage
# ─────────────────────────────────────────────────────────────────────────────


def prob_to_fair_odds(p: float) -> Optional[float]:
    if p <= 0:
        return None
    return round(1.0 / p, 4)


def compute_and_store_fair_odds(
    match_id: int,
    home_team: str,
    away_team: str,
    params: DCParams,
    ah_lines: Optional[list[float]] = None,
    ou_totals: Optional[list[float]] = None,
) -> dict:
    """
    Compute fair odds for a match and store in DB.
    Returns dict of computed values.
    """
    from core.database import SyncSessionLocal
    from core.models import FairOdds, MarketType

    lam_h = params.lambda_home(home_team, away_team)
    lam_a = params.lambda_away(home_team, away_team)

    if not np.isfinite(lam_h) or not np.isfinite(lam_a):
        logger.error("Invalid lambda for %s vs %s: %s, %s", home_team, away_team, lam_h, lam_a)
        return {}

    matrix = build_score_matrix(lam_h, lam_a, params.rho)

    ah_lines = ah_lines or [-1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
    ou_totals = ou_totals or [1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 4.5]

    p_home, p_draw, p_away = h2h_probs(matrix)
    result = {
        "lambda_home": lam_h,
        "lambda_away": lam_a,
        "h2h": {
            "home_prob": p_home,
            "draw_prob": p_draw,
            "away_prob": p_away,
            "home_fair_odds": prob_to_fair_odds(p_home),
            "draw_fair_odds": prob_to_fair_odds(p_draw),
            "away_fair_odds": prob_to_fair_odds(p_away),
        },
        "ah": {},
        "ou": {},
    }

    records = []

    # H2H
    records.append(
        FairOdds(
            match_id=match_id,
            market_type=MarketType.H2H,
            line=None,
            home_prob=p_home,
            draw_prob=p_draw,
            away_prob=p_away,
            home_fair_odds=prob_to_fair_odds(p_home),
            draw_fair_odds=prob_to_fair_odds(p_draw),
            away_fair_odds=prob_to_fair_odds(p_away),
            lambda_home=lam_h,
            lambda_away=lam_a,
            model_version=params.run_id,
        )
    )

    # AH
    for line in ah_lines:
        ph, pa = asian_handicap_probs(matrix, line)
        result["ah"][line] = {
            "home_prob": ph,
            "away_prob": pa,
            "home_fair_odds": prob_to_fair_odds(ph),
            "away_fair_odds": prob_to_fair_odds(pa),
        }
        records.append(
            FairOdds(
                match_id=match_id,
                market_type=MarketType.AH,
                line=line,
                home_prob=ph,
                away_prob=pa,
                home_fair_odds=prob_to_fair_odds(ph),
                away_fair_odds=prob_to_fair_odds(pa),
                lambda_home=lam_h,
                lambda_away=lam_a,
                model_version=params.run_id,
            )
        )

    # OU
    for total in ou_totals:
        po, pu = over_under_probs(matrix, total)
        result["ou"][total] = {
            "over_prob": po,
            "under_prob": pu,
            "over_fair_odds": prob_to_fair_odds(po),
            "under_fair_odds": prob_to_fair_odds(pu),
        }
        records.append(
            FairOdds(
                match_id=match_id,
                market_type=MarketType.OU,
                line=total,
                home_prob=po,
                away_prob=pu,
                home_fair_odds=prob_to_fair_odds(po),
                away_fair_odds=prob_to_fair_odds(pu),
                lambda_home=lam_h,
                lambda_away=lam_a,
                model_version=params.run_id,
            )
        )

    with SyncSessionLocal() as session:
        session.add_all(records)
        session.commit()

    logger.info(
        "Stored %d fair odds records for match_id=%d (λ_h=%.3f, λ_a=%.3f)",
        len(records), match_id, lam_h, lam_a,
    )
    return result
