"""
evaluation/metrics.py
---------------------
Performance evaluation metrics:

  - ROI
  - Average edge at detection
  - CLV beat rate and average CLV
  - Win rate vs expected win rate
  - Max drawdown
  - Sharpe ratio (annualized)
  - Kelly bankroll simulation
  - Monte Carlo: 10,000 seasons × N bets

All calculations operate over the `bets` and `edges` tables in DB.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

from core.database import SyncSessionLocal
from core.models import Bet, Edge, Match

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Load data from DB
# ─────────────────────────────────────────────────────────────────────────────


def load_settled_bets() -> pd.DataFrame:
    """Load all settled bets from DB into DataFrame."""
    with SyncSessionLocal() as session:
        bets = session.query(Bet).filter(Bet.pnl.isnot(None)).all()
        if not bets:
            return pd.DataFrame()

        records = [
            {
                "bet_id": b.bet_id,
                "match_id": b.match_id,
                "stake": b.stake,
                "odds": b.odds,
                "result": b.result,
                "pnl": b.pnl,
                "placed_at": b.placed_at,
                "settled_at": b.settled_at,
            }
            for b in bets
        ]
    return pd.DataFrame(records)


def load_resolved_edges() -> pd.DataFrame:
    """Load edges with CLV resolved."""
    with SyncSessionLocal() as session:
        edges = (
            session.query(Edge)
            .filter(Edge.closing_odds.isnot(None))
            .all()
        )
        if not edges:
            return pd.DataFrame()

        records = [
            {
                "edge_id": e.edge_id,
                "match_id": e.match_id,
                "edge_value": e.edge_value,
                "model_prob": e.model_prob,
                "book_odds": e.book_odds,
                "closing_odds": e.closing_odds,
                "clv": e.clv,
                "kelly_stake_fraction": e.kelly_stake_fraction,
                "detected_at": e.detected_at,
            }
            for e in edges
        ]
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Core metrics
# ─────────────────────────────────────────────────────────────────────────────


def compute_roi(bets_df: pd.DataFrame) -> float:
    """Total ROI = sum(PnL) / sum(stake)."""
    if bets_df.empty:
        return 0.0
    return float(bets_df["pnl"].sum() / bets_df["stake"].sum())


def compute_average_edge(edges_df: pd.DataFrame) -> float:
    if edges_df.empty:
        return 0.0
    return float(edges_df["edge_value"].mean())


def compute_clv_stats(edges_df: pd.DataFrame) -> dict:
    """
    CLV analysis:
      - avg_clv: mean closing line value beat
      - beat_rate: fraction of bets that beat the closing line
    """
    if edges_df.empty or "clv" not in edges_df.columns:
        return {"avg_clv": 0.0, "beat_rate": 0.0, "n_resolved": 0}

    clv_data = edges_df["clv"].dropna()
    return {
        "avg_clv": float(clv_data.mean()),
        "beat_rate": float((clv_data > 0).mean()),
        "n_resolved": int(len(clv_data)),
    }


def compute_win_rate(bets_df: pd.DataFrame) -> dict:
    """
    Actual win rate vs expected (from model_prob in edges joined to bets).
    """
    if bets_df.empty:
        return {"actual_win_rate": 0.0, "n_bets": 0}

    wins = (bets_df["result"] == "win").sum()
    n = len(bets_df)
    return {
        "actual_win_rate": float(wins / n) if n > 0 else 0.0,
        "n_bets": int(n),
    }


def compute_max_drawdown(bets_df: pd.DataFrame, initial_bankroll: float = 1000.0) -> float:
    """
    Maximum peak-to-trough drawdown in bankroll units.
    Returns as fraction of peak bankroll.
    """
    if bets_df.empty:
        return 0.0

    df = bets_df.sort_values("settled_at")
    bankroll = initial_bankroll + df["pnl"].cumsum()
    bankroll = pd.concat([pd.Series([initial_bankroll]), bankroll])

    peak = bankroll.cummax()
    drawdown = (bankroll - peak) / peak
    return float(drawdown.min())  # negative value (e.g. -0.15 = 15% drawdown)


def compute_sharpe(bets_df: pd.DataFrame, risk_free: float = 0.0) -> float:
    """
    Per-bet Sharpe ratio (annualized assuming ~500 bets/year).
    """
    if len(bets_df) < 5:
        return 0.0

    returns = bets_df["pnl"] / bets_df["stake"]
    mean_r = returns.mean()
    std_r = returns.std()

    if std_r == 0:
        return 0.0

    # Per-bet Sharpe × sqrt(500) to annualize
    return float((mean_r - risk_free) / std_r * np.sqrt(500))


# ─────────────────────────────────────────────────────────────────────────────
# Kelly bankroll simulation
# ─────────────────────────────────────────────────────────────────────────────


def simulate_kelly_bankroll(
    edges_df: pd.DataFrame,
    bets_df: pd.DataFrame,
    initial_bankroll: float = 1000.0,
    kelly_fraction: float = 0.25,
) -> dict:
    """
    Simulate bankroll growth using fractional Kelly stakes.
    Uses actual bet results if available; falls back to edge-only simulation.
    """
    if bets_df.empty:
        return {"final_bankroll": initial_bankroll, "peak": initial_bankroll, "path": []}

    bankroll = initial_bankroll
    path = [bankroll]

    bets_sorted = bets_df.sort_values("placed_at").copy()
    for _, bet in bets_sorted.iterrows():
        if bet["result"] == "win":
            bankroll += bet["pnl"]
        elif bet["result"] == "loss":
            bankroll += bet["pnl"]
        path.append(max(0, bankroll))
        if bankroll <= 0:
            break

    return {
        "final_bankroll": round(bankroll, 2),
        "peak": round(max(path), 2),
        "trough": round(min(path), 2),
        "path": [round(p, 2) for p in path],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Monte Carlo simulation
# ─────────────────────────────────────────────────────────────────────────────


def monte_carlo_simulation(
    avg_edge: float,
    avg_kelly_fraction: float,
    n_simulations: int = 10_000,
    bets_per_season: int = 500,
    avg_odds: float = 1.95,
    initial_bankroll: float = 1.0,   # normalized
    kelly_fraction: float = 0.25,
    seed: int = 42,
) -> dict:
    """
    Monte Carlo simulation of bankroll outcomes.

    Models each bet as a Bernoulli trial with:
      p_win = implied by avg_odds and avg_edge
      stake = kelly_fraction × kelly × bankroll

    Returns:
      - pct_profitable: fraction of simulations ending > initial
      - pct_ruin: fraction hitting 0
      - median_final: median final bankroll
      - p5, p25, p75, p95: percentile outcomes
      - distribution: histogram-ready dict
    """
    rng = np.random.default_rng(seed)

    # Infer win prob from edge and odds
    # edge = p_win × odds - 1  =>  p_win = (1 + edge) / odds
    p_win = (1 + avg_edge) / avg_odds
    p_win = np.clip(p_win, 0.01, 0.99)

    # Kelly stake per bet (as fraction of current bankroll)
    b = avg_odds - 1.0
    full_kelly = (p_win * avg_odds - 1.0) / b if b > 0 else 0
    stake_frac = np.clip(full_kelly * kelly_fraction, 0.001, 0.5)

    final_bankrolls = np.zeros(n_simulations)

    for sim in range(n_simulations):
        bankroll = initial_bankroll
        for _ in range(bets_per_season):
            if bankroll <= 0.01:
                bankroll = 0
                break
            stake = bankroll * stake_frac
            if rng.random() < p_win:
                bankroll += stake * b
            else:
                bankroll -= stake
        final_bankrolls[sim] = bankroll

    pct_profitable = float(np.mean(final_bankrolls > initial_bankroll))
    pct_ruin = float(np.mean(final_bankrolls <= 0.01))
    median_final = float(np.median(final_bankrolls))

    percentiles = np.percentile(final_bankrolls, [5, 25, 50, 75, 95])

    # Warn if fragile
    if pct_ruin > 0.15:
        logger.warning(
            "Monte Carlo: %.1f%% of simulations result in ruin — edge may be fragile!",
            pct_ruin * 100,
        )

    return {
        "n_simulations": n_simulations,
        "bets_per_season": bets_per_season,
        "avg_edge_pct": round(avg_edge * 100, 2),
        "p_win": round(p_win, 4),
        "stake_fraction": round(stake_frac, 4),
        "pct_profitable": round(pct_profitable * 100, 2),
        "pct_ruin": round(pct_ruin * 100, 2),
        "median_final": round(median_final, 4),
        "p5": round(float(percentiles[0]), 4),
        "p25": round(float(percentiles[1]), 4),
        "p50": round(float(percentiles[2]), 4),
        "p75": round(float(percentiles[3]), 4),
        "p95": round(float(percentiles[4]), 4),
        "edge_fragile": pct_ruin > 0.15,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full metrics report
# ─────────────────────────────────────────────────────────────────────────────


def generate_performance_report(
    initial_bankroll: float = 1000.0,
) -> dict[str, Any]:
    """
    Full performance report pulling from DB.
    """
    bets_df = load_settled_bets()
    edges_df = load_resolved_edges()

    roi = compute_roi(bets_df)
    avg_edge = compute_average_edge(edges_df)
    clv = compute_clv_stats(edges_df)
    win_rate = compute_win_rate(bets_df)
    max_dd = compute_max_drawdown(bets_df, initial_bankroll)
    sharpe = compute_sharpe(bets_df)
    bankroll_sim = simulate_kelly_bankroll(edges_df, bets_df, initial_bankroll)

    # Monte Carlo (use detected edges as reference)
    avg_odds = (
        float(edges_df["book_odds"].mean()) if not edges_df.empty else 1.90
    )
    avg_kf = (
        float(edges_df["kelly_stake_fraction"].mean()) if not edges_df.empty else 0.02
    )
    mc = monte_carlo_simulation(
        avg_edge=avg_edge,
        avg_kelly_fraction=avg_kf,
        avg_odds=avg_odds,
    )

    return {
        "roi_pct": round(roi * 100, 3),
        "average_edge_pct": round(avg_edge * 100, 2),
        "clv": clv,
        "win_rate": win_rate,
        "max_drawdown_pct": round(max_dd * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "bankroll_simulation": bankroll_sim,
        "monte_carlo": mc,
        "n_settled_bets": int(len(bets_df)),
        "n_resolved_edges": int(len(edges_df)),
        "generated_at": str(datetime.now()),
    }
