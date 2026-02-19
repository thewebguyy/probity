"""
market/comparison.py
--------------------
Market comparison engine: removes bookmaker margin, computes implied
probability, measures edge, and logs +EV detections.

Margin removal method: Power method (most conservative, recommended).

Power method:
  Given decimal odds o_1, o_2 (and o_draw for H2H):
  Find k such that: sum(1/o_i^k) = 1
  Solve numerically.
  Implied prob: p_i = (1/o_i)^k / sum((1/o_j)^k)

Edge formula:
  Edge = (Model_Prob × Book_Odds) − 1

Trigger:
  Edge > settings.MIN_EDGE_THRESHOLD (default 2.5%)

CLV (Closing Line Value):
  CLV = (Book_Odds_at_detection / Closing_Odds) − 1
  Positive CLV = beat the closing line = long-run edge indicator.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
from scipy.optimize import brentq

from core.config import settings
from core.database import SyncSessionLocal
from core.models import (
    BetSide,
    Edge,
    FairOdds,
    MarketType,
    Match,
    MatchStatus,
    OddsSnapshot,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Margin removal — Power method
# ─────────────────────────────────────────────────────────────────────────────


def remove_margin_power(
    odds: list[float],
) -> list[float]:
    """
    Remove bookmaker margin using the power method.

    Given decimal odds list, returns fair probabilities summing to 1.

    Example:
        odds = [2.10, 3.50, 3.60]  (H2H: home, draw, away)
        Returns [fair_p_home, fair_p_draw, fair_p_away]
    """
    odds = [o for o in odds if o and o > 1.0]
    if not odds:
        return []

    inv_odds = [1.0 / o for o in odds]
    overround = sum(inv_odds)

    if abs(overround - 1.0) < 1e-6:
        return inv_odds

    def _equation(k: float) -> float:
        return sum(p ** k for p in inv_odds) - 1.0

    try:
        k = brentq(_equation, 0.01, 10.0, xtol=1e-8)
    except ValueError:
        # Fallback: simple normalization
        return [p / overround for p in inv_odds]

    raw = [p ** k for p in inv_odds]
    total = sum(raw)
    return [r / total for r in raw]


def implied_prob_power(odds_value: float, all_odds: list[float]) -> float:
    """
    Implied probability for one side using the power method.
    """
    fair_probs = remove_margin_power(all_odds)
    if not fair_probs:
        return 0.0
    idx = all_odds.index(odds_value)
    if idx >= len(fair_probs):
        return 0.0
    return fair_probs[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Edge calculation
# ─────────────────────────────────────────────────────────────────────────────


def compute_edge(model_prob: float, book_odds: float) -> float:
    """
    Edge = (model_prob × book_odds) − 1
    Positive value = expected value positive.
    """
    return (model_prob * book_odds) - 1.0


def kelly_fraction(model_prob: float, book_odds: float, fraction: float = None) -> float:
    """
    Fractional Kelly criterion stake size as proportion of bankroll.
    
    Full Kelly = (model_prob × book_odds - 1) / (book_odds - 1)
    Fractional Kelly = Kelly × fraction
    """
    fraction = fraction or settings.KELLY_FRACTION
    b = book_odds - 1.0  # decimal odds to b-format
    full_kelly = ((model_prob * book_odds) - 1.0) / b if b > 0 else 0.0
    return max(0.0, full_kelly * fraction)


# ─────────────────────────────────────────────────────────────────────────────
# Edge scanner — main comparison loop
# ─────────────────────────────────────────────────────────────────────────────


def _get_model_prob(
    session,
    match_id: int,
    market_type: MarketType,
    line: Optional[float],
    side: BetSide,
) -> Optional[float]:
    """
    Fetch model probability for given market + side from fair_odds table.
    Returns the most recent fair_odds record for this match/market/line.
    """
    q = (
        session.query(FairOdds)
        .filter_by(match_id=match_id, market_type=market_type)
    )
    if line is not None:
        q = q.filter(FairOdds.line == line)
    fo = q.order_by(FairOdds.computed_at.desc()).first()

    if not fo:
        return None

    if side == BetSide.HOME:
        return fo.home_prob
    elif side == BetSide.AWAY:
        return fo.away_prob
    elif side == BetSide.OVER:
        return fo.home_prob   # OU: stored as home_prob=over, away_prob=under
    elif side == BetSide.UNDER:
        return fo.away_prob
    return None


def _check_rapid_line_movement(
    session,
    match_id: int,
    market_type: MarketType,
    line: Optional[float],
    book_odds: float,
    lookback_minutes: int = 10,
) -> bool:
    """
    Returns True if the line has moved adversely in the last N minutes.
    Adverse = odds shortened by >3% since lookback window.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)
    snaps = (
        session.query(OddsSnapshot)
        .filter(
            OddsSnapshot.match_id == match_id,
            OddsSnapshot.market_type == market_type,
            OddsSnapshot.snapshot_at >= cutoff,
        )
        .order_by(OddsSnapshot.snapshot_at.asc())
        .all()
    )
    if len(snaps) < 2:
        return False  # not enough data

    earliest_odds = snaps[0].home_odds or 0
    if earliest_odds > 0 and book_odds > 0:
        movement = (earliest_odds - book_odds) / earliest_odds
        if movement > 0.03:  # odds shortened by >3% = adverse
            logger.info(
                "Rapid adverse line movement detected for match %d — skipping", match_id
            )
            return True
    return False


def scan_for_edges(
    min_edge: Optional[float] = None,
    league: str = "NL1",
) -> list[dict]:
    """
    Main edge scanner.
    
    For each upcoming match → for each recent odds snapshot →
    compare to model's fair odds → log if edge > threshold.

    Returns list of edge dicts (also persisted to DB).
    """
    min_edge = min_edge or settings.MIN_EDGE_THRESHOLD
    detected = []
    now = datetime.now(timezone.utc)

    with SyncSessionLocal() as session:
        # Get upcoming/live matches
        upcoming = (
            session.query(Match)
            .filter(
                Match.league == league,
                Match.status.in_([MatchStatus.SCHEDULED, MatchStatus.LIVE]),
                Match.match_date >= now,
                Match.match_date <= now + timedelta(days=7),
            )
            .all()
        )

        for match in upcoming:
            # Get most recent snapshot per bookmaker/market/line
            snaps = (
                session.query(OddsSnapshot)
                .filter_by(match_id=match.match_id)
                .order_by(OddsSnapshot.snapshot_at.desc())
                .limit(100)
                .all()
            )

            # Group by bookmaker + market + line (take latest)
            seen = {}
            for s in snaps:
                key = (s.bookmaker, s.market_type, s.line)
                if key not in seen:
                    seen[key] = s

            for snap in seen.values():
                _scan_snapshot(
                    session=session,
                    match=match,
                    snap=snap,
                    min_edge=min_edge,
                    detected=detected,
                )

        session.commit()

    logger.info("Edge scan complete — %d edges detected", len(detected))
    return detected


def _scan_snapshot(session, match, snap, min_edge, detected):
    """Process a single odds snapshot for edge opportunities."""

    sides_to_check = []

    if snap.market_type == MarketType.H2H:
        all_odds = [snap.home_odds, snap.draw_odds, snap.away_odds]
        all_odds = [o for o in all_odds if o]
        sides_to_check = [
            (BetSide.HOME, snap.home_odds),
            (BetSide.AWAY, snap.away_odds),
        ]

    elif snap.market_type in (MarketType.AH, MarketType.OU):
        all_odds = [snap.home_odds, snap.away_odds]
        all_odds = [o for o in all_odds if o]
        side_a = BetSide.HOME if snap.market_type == MarketType.AH else BetSide.OVER
        side_b = BetSide.AWAY if snap.market_type == MarketType.AH else BetSide.UNDER
        sides_to_check = [
            (side_a, snap.home_odds),
            (side_b, snap.away_odds),
        ]

    for side, book_odds in sides_to_check:
        if not book_odds or book_odds <= 1.0:
            continue

        model_prob = _get_model_prob(
            session, match.match_id, snap.market_type, snap.line, side
        )
        if model_prob is None or model_prob <= 0:
            continue

        edge = compute_edge(model_prob, book_odds)

        if edge < min_edge:
            continue

        # Guard: no rapid adverse movement
        if _check_rapid_line_movement(session, match.match_id, snap.market_type, snap.line, book_odds):
            continue

        kelly = kelly_fraction(model_prob, book_odds)

        edge_record = Edge(
            match_id=match.match_id,
            bookmaker=snap.bookmaker,
            market_type=snap.market_type,
            line=snap.line,
            side=side,
            model_prob=model_prob,
            book_odds=book_odds,
            edge_value=edge,
            kelly_stake_fraction=kelly,
        )
        session.add(edge_record)

        detected.append(
            {
                "match_id": match.match_id,
                "home_team": match.home_team_id,
                "away_team": match.away_team_id,
                "match_date": str(match.match_date),
                "bookmaker": snap.bookmaker,
                "market": snap.market_type.value,
                "line": snap.line,
                "side": side.value,
                "model_prob": round(model_prob, 4),
                "book_odds": round(book_odds, 3),
                "edge": round(edge * 100, 2),  # in percent
                "kelly_fraction": round(kelly, 4),
                "detected_at": str(datetime.now(timezone.utc)),
            }
        )

        logger.info(
            "[EDGE] %s - %s | %s %s %s | model_prob=%.3f | book_odds=%.2f | edge=%.1f%% | kelly=%.2f%%",
            match.home_team_id,
            match.away_team_id,
            snap.market_type.value,
            snap.line,
            side.value,
            model_prob,
            book_odds,
            edge * 100,
            kelly * 100,
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLV Resolution (post-kickoff)
# ─────────────────────────────────────────────────────────────────────────────


def resolve_clv(hours_before_kickoff: int = 1) -> int:
    """
    For finished edges without CLV filled in, find the closing odds
    (snapshot closest to kickoff) and compute CLV.

    Returns number of edges resolved.
    """
    resolved = 0
    with SyncSessionLocal() as session:
        unresolved = (
            session.query(Edge)
            .filter(Edge.closing_odds.is_(None))
            .join(Match)
            .filter(Match.match_date <= datetime.now(timezone.utc))
            .all()
        )

        for edge in unresolved:
            cutoff = edge.match.match_date - timedelta(hours=hours_before_kickoff)

            closing_snap = (
                session.query(OddsSnapshot)
                .filter(
                    OddsSnapshot.match_id == edge.match_id,
                    OddsSnapshot.market_type == edge.market_type,
                    OddsSnapshot.snapshot_at <= cutoff,
                )
                .order_by(OddsSnapshot.snapshot_at.desc())
                .first()
            )

            if not closing_snap:
                continue

            closing_odds = (
                closing_snap.home_odds
                if edge.side in (BetSide.HOME, BetSide.OVER)
                else closing_snap.away_odds
            )

            if closing_odds and closing_odds > 0:
                edge.closing_odds = closing_odds
                edge.clv = (edge.book_odds / closing_odds) - 1.0
                edge.resolved_at = datetime.now(timezone.utc)
                resolved += 1

        session.commit()

    logger.info("Resolved CLV for %d edges", resolved)
    return resolved
