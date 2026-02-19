"""
api/routers/edges.py
GET /edges/live     — active +EV opportunities
GET /edges/history  — all historical edge detections
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from core.database import get_db
from core.models import Edge, Match, Team

router = APIRouter()


@router.get("/live")
async def get_live_edges(
    min_edge_pct: float = Query(2.5, description="Minimum edge % to show"),
    db: AsyncSession = Depends(get_db),
):
    """
    Active +EV opportunities detected in the last 24 hours
    for upcoming matches.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)

    result = await db.execute(
        select(Edge)
        .join(Match)
        .options(joinedload(Edge.match))
        .where(
            Edge.detected_at >= cutoff,
            Edge.edge_value >= (min_edge_pct / 100),
            Match.match_date >= datetime.now(timezone.utc),
        )
        .order_by(Edge.edge_value.desc())
        .limit(50)
    )
    edges = result.scalars().unique().all()

    return [
        {
            "edge_id": e.edge_id,
            "match_id": e.match_id,
            "match_date": e.match.match_date.isoformat() if e.match else None,
            "bookmaker": e.bookmaker,
            "market": e.market_type.value,
            "line": e.line,
            "side": e.side.value,
            "model_prob": round(e.model_prob, 4),
            "book_odds": round(e.book_odds, 3),
            "edge_pct": round(e.edge_value * 100, 2),
            "kelly_fraction_pct": round((e.kelly_stake_fraction or 0) * 100, 2),
            "detected_at": e.detected_at.isoformat() if e.detected_at else None,
            "clv": round(e.clv * 100, 2) if e.clv is not None else None,
        }
        for e in edges
    ]


@router.get("/history")
async def get_edge_history(
    days_back: int = Query(30, ge=1, le=365),
    clv_only: bool = Query(False, description="Only show edges with CLV resolved"),
    db: AsyncSession = Depends(get_db),
):
    """Historical edge detection log."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)

    q = (
        select(Edge)
        .join(Match)
        .options(joinedload(Edge.match))
        .where(Edge.detected_at >= cutoff)
    )
    if clv_only:
        q = q.where(Edge.closing_odds.isnot(None))

    result = await db.execute(q.order_by(Edge.detected_at.desc()).limit(200))
    edges = result.scalars().unique().all()

    return [
        {
            "edge_id": e.edge_id,
            "match_id": e.match_id,
            "match_date": e.match.match_date.isoformat() if e.match else None,
            "bookmaker": e.bookmaker,
            "market": e.market_type.value,
            "line": e.line,
            "side": e.side.value,
            "model_prob": round(e.model_prob, 4),
            "book_odds": round(e.book_odds, 3),
            "edge_pct": round(e.edge_value * 100, 2),
            "closing_odds": e.closing_odds,
            "clv_pct": round(e.clv * 100, 2) if e.clv is not None else None,
            "detected_at": e.detected_at.isoformat() if e.detected_at else None,
        }
        for e in edges
    ]
