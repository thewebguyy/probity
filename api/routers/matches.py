"""
api/routers/matches.py
GET /matches/upcoming
GET /matches/{match_id}
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from core.database import get_db
from core.models import Match, MatchStatus, Team

router = APIRouter()


@router.get("/upcoming")
async def get_upcoming_matches(
    days_ahead: int = Query(7, ge=1, le=30, description="Number of days to look ahead"),
    league: str = Query("NL1"),
    db: AsyncSession = Depends(get_db),
):
    """Upcoming fixtures in the next N days."""
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(days=days_ahead)

    result = await db.execute(
        select(Match)
        .options(joinedload(Match.home_team), joinedload(Match.away_team))
        .where(
            Match.league == league,
            Match.status == MatchStatus.SCHEDULED,
            Match.match_date >= now,
            Match.match_date <= cutoff,
        )
        .order_by(Match.match_date.asc())
    )
    matches = result.scalars().unique().all()

    return [
        {
            "match_id": m.match_id,
            "home_team": m.home_team.name if m.home_team else m.home_team_id,
            "away_team": m.away_team.name if m.away_team else m.away_team_id,
            "match_date": m.match_date.isoformat(),
            "league": m.league,
            "status": m.status.value,
        }
        for m in matches
    ]


@router.get("/{match_id}")
async def get_match(match_id: int, db: AsyncSession = Depends(get_db)):
    """Detail for a single match."""
    result = await db.execute(
        select(Match)
        .options(joinedload(Match.home_team), joinedload(Match.away_team))
        .where(Match.match_id == match_id)
    )
    match = result.scalars().first()
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

    return {
        "match_id": match.match_id,
        "home_team": match.home_team.name if match.home_team else match.home_team_id,
        "away_team": match.away_team.name if match.away_team else match.away_team_id,
        "match_date": match.match_date.isoformat(),
        "league": match.league,
        "season": match.season,
        "status": match.status.value,
        "home_goals": match.home_goals,
        "away_goals": match.away_goals,
        "home_xg": match.home_xg,
        "away_xg": match.away_xg,
    }
