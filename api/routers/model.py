"""
api/routers/model.py
GET /model/fair-odds/{match_id}
GET /model/parameters
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.models import FairOdds, Match, ModelRun, Team

router = APIRouter()


@router.get("/fair-odds/{match_id}")
async def get_fair_odds(match_id: int, db: AsyncSession = Depends(get_db)):
    """
    All model-computed fair probabilities and fair odds for a match,
    grouped by market type and line.
    """
    # Verify match exists
    match_q = await db.execute(
        select(Match).where(Match.match_id == match_id)
    )
    match = match_q.scalars().first()
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

    # Get fair odds
    fo_q = await db.execute(
        select(FairOdds)
        .where(FairOdds.match_id == match_id)
        .order_by(FairOdds.market_type, FairOdds.line, FairOdds.computed_at.desc())
    )
    records = fo_q.scalars().all()

    if not records:
        raise HTTPException(
            status_code=404,
            detail="No fair odds computed for this match. Run model first.",
        )

    # Group by market_type + line (take latest per market/line)
    seen = {}
    for r in records:
        key = (r.market_type.value, r.line)
        if key not in seen:
            seen[key] = r

    h2h = None
    ah_markets = []
    ou_markets = []

    for (mtype, line), r in seen.items():
        entry = {
            "line": line,
            "home_prob": round(r.home_prob, 4),
            "away_prob": round(r.away_prob, 4) if r.away_prob else None,
            "home_fair_odds": round(r.home_fair_odds, 3) if r.home_fair_odds else None,
            "away_fair_odds": round(r.away_fair_odds, 3) if r.away_fair_odds else None,
            "computed_at": r.computed_at.isoformat() if r.computed_at else None,
        }
        if mtype == "H2H":
            h2h = {
                **entry,
                "draw_prob": round(r.draw_prob, 4) if r.draw_prob else None,
                "draw_fair_odds": round(r.draw_fair_odds, 3) if r.draw_fair_odds else None,
            }
        elif mtype == "AH":
            ah_markets.append(entry)
        elif mtype == "OU":
            ou_markets.append({
                **entry,
                "over_prob": round(r.home_prob, 4),
                "under_prob": round(r.away_prob, 4) if r.away_prob else None,
                "over_fair_odds": round(r.home_fair_odds, 3) if r.home_fair_odds else None,
                "under_fair_odds": round(r.away_fair_odds, 3) if r.away_fair_odds else None,
            })

    return {
        "match_id": match_id,
        "model_version": records[0].model_version if records else None,
        "lambda_home": round(records[0].lambda_home, 4) if records and records[0].lambda_home else None,
        "lambda_away": round(records[0].lambda_away, 4) if records and records[0].lambda_away else None,
        "h2h": h2h,
        "asian_handicaps": sorted(ah_markets, key=lambda x: x["line"] or 0),
        "over_unders": sorted(ou_markets, key=lambda x: x["line"] or 0),
    }


@router.get("/parameters")
async def get_model_parameters(league: str = "NL1", db: AsyncSession = Depends(get_db)):
    """Latest Dixon-Coles model parameters (attack/defense per team)."""
    run_q = await db.execute(
        select(ModelRun)
        .where(ModelRun.league == league)
        .order_by(ModelRun.fitted_at.desc())
        .limit(1)
    )
    run = run_q.scalars().first()
    if not run:
        raise HTTPException(status_code=404, detail="No model run found. Run model fit first.")

    return {
        "run_id": run.run_id,
        "league": run.league,
        "home_advantage": round(run.home_advantage, 4),
        "rho": round(run.rho, 4),
        "log_likelihood": round(run.log_likelihood, 2),
        "n_matches_used": run.n_matches,
        "window_days": run.window_days,
        "fitted_at": run.fitted_at.isoformat() if run.fitted_at else None,
    }
