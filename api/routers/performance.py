"""
api/routers/performance.py
GET /performance/metrics   — full performance report
GET /performance/montecarlo — Monte Carlo simulation
"""

from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from core.config import settings
from evaluation.metrics import generate_performance_report, monte_carlo_simulation

router = APIRouter()


@router.get("/metrics")
async def get_performance_metrics():
    """
    Full performance report: ROI, CLV, drawdown, Sharpe, bankroll simulation.
    Pulls from DB — returns empty metrics if no bets settled yet.
    """
    try:
        report = generate_performance_report(
            initial_bankroll=settings.BANKROLL_INITIAL
        )
        return report
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "detail": "Could not generate report"},
        )


@router.get("/montecarlo")
async def run_montecarlo(
    avg_edge_pct: float = Query(2.5, description="Average edge in %"),
    avg_odds: float = Query(1.90, description="Average decimal odds"),
    bets_per_season: int = Query(500, ge=10, le=5000),
    n_simulations: int = Query(10000, ge=100, le=100000),
    kelly_fraction: float = Query(0.25, ge=0.05, le=1.0),
):
    """
    Run Monte Carlo bankroll simulation on demand.
    Does not require DB data — uses input parameters directly.
    """
    result = monte_carlo_simulation(
        avg_edge=avg_edge_pct / 100,
        avg_kelly_fraction=kelly_fraction,
        n_simulations=n_simulations,
        bets_per_season=bets_per_season,
        avg_odds=avg_odds,
        kelly_fraction=kelly_fraction,
    )
    return result
