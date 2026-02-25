"""
api/routers/performance.py
GET /performance/metrics   — full performance report (cached 60s, run in executor)
GET /performance/montecarlo — Monte Carlo simulation
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from core.config import settings
from evaluation.metrics import generate_performance_report, monte_carlo_simulation

logger = logging.getLogger(__name__)
router = APIRouter()

# Cache for performance report: avoid blocking event loop and recomputing on every request
_metrics_cache: dict | None = None
_metrics_cache_ts: float = 0
METRICS_CACHE_TTL_SECONDS = 60
_executor = ThreadPoolExecutor(max_workers=2)


def _generate_report_sync() -> dict:
    """Sync DB-heavy call to run in thread pool."""
    return generate_performance_report(initial_bankroll=settings.BANKROLL_INITIAL)


def _empty_report() -> dict:
    return {
        "roi_pct": 0.0,
        "average_edge_pct": 0.0,
        "clv": {"avg_clv": 0.0, "beat_rate": 0.0, "n_resolved": 0},
        "win_rate": {"actual_win_rate": 0.0, "n_bets": 0},
        "max_drawdown_pct": 0.0,
        "sharpe_ratio": 0.0,
        "bankroll_simulation": {"final_bankroll": settings.BANKROLL_INITIAL, "path": []},
        "monte_carlo": None,
        "n_settled_bets": 0,
        "n_resolved_edges": 0,
    }


@router.get("/metrics")
async def get_performance_metrics():
    """
    Full performance report: ROI, CLV, drawdown, Sharpe, bankroll simulation.
    Cached 60s; DB work runs in thread pool so the event loop is not blocked.
    """
    global _metrics_cache, _metrics_cache_ts
    now_ts = time.monotonic()
    if _metrics_cache is not None and (now_ts - _metrics_cache_ts) < METRICS_CACHE_TTL_SECONDS:
        return _metrics_cache
    try:
        loop = asyncio.get_event_loop()
        report = await loop.run_in_executor(_executor, _generate_report_sync)
        _metrics_cache = report
        _metrics_cache_ts = now_ts
        return report
    except Exception as e:
        logger.exception("Metrics generation error: %s", e)
        return _empty_report()


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
    Runs in executor to avoid blocking the event loop.
    """
    def _run() -> dict:
        return monte_carlo_simulation(
            avg_edge=avg_edge_pct / 100,
            avg_kelly_fraction=kelly_fraction,
            n_simulations=n_simulations,
            bets_per_season=bets_per_season,
            avg_odds=avg_odds,
            kelly_fraction=kelly_fraction,
        )
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _run)
