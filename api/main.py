"""
api/main.py
-----------
FastAPI application exposing the probability pricing engine's outputs:

  GET /matches/upcoming           — upcoming fixtures
  GET /model/fair-odds/{match_id} — fair probabilities + fair odds
  GET /edges/live                 — active +EV opportunities
  GET /performance/metrics        — ROI, CLV, drawdown stats
  GET /simulation/montecarlo      — Monte Carlo bankroll simulation

Also serves the HTML dashboard at GET /dashboard.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from api.routers import edges, matches, model, performance
from api.routers import dashboard as dashboard_router
from core.config import settings
from core.database import async_engine, get_db

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Probity API starting up...")
    # Verify DB connectivity
    try:
        async with async_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("Database connection OK")
    except Exception as e:
        logger.error("Database connection failed: %s", e)
    yield
    logger.info("Probity API shutting down...")
    await async_engine.dispose()


app = FastAPI(
    title="Probity — Probability Pricing Engine",
    description=(
        "Estimates true outcome probabilities for Eredivisie matches, "
        "converts them to fair prices, and detects bookmaker mispricing."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — adjust origins for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Static files (dashboard assets)
_static_dir = Path(__file__).parent.parent / "dashboard" / "static"
_static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

# ─────────────────────────────────────────────────────────────────────────────
# Routers
# ─────────────────────────────────────────────────────────────────────────────

app.include_router(matches.router, prefix="/matches", tags=["Matches"])
app.include_router(model.router, prefix="/model", tags=["Model"])
app.include_router(edges.router, prefix="/edges", tags=["Edges"])
app.include_router(performance.router, prefix="/performance", tags=["Performance"])
app.include_router(dashboard_router.router, tags=["Dashboard"])


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/health", tags=["System"])
async def health(db: AsyncSession = Depends(get_db)):
    try:
        await db.execute(text("SELECT 1"))
        return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "error", "detail": str(e)})


@app.get("/", tags=["System"])
async def root():
    return {
        "name": "Probity",
        "description": "Probability Pricing Engine & Market Mispricing Detector",
        "version": "1.0.0",
        "docs": "/docs",
        "dashboard": "/dashboard",
    }
