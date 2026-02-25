"""
ingestion/odds_scraper.py
-------------------------
Polls The Odds API (https://the-odds-api.com) for live bookmaker lines
on Eredivisie matches and appends them as OddsSnapshot records.

The Odds API free tier: ~500 requests/month.
Polling every 90 sec × 9 matches/week ≈ manageable.

Supported markets:
  - h2h     → H2H (1X2)
  - asian_handicaps → AH
  - totals   → OU

We resolve matches to our DB via team name matching.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import requests

from core.config import settings
from core.database import SyncSessionLocal
from core.models import Match, MarketType, OddsSnapshot, Team

logger = logging.getLogger(__name__)

ODDS_API_SPORT = "soccer_netherlands_eredivisie"

# Map Odds API market keys to our MarketType enum
MARKET_MAP = {
    "h2h": MarketType.H2H,
    "totals": MarketType.OU,
    "asian_handicaps": MarketType.AH,
}


def _resolve_match_id(session, home_name: str, away_name: str) -> Optional[int]:
    """
    Try to find an existing Match by team names. Returns match_id (scalar) or None.
    Do not pass ORM objects outside session scope.
    """
    home_team = (
        session.query(Team)
        .filter(Team.name.ilike(f"%{home_name.split()[0]}%"))
        .first()
    )
    away_team = (
        session.query(Team)
        .filter(Team.name.ilike(f"%{away_name.split()[0]}%"))
        .first()
    )
    if not (home_team and away_team):
        return None
    home_id = int(home_team.team_id)
    away_id = int(away_team.team_id)

    match = (
        session.query(Match.match_id)
        .filter_by(home_team_id=home_id, away_team_id=away_id)
        .order_by(Match.match_date.desc())
        .first()
    )
    return int(match[0]) if match else None


def fetch_odds() -> list[dict]:
    """
    Fetch live odds from The Odds API for Eredivisie.
    Returns raw list of event dicts.
    """
    if not settings.ODDS_API_KEY:
        logger.warning("ODDS_API_KEY not set — skipping live odds fetch")
        return []

    url = f"{settings.ODDS_API_BASE_URL}/sports/{ODDS_API_SPORT}/odds"
    params = {
        "apiKey": settings.ODDS_API_KEY,
        "regions": "eu",
        "markets": "h2h,totals,asian_handicaps",
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        remaining = resp.headers.get("x-requests-remaining", "?")
        logger.info("Odds API call OK — requests remaining: %s", remaining)
        return resp.json()
    except Exception as exc:
        logger.error("Odds API error: %s", exc)
        return []


def _snapshot_from_outcome(outcomes: list[dict], side: str) -> Optional[float]:
    """Extract odds for a specific side from Odds API outcome list."""
    for o in outcomes:
        if o.get("name", "").lower() == side.lower():
            return o.get("price")
    return None


def process_and_store(events: list[dict]) -> int:
    """
    Parse Odds API events and insert OddsSnapshot rows.
    Returns count of snapshots inserted.
    """
    count = 0
    now = datetime.now(timezone.utc)

    with SyncSessionLocal() as session:
        for event in events:
            home_name = event.get("home_team", "")
            away_name = event.get("away_team", "")
            commence_time = event.get("commence_time")

            match_id = _resolve_match_id(session, home_name, away_name)
            if not match_id:
                logger.debug("No DB match for %s vs %s — skipping", home_name, away_name)
                continue

            bookmakers = event.get("bookmakers", [])
            for bm in bookmakers:
                bm_key = bm.get("key", "")
                for market in bm.get("markets", []):
                    market_key = market.get("key")
                    mtype = MARKET_MAP.get(market_key)
                    if not mtype:
                        continue

                    outcomes = market.get("outcomes", [])

                    if mtype == MarketType.H2H:
                        snapshot = OddsSnapshot(
                            match_id=match_id,
                            bookmaker=bm_key,
                            market_type=mtype,
                            line=None,
                            home_odds=_snapshot_from_outcome(outcomes, home_name),
                            away_odds=_snapshot_from_outcome(outcomes, away_name),
                            draw_odds=_snapshot_from_outcome(outcomes, "Draw"),
                            snapshot_at=now,
                        )
                        session.add(snapshot)
                        count += 1

                    elif mtype == MarketType.OU:
                        for o in outcomes:
                            point = o.get("point")
                            name = o.get("name", "").lower()
                            if name not in ("over", "under"):
                                continue
                            # Find the partner outcome
                            partner = next(
                                (x for x in outcomes if x.get("point") == point and x.get("name", "").lower() != name),
                                None,
                            )
                            if name == "over":
                                snapshot = OddsSnapshot(
                                    match_id=match_id,
                                    bookmaker=bm_key,
                                    market_type=mtype,
                                    line=point,
                                    home_odds=o.get("price"),          # over
                                    away_odds=partner.get("price") if partner else None,  # under
                                    snapshot_at=now,
                                )
                                session.add(snapshot)
                                count += 1
                                break  # one row per total line per bookmaker

                    elif mtype == MarketType.AH:
                        for o in outcomes:
                            if o.get("name", "").lower() == home_name.lower():
                                point = o.get("point", 0)
                                partner = next(
                                    (x for x in outcomes if x.get("name", "").lower() != home_name.lower()),
                                    None,
                                )
                                snapshot = OddsSnapshot(
                                    match_id=match_id,
                                    bookmaker=bm_key,
                                    market_type=mtype,
                                    line=point,
                                    home_odds=o.get("price"),
                                    away_odds=partner.get("price") if partner else None,
                                    snapshot_at=now,
                                )
                                session.add(snapshot)
                                count += 1
                                break

        session.commit()

    logger.info("Stored %d odds snapshots", count)
    return count


def run_poll_cycle() -> int:
    """Single poll: fetch + store. Called by scheduler."""
    events = fetch_odds()
    return process_and_store(events)
