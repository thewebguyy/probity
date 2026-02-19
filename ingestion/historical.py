"""
ingestion/historical.py
-----------------------
Downloads Eredivisie match CSVs from football-data.co.uk and upserts
into the `matches` and `teams` tables.

Data source: https://www.football-data.co.uk/netherlandsm.php
CSV URL pattern:
  https://www.football-data.co.uk/mmz4281/{SEASON}/N1.csv
  e.g. 2324/N1.csv = 2023/24 season

Column mapping (football-data.co.uk):
  Date, HomeTeam, AwayTeam, FTHG, FTAG  (full-time home/away goals)
  xG columns vary by season: AvgCH, AvgCA (average xG from Whoscored)
  We fall back to None if xG columns absent.
"""

from __future__ import annotations

import io
import logging
from datetime import datetime
from typing import Optional

import pandas as pd
import requests

from core.config import settings
from core.database import SyncSessionLocal
from core.models import Match, MatchStatus, Team

logger = logging.getLogger(__name__)


# ── Column aliases from football-data.co.uk ───────────────────────────────────
_DATE_COLS = ["Date"]
_HOME_COLS = ["HomeTeam", "Home"]
_AWAY_COLS = ["AwayTeam", "Away"]
_HGOAL_COLS = ["FTHG", "HG"]
_AGOAL_COLS = ["FTAG", "AG"]
_HXG_COLS = ["AvgCH", "B365CH"]   # best proxy when Whoscored not available
_AXG_COLS = ["AvgCA", "B365CA"]


def _first(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Return the first column name from candidates that exists in df."""
    return next((c for c in candidates if c in df.columns), None)


def _download_csv(season: str, league_code: str = "N1") -> Optional[pd.DataFrame]:
    url = f"{settings.FOOTBALL_DATA_BASE_URL}/{season}/{league_code}.csv"
    logger.info("Fetching %s", url)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), encoding="ISO-8859-1")
        # Drop fully empty rows that appear at CSV end
        df = df.dropna(how="all")
        logger.info("Downloaded %d rows for season %s", len(df), season)
        return df
    except Exception as exc:
        logger.warning("Could not download %s: %s", url, exc)
        return None


def _get_or_create_team(session, name: str) -> Team:
    name = name.strip()
    team = session.query(Team).filter_by(name=name).first()
    if not team:
        team = Team(name=name, league="NL1")
        session.add(team)
        session.flush()
        logger.debug("Created team: %s", name)
    return team


def _parse_date(val: str) -> Optional[datetime]:
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(val).strip(), fmt)
        except ValueError:
            continue
    return None


def ingest_season(season: str) -> int:
    """
    Download and upsert one season of Eredivisie data.
    Returns number of matches inserted/updated.
    """
    df = _download_csv(season)
    if df is None or df.empty:
        return 0

    # Column resolution
    date_col = _first(df, _DATE_COLS)
    home_col = _first(df, _HOME_COLS)
    away_col = _first(df, _AWAY_COLS)
    hgoal_col = _first(df, _HGOAL_COLS)
    agoal_col = _first(df, _AGOAL_COLS)
    hxg_col = _first(df, _HXG_COLS)
    axg_col = _first(df, _AXG_COLS)

    if not all([date_col, home_col, away_col, hgoal_col, agoal_col]):
        logger.error("Missing required columns in season %s. Cols: %s", season, df.columns.tolist())
        return 0

    count = 0
    with SyncSessionLocal() as session:
        for _, row in df.iterrows():
            raw_date = row.get(date_col)
            if pd.isna(raw_date):
                continue
            match_date = _parse_date(raw_date)
            if match_date is None:
                continue

            home_name = str(row[home_col]).strip()
            away_name = str(row[away_col]).strip()
            if not home_name or not away_name or home_name == "nan":
                continue

            home_team = _get_or_create_team(session, home_name)
            away_team = _get_or_create_team(session, away_name)

            # Goals
            def _safe_int(v):
                try:
                    return int(v) if not pd.isna(v) else None
                except (ValueError, TypeError):
                    return None

            def _safe_float(v):
                try:
                    return float(v) if not pd.isna(v) else None
                except (ValueError, TypeError):
                    return None

            home_goals = _safe_int(row.get(hgoal_col))
            away_goals = _safe_int(row.get(agoal_col))
            home_xg = _safe_float(row.get(hxg_col)) if hxg_col else None
            away_xg = _safe_float(row.get(axg_col)) if axg_col else None

            status = (
                MatchStatus.FINISHED
                if (home_goals is not None and away_goals is not None)
                else MatchStatus.SCHEDULED
            )

            # Upsert using unique constraint
            existing = (
                session.query(Match)
                .filter_by(
                    home_team_id=home_team.team_id,
                    away_team_id=away_team.team_id,
                    match_date=match_date,
                )
                .first()
            )
            if existing:
                existing.home_goals = home_goals
                existing.away_goals = away_goals
                existing.home_xg = home_xg
                existing.away_xg = away_xg
                existing.status = status
            else:
                match = Match(
                    league="NL1",
                    match_date=match_date,
                    season=season,
                    home_team_id=home_team.team_id,
                    away_team_id=away_team.team_id,
                    home_goals=home_goals,
                    away_goals=away_goals,
                    home_xg=home_xg,
                    away_xg=away_xg,
                    status=status,
                )
                session.add(match)
                count += 1

        session.commit()

    logger.info("Ingested %d new matches for season %s", count, season)
    return count


def ingest_all_seasons(seasons: Optional[list[str]] = None) -> int:
    """Ingest multiple seasons. Default: settings.SEASONS."""
    seasons = seasons or settings.SEASONS
    total = 0
    for s in seasons:
        total += ingest_season(s)
    return total
