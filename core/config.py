"""
core/config.py
--------------
Central configuration loaded from environment / .env file.
Every module imports from here — no scattered os.getenv() calls.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (two levels up from core/)
_root = Path(__file__).resolve().parent.parent
load_dotenv(_root / ".env", override=False)


class Settings:
    # ─── Database ────────────────────────────────────────────────────────────
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://postgres:password@localhost:5432/probity",
    )
    DATABASE_URL_SYNC: str = os.getenv(
        "DATABASE_URL_SYNC",
        "postgresql://postgres:password@localhost:5432/probity",
    )

    # ─── Odds API ─────────────────────────────────────────────────────────────
    ODDS_API_KEY: str = os.getenv("ODDS_API_KEY", "")
    ODDS_API_BASE_URL: str = os.getenv(
        "ODDS_API_BASE_URL", "https://api.the-odds-api.com/v4"
    )

    # ─── Historical data source ────────────────────────────────────────────────
    FOOTBALL_DATA_BASE_URL: str = os.getenv(
        "FOOTBALL_DATA_BASE_URL",
        "https://www.football-data.co.uk/mmz4281",
    )

    # ─── Model ────────────────────────────────────────────────────────────────
    MODEL_CALIBRATION_WINDOW_DAYS: int = int(
        os.getenv("MODEL_CALIBRATION_WINDOW_DAYS", "1095")
    )
    LAMBDA_DECAY: float = float(os.getenv("LAMBDA_DECAY", "0.0065"))
    MAX_GOALS: int = int(os.getenv("MAX_GOALS", "7"))

    # ─── Edge detection ───────────────────────────────────────────────────────
    MIN_EDGE_THRESHOLD: float = float(os.getenv("MIN_EDGE_THRESHOLD", "0.025"))
    EDGE_SCAN_INTERVAL_SECONDS: int = int(
        os.getenv("EDGE_SCAN_INTERVAL_SECONDS", "90")
    )
    EDGE_DEDUP_COOLDOWN_HOURS: float = float(
        os.getenv("EDGE_DEDUP_COOLDOWN_HOURS", "4")
    )

    # ─── CLV resolution ───────────────────────────────────────────────────────
    CLV_HOURS_BEFORE_KICKOFF: int = int(
        os.getenv("CLV_HOURS_BEFORE_KICKOFF", "1")
    )
    CLV_MAX_STALENESS_HOURS: int = int(
        os.getenv("CLV_MAX_STALENESS_HOURS", "24")
    )

    # ─── Kelly ────────────────────────────────────────────────────────────────
    KELLY_FRACTION: float = float(os.getenv("KELLY_FRACTION", "0.25"))
    BANKROLL_INITIAL: float = float(os.getenv("BANKROLL_INITIAL", "1000.0"))

    # ─── API ───────────────────────────────────────────────────────────────────
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_SECRET_KEY: str = os.getenv("API_SECRET_KEY", "change-me")

    # ─── Leagues map (football-data.co.uk codes) ──────────────────────────────
    # Eredivisie = NL1
    LEAGUE_CODES: dict[str, str] = {
        "NL1": "Eredivisie",
    }

    # Season codes: generated dynamically (see get_seasons()). Rollover in July.
    # e.g. 2526 => 2025/26; as of Feb 2026 current season is 2526.


def get_seasons(years_back: int = 5) -> list[str]:
    """
    Generate season codes dynamically. Season rollover in July.
    As of Feb 2026 → current season is 2526 (2025/26).
    Returns list of strings oldest-first e.g. ["2122", "2223", ..., "2526"].
    """
    from datetime import date
    today = date.today()
    # Current season end year: if month >= July we're in YY(Y+1); else (Y-1)Y
    if today.month >= 7:
        end_yy = today.year % 100
    else:
        end_yy = (today.year - 1) % 100
    seasons = []
    for i in range(years_back):
        y = end_yy - (years_back - 1 - i)
        if y < 0:
            continue
        y1 = (y + 1) % 100
        seasons.append(f"{y:02d}{y1:02d}")
    return seasons


settings = Settings()
