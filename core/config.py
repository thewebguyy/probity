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

    # Season codes used by football-data.co.uk URL pattern
    # e.g. 2324 => 2023/24
    SEASONS: list[str] = [
        "2122",
        "2223",
        "2324",
        "2425",
    ]


settings = Settings()
