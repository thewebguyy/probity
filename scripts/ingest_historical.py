"""
scripts/ingest_historical.py
-----------------------------
Download and upsert historical Eredivisie match data.

Usage:
    python scripts/ingest_historical.py
    python scripts/ingest_historical.py --seasons 2122 2223 2324 2425
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.historical import ingest_all_seasons, ingest_season
from core.config import get_seasons

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Ingest historical Eredivisie data")
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=None,
        help="Season codes e.g. 2223 2324 (default: dynamic from current date, July rollover)",
    )
    args = parser.parse_args()

    seasons = args.seasons or get_seasons()
    logger.info("Starting historical ingestion for seasons: %s", seasons)
    total = ingest_all_seasons(seasons)
    logger.info("Ingestion complete. Total new matches inserted: %d", total)


if __name__ == "__main__":
    main()
