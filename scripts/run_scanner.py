"""
scripts/run_scanner.py
-----------------------
Edge scanner with APScheduler — polls odds and scans for +EV
every EDGE_SCAN_INTERVAL_SECONDS (default 90 sec).

Also resolves CLV for edges that have closed.

Usage:
    python scripts/run_scanner.py

Run continuously in background (Service 3 in deployment).
"""

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def poll_and_scan():
    """One complete poll+scan cycle."""
    from ingestion.odds_scraper import run_poll_cycle
    from market.comparison import scan_for_edges, resolve_clv

    logger.info("=== Starting poll cycle ===")

    # 1. Fetch live odds
    n_snaps = run_poll_cycle()
    logger.info("Fetched %d new odds snapshots", n_snaps)

    # 2. Scan for edges
    edges = scan_for_edges()
    if edges:
        logger.info("*** %d EDGES DETECTED ***", len(edges))
        for e in edges:
            logger.info(
                "  [+EV] %s | %s line=%s %s | edge=%.1f%% | odds=%.3f | kelly=%.2f%%",
                e["match_date"][:10],
                e["market"],
                e["line"],
                e["side"].upper(),
                e["edge"],
                e["book_odds"],
                e["kelly_fraction"] * 100,
            )
    else:
        logger.info("No edges detected this cycle.")

    # 3. Resolve CLV for closed markets
    resolved = resolve_clv()
    if resolved:
        logger.info("Resolved CLV for %d edges", resolved)

    logger.info("=== Cycle complete ===\n")


def main():
    from apscheduler.schedulers.blocking import BlockingScheduler
    from core.config import settings

    scheduler = BlockingScheduler()

    interval = settings.EDGE_SCAN_INTERVAL_SECONDS
    logger.info("Edge scanner starting — polling every %d seconds", interval)

    # Run immediately on start
    poll_and_scan()

    scheduler.add_job(
        poll_and_scan,
        "interval",
        seconds=interval,
        id="edge_scan",
        max_instances=1,
        coalesce=True,
    )

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Scanner stopped.")


if __name__ == "__main__":
    main()
