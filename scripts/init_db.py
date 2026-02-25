"""
scripts/init_db.py
------------------
Create all database tables.
Run once before first use.

Usage:
    python scripts/init_db.py
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.database import Base, sync_engine
import core.models  # noqa: F401 — import all models to register them

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def init_db():
    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=sync_engine)
    logger.info("Done. Tables created:")
    for table_name in Base.metadata.tables.keys():
        logger.info("  ✓ %s", table_name)

    from sqlalchemy import text
    with sync_engine.connect() as conn:
        try:
            conn.execute(text(
                "ALTER TABLE edges ADD COLUMN IF NOT EXISTS last_seen_at TIMESTAMP WITH TIME ZONE"
            ))
            conn.commit()
            logger.info("  ✓ edges.last_seen_at (if added)")
        except Exception as e:
            logger.warning("Could not add last_seen_at: %s", e)
        try:
            conn.execute(text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_edge_recent ON edges "
                "(match_id, bookmaker, market_type, line, side, date_trunc('hour', detected_at))"
            ))
            conn.commit()
            logger.info("  ✓ uq_edge_recent index on edges")
        except Exception as e:
            logger.warning("Could not create uq_edge_recent index (may already exist or need PostgreSQL): %s", e)


if __name__ == "__main__":
    init_db()
