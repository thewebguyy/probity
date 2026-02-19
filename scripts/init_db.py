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


if __name__ == "__main__":
    init_db()
