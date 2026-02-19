"""
core/database.py
----------------
SQLAlchemy async engine + session factory.
Also provides a synchronous engine for scripts/init_db.
"""

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from core.config import settings

# ── Async engine (used by FastAPI) ────────────────────────────────────────────
async_engine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    echo=False,
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)

# ── Sync engine (used by scripts) ─────────────────────────────────────────────
sync_engine = create_engine(
    settings.DATABASE_URL_SYNC,
    pool_pre_ping=True,
)

SyncSessionLocal = sessionmaker(bind=sync_engine, autocommit=False, autoflush=False)


# ── Base class ────────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


# ── Dependency for FastAPI ────────────────────────────────────────────────────
async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
