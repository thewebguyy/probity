"""
core/models.py
--------------
SQLAlchemy ORM models for all database tables.

Tables
------
teams            – team registry
matches          – fixture results + xG
odds_snapshots   – timestamped bookmaker odds (append-only)
fair_odds        – model-computed fair prices per match
edges            – detected +EV opportunities
bets             – logged execution (manual or automated)
"""

from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import relationship

from core.database import Base


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────


class MatchStatus(str, enum.Enum):
    SCHEDULED = "scheduled"
    LIVE = "live"
    FINISHED = "finished"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"


class MarketType(str, enum.Enum):
    AH = "AH"   # Asian Handicap
    OU = "OU"   # Over/Under
    H2H = "H2H" # Head-to-Head (1X2)


class BetSide(str, enum.Enum):
    HOME = "home"
    AWAY = "away"
    OVER = "over"
    UNDER = "under"


# ─────────────────────────────────────────────────────────────────────────────
# Teams
# ─────────────────────────────────────────────────────────────────────────────


class Team(Base):
    __tablename__ = "teams"

    team_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(120), nullable=False, unique=True)
    short_name = Column(String(20))
    league = Column(String(20), default="NL1")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    home_matches = relationship(
        "Match", foreign_keys="Match.home_team_id", back_populates="home_team"
    )
    away_matches = relationship(
        "Match", foreign_keys="Match.away_team_id", back_populates="away_team"
    )
    attack_params = relationship("TeamParam", back_populates="team")


# ─────────────────────────────────────────────────────────────────────────────
# Matches
# ─────────────────────────────────────────────────────────────────────────────


class Match(Base):
    __tablename__ = "matches"

    match_id = Column(Integer, primary_key=True, autoincrement=True)
    external_id = Column(String(100), unique=True, nullable=True)  # API ref id
    league = Column(String(20), default="NL1", nullable=False)
    match_date = Column(DateTime(timezone=True), nullable=False)
    season = Column(String(10))  # e.g. "2324"

    home_team_id = Column(Integer, ForeignKey("teams.team_id"), nullable=False)
    away_team_id = Column(Integer, ForeignKey("teams.team_id"), nullable=False)

    home_goals = Column(Integer, nullable=True)
    away_goals = Column(Integer, nullable=True)

    # xG (from data source if available)
    home_xg = Column(Float, nullable=True)
    away_xg = Column(Float, nullable=True)

    status = Column(Enum(MatchStatus), default=MatchStatus.SCHEDULED, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    home_team = relationship("Team", foreign_keys=[home_team_id], back_populates="home_matches")
    away_team = relationship("Team", foreign_keys=[away_team_id], back_populates="away_matches")
    odds_snapshots = relationship("OddsSnapshot", back_populates="match")
    fair_odds = relationship("FairOdds", back_populates="match")
    edges = relationship("Edge", back_populates="match")

    __table_args__ = (
        UniqueConstraint("home_team_id", "away_team_id", "match_date", name="uq_match"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Odds Snapshots — APPEND ONLY. Never update, never delete.
# ─────────────────────────────────────────────────────────────────────────────


class OddsSnapshot(Base):
    """
    Immutable record of bookmaker odds at a specific point in time.
    Append-only: this table is the audit trail for CLV calculation.
    """

    __tablename__ = "odds_snapshots"

    snapshot_id = Column(BigInteger, primary_key=True, autoincrement=True)
    match_id = Column(Integer, ForeignKey("matches.match_id"), nullable=False)

    bookmaker = Column(String(60), nullable=False)
    market_type = Column(Enum(MarketType), nullable=False)
    line = Column(Float, nullable=True)  # AH line (e.g. -0.5) or OU total (e.g. 2.5)

    home_odds = Column(Float, nullable=True)
    away_odds = Column(Float, nullable=True)

    # For H2H: draw odds
    draw_odds = Column(Float, nullable=True)

    is_closing = Column(Boolean, default=False)  # True when snapshot near kickoff
    liquidity = Column(Float, nullable=True)     # Optional: market volume/depth
    snapshot_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    match = relationship("Match", back_populates="odds_snapshots")


# ─────────────────────────────────────────────────────────────────────────────
# Fair Odds (Model Output)
# ─────────────────────────────────────────────────────────────────────────────


class FairOdds(Base):
    """
    Model-computed fair probabilities and fair decimal odds for a match.
    One row per market per model run.
    """

    __tablename__ = "fair_odds"

    fair_odds_id = Column(BigInteger, primary_key=True, autoincrement=True)
    match_id = Column(Integer, ForeignKey("matches.match_id"), nullable=False)
    market_type = Column(Enum(MarketType), nullable=False)
    line = Column(Float, nullable=True)

    home_prob = Column(Float, nullable=False)
    away_prob = Column(Float, nullable=True)
    draw_prob = Column(Float, nullable=True)

    home_fair_odds = Column(Float, nullable=False)
    away_fair_odds = Column(Float, nullable=True)
    draw_fair_odds = Column(Float, nullable=True)

    # Model metadata
    lambda_home = Column(Float)   # Poisson rate home
    lambda_away = Column(Float)   # Poisson rate away
    model_version = Column(String(40))
    computed_at = Column(DateTime(timezone=True), server_default=func.now())

    match = relationship("Match", back_populates="fair_odds")


# ─────────────────────────────────────────────────────────────────────────────
# Edges (Detected +EV Opportunities)
# ─────────────────────────────────────────────────────────────────────────────


class Edge(Base):
    """
    Logged edge detection event. One row per +EV opportunity detected.
    """

    __tablename__ = "edges"

    edge_id = Column(BigInteger, primary_key=True, autoincrement=True)
    match_id = Column(Integer, ForeignKey("matches.match_id"), nullable=False)

    bookmaker = Column(String(60))
    market_type = Column(Enum(MarketType), nullable=False)
    line = Column(Float, nullable=True)
    side = Column(Enum(BetSide), nullable=False)

    model_prob = Column(Float, nullable=False)
    book_odds = Column(Float, nullable=False)   # odds at detection time
    edge_value = Column(Float, nullable=False)  # (model_prob × book_odds) - 1
    liquidity = Column(Float, nullable=True)    # liquidity at detection

    # CLV tracking
    closing_odds = Column(Float, nullable=True)  # filled in post-kickoff
    clv = Column(Float, nullable=True)           # closing_odds / detection_odds - 1

    kelly_stake_fraction = Column(Float, nullable=True)

    detected_at = Column(DateTime(timezone=True), server_default=func.now())
    resolved_at = Column(DateTime(timezone=True), nullable=True)

    match = relationship("Match", back_populates="edges")


# ─────────────────────────────────────────────────────────────────────────────
# Team Model Parameters (MLE output, stored per model run)
# ─────────────────────────────────────────────────────────────────────────────


class TeamParam(Base):
    """
    Dixon-Coles MLE parameters for each team in a given model run.
    """

    __tablename__ = "team_params"

    param_id = Column(BigInteger, primary_key=True, autoincrement=True)
    team_id = Column(Integer, ForeignKey("teams.team_id"), nullable=False)

    attack = Column(Float, nullable=False)
    defense = Column(Float, nullable=False)

    model_run_id = Column(String(40), nullable=False)  # UUID of model run
    fitted_at = Column(DateTime(timezone=True), server_default=func.now())

    team = relationship("Team", back_populates="attack_params")


# ─────────────────────────────────────────────────────────────────────────────
# Model Runs
# ─────────────────────────────────────────────────────────────────────────────


class ModelRun(Base):
    """
    Metadata record for each Dixon-Coles model fit.
    """

    __tablename__ = "model_runs"

    run_id = Column(String(40), primary_key=True)
    league = Column(String(20), default="NL1")
    window_days = Column(Integer)
    home_advantage = Column(Float)
    rho = Column(Float)                  # Dixon-Coles rho
    log_likelihood = Column(Float)
    n_matches = Column(Integer)
    fitted_at = Column(DateTime(timezone=True), server_default=func.now())
    notes = Column(Text, nullable=True)


# ─────────────────────────────────────────────────────────────────────────────
# Bets (execution log – optional, manual)
# ─────────────────────────────────────────────────────────────────────────────


class Bet(Base):
    """
    Records a placed bet (manual entry). Linked to an Edge.
    """

    __tablename__ = "bets"

    bet_id = Column(BigInteger, primary_key=True, autoincrement=True)
    edge_id = Column(BigInteger, ForeignKey("edges.edge_id"), nullable=True)
    match_id = Column(Integer, ForeignKey("matches.match_id"), nullable=False)

    bookmaker = Column(String(60))
    market_type = Column(Enum(MarketType))
    line = Column(Float, nullable=True)
    side = Column(Enum(BetSide))

    stake = Column(Float, nullable=False)
    odds = Column(Float, nullable=False)

    result = Column(String(10), nullable=True)   # "win" | "loss" | "void"
    pnl = Column(Float, nullable=True)           # profit/loss in currency units

    placed_at = Column(DateTime(timezone=True), server_default=func.now())
    settled_at = Column(DateTime(timezone=True), nullable=True)
    notes = Column(Text, nullable=True)
