"""
scripts/setup_dev.py
--------------------
Automated setup for the development environment.
1. Initializes the database (Postgres or SQLite).
2. Ingests 3 seasons of historical Eredivisie data.
3. Fits the initial Dixon-Coles model.
4. Generates mock live odds and edges for demonstration.

Usage:
    python scripts/setup_dev.py
"""

import sys
import os
import argparse
from datetime import datetime, timedelta, timezone
import random

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database import Base, sync_engine, SyncSessionLocal
from ingestion.historical import ingest_all_seasons
from model.dixon_coles import fit_dixon_coles, save_params_to_db
from model.score_matrix import compute_and_store_fair_odds
from features.engineering import load_match_dataframe, compute_decay_weights
from core.models import Match, MatchStatus, Team, OddsSnapshot, MarketType, Edge, BetSide, Bet

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock-history", action="store_true", help="Generate mock bets for performance testing")
    args = parser.parse_args()

    print("--- 1. Initializing Database ---")
    Base.metadata.create_all(bind=sync_engine)
    print("✓ Tables created.")

    print("\n--- 2. Ingesting Historical Data (Phase 1) ---")
    # This reaches out to football-data.co.uk
    try:
        total = ingest_all_seasons(["2122", "2223", "2324"])
        print(f"✓ Ingested {total} matches from 3 seasons.")
    except Exception as e:
        print(f"× Error ingesting: {e}")
        return

    print("\n--- 3. Fitting Dixon-Coles Model ---")
    df = load_match_dataframe(league="NL1")
    if df.empty:
        print("× No data found to fit model.")
        return
    weights = compute_decay_weights(df)
    params = fit_dixon_coles(df, weights=weights)
    run_id = save_params_to_db(params, league="NL1")
    print(f"✓ Model fitted. Run ID: {run_id}")

    print("\n--- 4. Precomputing Fair Odds for Upcoming Matches ---")
    now = datetime.now(timezone.utc)
    with SyncSessionLocal() as session:
        upcoming = session.query(Match).filter(
            Match.status == MatchStatus.SCHEDULED,
            Match.match_date >= now
        ).all()
        teams = {t.team_id: t.name for t in session.query(Team).all()}
        
        for match in upcoming:
            compute_and_store_fair_odds(
                match.match_id, 
                teams[match.home_team_id], 
                teams[match.away_team_id], 
                params
            )
    print(f"✓ Fair odds computed for {len(upcoming)} matches.")

    if args.mock_history:
        print("\n--- 5. Generating Mock Performance Data ---")
        generate_mock_performance()
        print("✓ Mock bets and edges generated.")

    print("\nSetup Complete. You can now run:")
    print("  python scripts/run_api.py --reload")

def generate_mock_performance():
    """Generate 50 historical bets with varying results to populate the dashboard."""
    with SyncSessionLocal() as session:
        matches = session.query(Match).filter(Match.status == MatchStatus.FINISHED).limit(50).all()
        current_bankroll = 1000.0
        
        for i, match in enumerate(matches):
            # Create a mock edge
            odds = random.uniform(1.8, 3.5)
            prob = (1.0 / odds) * 1.05 # 5% edge
            
            edge = Edge(
                match_id=match.match_id,
                bookmaker="Pinnacle",
                market_type=MarketType.H2H,
                side=BetSide.HOME,
                model_prob=prob,
                book_odds=odds,
                edge_value=prob * odds - 1,
                detected_at=match.match_date - timedelta(hours=5),
                closing_odds=odds * random.uniform(0.95, 1.05),
                clv=(odds / (odds * 0.98)) - 1 # mock positive CLV
            )
            session.add(edge)
            session.flush()

            # Create a mock bet
            stake = 20.0
            win = random.random() < (1.0/odds)
            pnl = stake * (odds - 1) if win else -stake
            
            bet = Bet(
                edge_id=edge.edge_id,
                match_id=match.match_id,
                bookmaker="Pinnacle",
                market_type=MarketType.H2H,
                side=BetSide.HOME,
                stake=stake,
                odds=odds,
                result="win" if win else "loss",
                pnl=pnl,
                placed_at=match.match_date - timedelta(hours=1),
                settled_at=match.match_date + timedelta(hours=2)
            )
            session.add(bet)
            
        session.commit()

if __name__ == "__main__":
    main()
