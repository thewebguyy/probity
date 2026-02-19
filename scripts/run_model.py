"""
scripts/run_model.py
---------------------
Fit the Dixon-Coles model on historical data and precompute fair odds
for all upcoming matches.

Usage:
    python scripts/run_model.py
    python scripts/run_model.py --league NL1 --dry-run

Schedule: run nightly (e.g. 02:00 via cron or APScheduler).
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Fit Dixon-Coles model")
    parser.add_argument("--league", default="NL1")
    parser.add_argument("--dry-run", action="store_true", help="Fit but don't save to DB")
    args = parser.parse_args()

    from core.config import settings
    from features.engineering import load_match_dataframe, compute_decay_weights
    from model.dixon_coles import fit_dixon_coles, save_params_to_db
    from model.score_matrix import compute_and_store_fair_odds
    from core.database import SyncSessionLocal
    from core.models import Match, MatchStatus, Team
    from datetime import datetime, timezone, timedelta
    from sqlalchemy import select

    # 1. Load feature data
    logger.info("Loading match data (window: %d days)...", settings.MODEL_CALIBRATION_WINDOW_DAYS)
    df = load_match_dataframe(league=args.league)

    if df.empty:
        logger.error("No match data available. Run ingest_historical.py first.")
        sys.exit(1)

    logger.info("Loaded %d matches for model fitting", len(df))

    # 2. Compute decay weights
    weights = compute_decay_weights(df)
    logger.info(
        "Weight stats — min: %.4f, max: %.4f, mean: %.4f",
        weights.min(), weights.max(), weights.mean(),
    )

    # 3. Fit Dixon-Coles
    params = fit_dixon_coles(df, weights=weights)
    logger.info(
        "Model fit complete:\n"
        "  Teams: %d\n"
        "  Home advantage: %.4f\n"
        "  Rho: %.4f\n"
        "  Log-likelihood: %.2f",
        len(params.teams),
        params.home_advantage,
        params.rho,
        params.log_likelihood,
    )

    # Print top attack/defense teams
    top_attack = sorted(params.attack.items(), key=lambda x: -x[1])[:5]
    top_defense = sorted(params.defense.items(), key=lambda x: x[1])[:5]
    logger.info("Top 5 attack teams: %s", top_attack)
    logger.info("Top 5 defense teams (lower=better): %s", top_defense)

    if args.dry_run:
        logger.info("Dry run — not saving to DB")
        return

    # 4. Save params
    run_id = save_params_to_db(params, league=args.league)
    logger.info("Saved model run: %s", run_id)

    # 5. Precompute fair odds for upcoming matches
    now = datetime.now(timezone.utc)
    with SyncSessionLocal() as session:
        upcoming = (
            session.query(Match)
            .filter(
                Match.league == args.league,
                Match.status == MatchStatus.SCHEDULED,
                Match.match_date >= now,
                Match.match_date <= now + timedelta(days=14),
            )
            .all()
        )
        teams = {t.team_id: t.name for t in session.query(Team).all()}

    logger.info("Computing fair odds for %d upcoming matches...", len(upcoming))

    for match in upcoming:
        home_name = teams.get(match.home_team_id)
        away_name = teams.get(match.away_team_id)

        if not home_name or not away_name:
            logger.warning("Missing team name for match %d", match.match_id)
            continue

        if home_name not in params.attack or away_name not in params.attack:
            logger.warning(
                "Team %s or %s not in model params — not enough history",
                home_name, away_name,
            )
            continue

        try:
            result = compute_and_store_fair_odds(
                match_id=match.match_id,
                home_team=home_name,
                away_team=away_name,
                params=params,
            )
            lh = result.get("lambda_home", 0)
            la = result.get("lambda_away", 0)
            h2h = result.get("h2h", {})
            logger.info(
                "  %s vs %s | λ_h=%.3f λ_a=%.3f | H=%.3f D=%.3f A=%.3f",
                home_name, away_name, lh, la,
                h2h.get("home_prob", 0),
                h2h.get("draw_prob", 0),
                h2h.get("away_prob", 0),
            )
        except Exception as e:
            logger.error("Failed for match %d: %s", match.match_id, e)

    logger.info("Model run complete.")


if __name__ == "__main__":
    main()
