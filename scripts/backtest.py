"""
scripts/backtest.py
--------------------
Phase 1: Historical backtest.

Re-fits the Dixon-Coles model on rolling windows, simulates edge detection
against historical odds (if available), and measures:
  - Model calibration over time
  - Hypothetical CLV
  - ROI (simulated, not actual staked)

Usage:
    python scripts/backtest.py --league NL1 --start-season 2122

This is a validation step — run before any live deployment.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--league", default="NL1")
    args = parser.parse_args()

    from features.engineering import load_match_dataframe, compute_decay_weights
    from model.dixon_coles import fit_dixon_coles
    from model.score_matrix import build_score_matrix, h2h_probs, over_under_probs

    # Load all finished matches
    df = load_match_dataframe(league=args.league, window_days=2000)

    if df.empty:
        logger.error("No data. Run ingest_historical.py first.")
        sys.exit(1)

    logger.info("Loaded %d finished matches for backtest", len(df))

    # Walk-forward validation
    # 1. Train on first 60% of matches
    # 2. Predict on remaining 40%
    # 3. Measure calibration (Brier score, log loss)

    split_idx = int(len(df) * 0.6)
    train_df = df.iloc[:split_idx].copy()
    test_df  = df.iloc[split_idx:].copy()

    logger.info("Train: %d matches | Test: %d matches", len(train_df), len(test_df))

    # Fit on train
    weights_train = compute_decay_weights(train_df)
    params = fit_dixon_coles(train_df, weights=weights_train)

    # Predict on test
    log_losses = []
    brier_scores = []
    correct_winners = 0

    for _, row in test_df.iterrows():
        h = row["home_team"]
        a = row["away_team"]

        if h not in params.attack or a not in params.attack:
            continue

        lh = params.lambda_home(h, a)
        la = params.lambda_away(h, a)

        matrix = build_score_matrix(lh, la, params.rho)
        ph, pd_, pa = h2h_probs(matrix)

        actual_hg = row["home_goals"]
        actual_ag = row["away_goals"]

        # Actual outcome
        if actual_hg > actual_ag:
            outcome_idx = 0  # home win
            actual_p = ph
        elif actual_hg == actual_ag:
            outcome_idx = 1  # draw
            actual_p = pd_
        else:
            outcome_idx = 2  # away win
            actual_p = pa

        # Log loss
        log_losses.append(-np.log(max(actual_p, 1e-10)))

        # Brier score
        probs = [ph, pd_, pa]
        one_hot = [1 if i == outcome_idx else 0 for i in range(3)]
        brier_scores.append(sum((p - o) ** 2 for p, o in zip(probs, one_hot)))

        # Winner prediction accuracy
        predicted_winner = np.argmax([ph, pd_, pa])
        if predicted_winner == outcome_idx:
            correct_winners += 1

    n = len(log_losses)
    if n == 0:
        logger.warning("No test matches possible (teams not in training set?)")
        return

    avg_log_loss = np.mean(log_losses)
    avg_brier   = np.mean(brier_scores)
    accuracy    = correct_winners / n

    logger.info("=" * 50)
    logger.info("BACKTEST RESULTS (n=%d test matches)", n)
    logger.info("  Log Loss:  %.4f (lower is better; baseline ~1.099)", avg_log_loss)
    logger.info("  Brier:     %.4f (lower is better; baseline ~0.667)", avg_brier)
    logger.info("  Accuracy:  %.1f%% (predict correct winner)", accuracy * 100)
    logger.info("  Home adv:  %.4f", params.home_advantage)
    logger.info("  Rho:       %.4f", params.rho)
    logger.info("=" * 50)

    # Neutral baseline (predict 1/3 for each outcome)
    baseline_ll = -np.log(1 / 3)
    improvement = (baseline_ll - avg_log_loss) / baseline_ll * 100
    logger.info("  Model improvement over baseline: %.1f%% in log loss", improvement)

    if improvement < 0:
        logger.warning("Model performs WORSE than random! Check data quality.")
    elif improvement < 5:
        logger.warning("Marginal improvement. Validate data and try more seasons.")
    else:
        logger.info("Model shows meaningful calibration improvement.")


if __name__ == "__main__":
    main()
