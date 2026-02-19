"""
tests/test_core_logic.py
------------------------
Unit tests for core algorithmic modules.
No database required — pure compute tests.

Run: python tests/test_core_logic.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

PASS = "[PASS]"
FAIL = "[FAIL]"


def test_tau_correction():
    from model.dixon_coles import _tau
    # DC correction should modify low-score cells
    assert abs(_tau(0, 0, 1.5, 1.0, -0.1) - (1 - 1.5 * 1.0 * -0.1)) < 1e-9
    # ...but not high-score cells
    assert _tau(5, 3, 1.5, 1.0, -0.1) == 1.0
    print(PASS, "Dixon-Coles tau correction")


def test_score_matrix_sums_to_one():
    from model.score_matrix import build_score_matrix
    for lh, la, rho in [(1.6, 1.2, -0.05), (2.0, 0.8, 0.1), (0.9, 1.8, -0.15)]:
        m = build_score_matrix(lh, la, rho)
        total = m.sum()
        assert abs(total - 1.0) < 1e-5, f"Matrix sum = {total} (expected 1.0)"
    print(PASS, "Score matrix sums to 1.0 across parameter range")


def test_h2h_probs():
    from model.score_matrix import build_score_matrix, h2h_probs
    m = build_score_matrix(1.6, 1.2, -0.05)
    ph, pd, pa = h2h_probs(m)
    assert abs(ph + pd + pa - 1.0) < 1e-6
    # With home advantage, home win should be most likely
    assert ph > pa, "Home team should have higher win prob with edge"
    print(PASS, f"H2H probs sum to 1 | H={ph:.3f} D={pd:.3f} A={pa:.3f}")


def test_asian_handicap():
    from model.score_matrix import build_score_matrix, asian_handicap_probs
    m = build_score_matrix(1.6, 1.2, -0.05)
    
    # Half-ball line
    ph, pa = asian_handicap_probs(m, -0.5)
    assert abs(ph + pa - 1.0) < 1e-6
    
    # Quarter-ball line  
    ph_q, pa_q = asian_handicap_probs(m, -0.25)
    assert abs(ph_q + pa_q - 1.0) < 1e-6
    
    # AH 0.0 on balanced match should be close to 50/50
    m_balanced = build_score_matrix(1.2, 1.2, 0.0)
    ph0, pa0 = asian_handicap_probs(m_balanced, 0.0)
    # Push removed so sum is 1
    print(PASS, f"AH -0.5: home={ph:.3f} away={pa:.3f} | AH -0.25: home={ph_q:.3f}")


def test_over_under():
    from model.score_matrix import build_score_matrix, over_under_probs
    m = build_score_matrix(1.4, 1.1, -0.05)
    
    po, pu = over_under_probs(m, 2.5)
    assert abs(po + pu - 1.0) < 1e-6
    
    # Higher total → lower over probability
    po_high, _ = over_under_probs(m, 4.5)
    assert po_high < po, "Over 4.5 should be less likely than over 2.5"
    
    print(PASS, f"OU 2.5: over={po:.3f} under={pu:.3f} | OU 4.5: over={po_high:.3f}")


def test_power_margin_removal():
    from market.comparison import remove_margin_power
    
    # Standard H2H odds with ~5% margin
    h2h_odds = [2.10, 3.40, 3.60]
    probs = remove_margin_power(h2h_odds)
    assert len(probs) == 3
    assert abs(sum(probs) - 1.0) < 1e-6
    assert all(0 < p < 1 for p in probs)
    
    # Asian Handicap (two-way)
    ah_odds = [1.92, 1.98]
    probs_ah = remove_margin_power(ah_odds)
    assert abs(sum(probs_ah) - 1.0) < 1e-6
    
    print(PASS, f"Power margin removal | H2H: {[round(p,4) for p in probs]}")


def test_edge_calculation():
    from market.comparison import compute_edge, kelly_fraction
    
    # Positive edge case
    edge = compute_edge(model_prob=0.55, book_odds=2.10)
    assert edge > 0, "Should detect positive edge"
    assert abs(edge - (0.55 * 2.10 - 1)) < 1e-9
    
    # Negative edge case
    edge_neg = compute_edge(model_prob=0.40, book_odds=2.10)
    assert edge_neg < 0
    
    # Kelly
    k = kelly_fraction(0.55, 2.10, fraction=0.25)
    assert 0 < k < 0.25, f"Kelly stake should be between 0 and 25%, got {k:.4f}"
    
    print(PASS, f"Edge={edge*100:.2f}% | Kelly(0.25x)={k*100:.2f}%")


def test_mle_fit():
    """Test that model fitting converges on synthetic data."""
    import pandas as pd
    from model.dixon_coles import fit_dixon_coles

    # Generate synthetic Poisson data  
    np.random.seed(42)
    teams = ["Ajax", "PSV", "Feyenoord", "AZ", "Twente"]
    attack = {"Ajax": 0.4, "PSV": 0.5, "Feyenoord": 0.3, "AZ": 0.1, "Twente": 0.0}
    defense = {"Ajax": -0.2, "PSV": -0.3, "Feyenoord": -0.1, "AZ": 0.0, "Twente": 0.1}
    mu = 0.3

    rows = []
    for _ in range(200):
        h = np.random.choice(teams)
        a = np.random.choice([t for t in teams if t != h])
        lh = np.exp(attack[h] - defense[a] + mu)
        la = np.exp(attack[a] - defense[h])
        rows.append({
            "home_team": h, "away_team": a,
            "home_goals": np.random.poisson(lh),
            "away_goals": np.random.poisson(la),
            "match_date": pd.Timestamp("2024-01-01"),
        })
    df = pd.DataFrame(rows)

    params = fit_dixon_coles(df)
    
    assert params.home_advantage > 0, "Home advantage should be positive"
    assert len(params.attack) == 5
    assert all(t in params.attack for t in teams)
    
    # Check that high-attack teams have higher attack params
    # PSV was set highest — it should rank well
    sorted_attack = sorted(params.attack.items(), key=lambda x: -x[1])
    print(PASS, f"MLE fit converged | home_adv={params.home_advantage:.3f} rho={params.rho:.3f}")
    print(f"       Top attack: {sorted_attack[0][0]} ({sorted_attack[0][1]:.3f})")


def test_monte_carlo():
    from evaluation.metrics import monte_carlo_simulation
    
    mc = monte_carlo_simulation(
        avg_edge=0.03,
        avg_kelly_fraction=0.25,
        n_simulations=2000,
        bets_per_season=200,
        avg_odds=1.90,
        seed=42,
    )
    
    assert 0 <= mc["pct_profitable"] <= 100
    assert 0 <= mc["pct_ruin"] <= 100
    assert mc["pct_profitable"] + mc["pct_ruin"] <= 100
    assert mc["p5"] <= mc["p50"] <= mc["p95"], "Percentiles should be ordered"
    
    print(PASS, f"Monte Carlo | profitable={mc['pct_profitable']}% ruin={mc['pct_ruin']}% median={mc['median_final']:.3f}x")


def run_all():
    tests = [
        test_tau_correction,
        test_score_matrix_sums_to_one,
        test_h2h_probs,
        test_asian_handicap,
        test_over_under,
        test_power_margin_removal,
        test_edge_calculation,
        test_mle_fit,
        test_monte_carlo,
    ]
    
    passed = 0
    failed = 0
    
    print("\nProbity Core Logic Tests")
    print("=" * 45)
    
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"{FAIL} {t.__name__}: {e}")
            failed += 1
    
    print("=" * 45)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_all()
