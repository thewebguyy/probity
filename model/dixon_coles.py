"""
model/dixon_coles.py
--------------------
Dixon-Coles Poisson model with MLE parameter estimation.

Reference:
  Dixon, M.J. & Coles, S.G. (1997)
  "Modelling Association Football Scores and Inefficiencies in the Football Betting Market"
  Applied Statistics, 46(2), 265–280.

Parameters estimated per team:
  attack_i    — team i attack strength
  defense_i   — team i defensive weakness
  mu          — home advantage (additive in log-λ space = multiplicative)
  rho         — Dixon-Coles low-score correction parameter

Model:
  λ_home = exp(attack_home - defense_away + mu)
  λ_away = exp(attack_away - defense_home)

Score probability:
  P(X=i, Y=j) = τ_ρ(i,j) × Poisson(i;λ_home) × Poisson(j;λ_away)

Where τ_ρ is the Dixon-Coles correction for (0,0), (1,0), (0,1), (1,1).

Identifiability constraint:
  Fix one team's attack parameter to zero; exclude from optimization vector;
  reconstruct full attack array post-fit. No penalty terms.

Optimization:
  Maximize weighted log-likelihood using scipy.optimize.minimize (L-BFGS-B).
  Fully vectorized: no Python-level per-match loops.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from core.config import settings
from core.database import SyncSessionLocal
from core.models import ModelRun, TeamParam

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Dixon-Coles low-score correction (vectorized)
# ─────────────────────────────────────────────────────────────────────────────


def _tau(x: int, y: int, lam_home: float, lam_away: float, rho: float) -> float:
    """
    Dixon-Coles τ correction for a single (x, y) score pair (for tests).
    Only affects (0,0), (1,0), (0,1), (1,1).
    """
    if x == 0 and y == 0:
        return 1 - lam_home * lam_away * rho
    elif x == 1 and y == 0:
        return 1 + lam_away * rho
    elif x == 0 and y == 1:
        return 1 + lam_home * rho
    elif x == 1 and y == 1:
        return 1 - rho
    return 1.0


def _tau_vectorized(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    lam_home: np.ndarray,
    lam_away: np.ndarray,
    rho: float,
) -> np.ndarray:
    """
    Dixon-Coles τ correction for score pairs. Vectorized.
    Only affects (0,0), (1,0), (0,1), (1,1); else 1.0.
    """
    tau = np.ones_like(lam_home, dtype=np.float64)
    # (0,0): 1 - lam_h * lam_a * rho  (DC paper: 1 - λμρ for 0-0)
    mask_00 = (home_goals == 0) & (away_goals == 0)
    tau[mask_00] = 1.0 - lam_home[mask_00] * lam_away[mask_00] * rho
    # (1,0): 1 + lam_away * rho
    mask_10 = (home_goals == 1) & (away_goals == 0)
    tau[mask_10] = 1.0 + lam_away[mask_10] * rho
    # (0,1): 1 + lam_home * rho
    mask_01 = (home_goals == 0) & (away_goals == 1)
    tau[mask_01] = 1.0 + lam_home[mask_01] * rho
    # (1,1): 1 - rho
    mask_11 = (home_goals == 1) & (away_goals == 1)
    tau[mask_11] = 1.0 - rho
    return tau


def _log_poisson_pmf(k: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Log Poisson PMF: log(lam^k * exp(-lam) / k!) = k*log(lam) - lam - gammaln(k+1)."""
    k = np.asarray(k, dtype=np.float64)
    lam = np.asarray(lam, dtype=np.float64)
    # Stable: avoid log(0), use gammaln for factorial
    log_lam = np.where(lam > 1e-300, np.log(lam), -np.inf)
    return k * log_lam - lam - gammaln(k + 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Parameter packing/unpacking (with fixed first-team attack = 0)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DCParams:
    teams: list[str]
    attack: dict[str, float] = field(default_factory=dict)
    defense: dict[str, float] = field(default_factory=dict)
    home_advantage: float = 0.3
    rho: float = -0.1
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    log_likelihood: float = 0.0
    n_matches: int = 0
    fitted_at: Optional[datetime] = None

    def pack(self, fixed_attack_team: Optional[str] = None) -> np.ndarray:
        """Pack parameters into a flat array for scipy. Excludes one team's attack (fixed to 0)."""
        n = len(self.teams)
        if fixed_attack_team is None:
            fixed_attack_team = self.teams[0]
        # Order: attack (n-1, excluding fixed), defense (n), mu, rho
        x = np.zeros(n - 1 + n + 2)
        idx = 0
        for i, t in enumerate(self.teams):
            if t != fixed_attack_team:
                x[idx] = self.attack.get(t, 0.0)
                idx += 1
        for i, t in enumerate(self.teams):
            x[idx] = self.defense.get(t, 0.0)
            idx += 1
        x[idx] = self.home_advantage
        x[idx + 1] = self.rho
        return x

    def unpack(
        self,
        x: np.ndarray,
        fixed_attack_team: Optional[str] = None,
    ) -> "DCParams":
        """Unpack from optimizer vector. First team's attack = 0."""
        n = len(self.teams)
        if fixed_attack_team is None:
            fixed_attack_team = self.teams[0]
        self.attack = {}
        self.attack[fixed_attack_team] = 0.0
        idx = 0
        for t in self.teams:
            if t != fixed_attack_team:
                self.attack[t] = float(x[idx])
                idx += 1
        for i, t in enumerate(self.teams):
            self.defense[t] = float(x[idx])
            idx += 1
        self.home_advantage = float(x[idx])
        self.rho = float(x[idx + 1])
        return self

    def lambda_home(self, home: str, away: str) -> float:
        return np.exp(
            self.attack.get(home, 0.0)
            - self.defense.get(away, 0.0)
            + self.home_advantage
        )

    def lambda_away(self, home: str, away: str) -> float:
        return np.exp(
            self.attack.get(away, 0.0)
            - self.defense.get(home, 0.0)
        )


# ─────────────────────────────────────────────────────────────────────────────
# MLE Fitting (fully vectorized)
# ─────────────────────────────────────────────────────────────────────────────


def fit_dixon_coles(
    df: pd.DataFrame,
    weights: Optional[np.ndarray] = None,
    initial_params: Optional[DCParams] = None,
) -> DCParams:
    """
    Fit the Dixon-Coles model via MLE. Fully vectorized; no per-match Python loops.

    Identifiability: first team's attack is fixed to 0 and excluded from the
    optimization vector; full attack array is reconstructed after fit.
    """
    if df.empty:
        raise ValueError("Cannot fit model on empty DataFrame")

    teams = sorted(set(df["home_team"].tolist()) | set(df["away_team"].tolist())
    n_teams = len(teams)
    team_idx = {t: i for i, t in enumerate(teams)}
    fixed_attack_team = teams[0]

    if weights is None:
        weights = np.ones(len(df), dtype=np.float64)
    weights = weights / weights.sum() * len(df)

    # Precompute arrays
    home_idx = np.array([team_idx[t] for t in df["home_team"]], dtype=np.intp)
    away_idx = np.array([team_idx[t] for t in df["away_team"]], dtype=np.intp)
    home_goals = df["home_goals"].values.astype(np.int64)
    away_goals = df["away_goals"].values.astype(np.int64)
    weights = np.asarray(weights, dtype=np.float64)

    # Optimization vector: attack[1..n-1], defense[0..n-1], mu, rho
    n_free = n_teams - 1 + n_teams + 2

    def neg_log_likelihood(x: np.ndarray) -> float:
        attack_free = x[: n_teams - 1]
        defense = x[n_teams - 1 : 2 * n_teams - 1]  # length n_teams
        mu = x[2 * n_teams - 1]
        rho = x[2 * n_teams]

        # Full attack: first team = 0, rest from attack_free
        attack_full = np.zeros(n_teams)
        attack_full[1:] = attack_free

        lam_home = np.exp(
            attack_full[home_idx] - defense[away_idx] + mu
        )
        lam_away = np.exp(
            attack_full[away_idx] - defense[home_idx]
        )

        tau = _tau_vectorized(home_goals, away_goals, lam_home, lam_away, rho)
        tau = np.maximum(tau, 1e-10)

        log_poisson_h = _log_poisson_pmf(home_goals, lam_home)
        log_poisson_a = _log_poisson_pmf(away_goals, lam_away)
        log_lik = np.log(tau) + log_poisson_h + log_poisson_a
        return -float(np.dot(weights, log_lik))

    if initial_params is not None:
        x0 = initial_params.pack(fixed_attack_team)
    else:
        x0 = np.zeros(n_free)
        x0[2 * n_teams - 1] = 0.3
        x0[2 * n_teams] = -0.1

    bounds = (
        [(None, None)] * (n_teams - 1)
        + [(None, None)] * n_teams
        + [(0.0, 1.0)]
        + [(-0.99, 0.2)]
    )

    logger.info("Fitting Dixon-Coles on %d matches, %d teams (vectorized)...", len(df), n_teams)
    result = minimize(
        neg_log_likelihood,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 2000, "ftol": 1e-10},
    )

    if not result.success:
        logger.warning("Optimizer did not fully converge: %s", result.message)

    params = DCParams(teams=teams)
    params.unpack(result.x, fixed_attack_team=fixed_attack_team)
    params.log_likelihood = -result.fun
    params.n_matches = len(df)
    params.fitted_at = datetime.now(timezone.utc)

    logger.info(
        "Fit complete — LL=%.2f, home_adv=%.3f, rho=%.3f",
        params.log_likelihood,
        params.home_advantage,
        params.rho,
    )
    return params


def save_params_to_db(params: DCParams, league: str = "NL1") -> str:
    """Persist model run + team params to database."""
    run_id = str(uuid.uuid4())

    with SyncSessionLocal() as session:
        run = ModelRun(
            run_id=run_id,
            league=league,
            window_days=settings.MODEL_CALIBRATION_WINDOW_DAYS,
            home_advantage=params.home_advantage,
            rho=params.rho,
            log_likelihood=params.log_likelihood,
            n_matches=params.n_matches,
        )
        session.add(run)

        from core.models import Team
        teams_in_db = session.query(Team).filter(Team.league == league).all()
        name_to_id = {t.name: t.team_id for t in teams_in_db}

        for team_name in params.teams:
            team_id = name_to_id.get(team_name)
            if not team_id:
                logger.warning("Team not found in DB: %s", team_name)
                continue
            tp = TeamParam(
                team_id=team_id,
                attack=params.attack[team_name],
                defense=params.defense[team_name],
                model_run_id=run_id,
            )
            session.add(tp)

        session.commit()

    logger.info("Saved model run %s to DB", run_id)
    return run_id


def load_latest_params(league: str = "NL1") -> Optional[DCParams]:
    """Load the most recent model run parameters from DB."""
    with SyncSessionLocal() as session:
        run = (
            session.query(ModelRun)
            .filter_by(league=league)
            .order_by(ModelRun.fitted_at.desc())
            .first()
        )
        if not run:
            logger.warning("No model run found in DB")
            return None

        team_params = (
            session.query(TeamParam)
            .filter_by(model_run_id=run.run_id)
            .all()
        )

        from core.models import Team
        teams_in_db = {t.team_id: t.name for t in session.query(Team).all()}

        teams = sorted([teams_in_db.get(tp.team_id, str(tp.team_id)) for tp in team_params])
        params = DCParams(teams=teams)
        params.attack = {teams_in_db.get(tp.team_id, str(tp.team_id)): tp.attack for tp in team_params}
        params.defense = {teams_in_db.get(tp.team_id, str(tp.team_id)): tp.defense for tp in team_params}
        params.home_advantage = run.home_advantage
        params.rho = run.rho
        params.log_likelihood = run.log_likelihood
        params.n_matches = run.n_matches
        params.run_id = run.run_id
        params.fitted_at = run.fitted_at

    return params
