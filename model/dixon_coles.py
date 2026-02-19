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
  mean(attack_i) = 0  (log-scale, so geometric mean = 1)

Optimization:
  Maximize weighted log-likelihood using scipy.optimize.minimize (L-BFGS-B).
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
from scipy.stats import poisson

from core.config import settings
from core.database import SyncSessionLocal
from core.models import ModelRun, TeamParam

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Dixon-Coles low-score correction
# ─────────────────────────────────────────────────────────────────────────────


def _tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """
    Dixon-Coles τ correction for (x, y) score pair.
    Only affects scores (0,0), (1,0), (0,1), (1,1).
    """
    if x == 0 and y == 0:
        return 1 - lam * mu * rho
    elif x == 1 and y == 0:
        return 1 + mu * rho
    elif x == 0 and y == 1:
        return 1 + lam * rho
    elif x == 1 and y == 1:
        return 1 - rho
    else:
        return 1.0


def _log_likelihood_single(
    home_g: int,
    away_g: int,
    lam_h: float,
    lam_a: float,
    rho: float,
    weight: float = 1.0,
) -> float:
    """
    Log-likelihood contribution of one match.
    """
    eps = 1e-10
    p_home = poisson.pmf(home_g, lam_h)
    p_away = poisson.pmf(away_g, lam_a)
    tau_val = _tau(home_g, away_g, lam_h, lam_a, rho)
    ll = np.log(max(tau_val * p_home * p_away, eps))
    return weight * ll


# ─────────────────────────────────────────────────────────────────────────────
# Parameter packing/unpacking
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

    def pack(self) -> np.ndarray:
        """Pack parameters into a flat array for scipy."""
        n = len(self.teams)
        x = np.zeros(2 * n + 2)
        for i, t in enumerate(self.teams):
            x[i] = self.attack.get(t, 0.0)
            x[n + i] = self.defense.get(t, 0.0)
        x[2 * n] = self.home_advantage
        x[2 * n + 1] = self.rho
        return x

    def unpack(self, x: np.ndarray) -> "DCParams":
        n = len(self.teams)
        self.attack = {t: x[i] for i, t in enumerate(self.teams)}
        self.defense = {t: x[n + i] for i, t in enumerate(self.teams)}
        self.home_advantage = float(x[2 * n])
        self.rho = float(x[2 * n + 1])
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
# MLE Fitting
# ─────────────────────────────────────────────────────────────────────────────


def fit_dixon_coles(
    df: pd.DataFrame,
    weights: Optional[np.ndarray] = None,
    initial_params: Optional[DCParams] = None,
) -> DCParams:
    """
    Fit the Dixon-Coles model via MLE.

    Parameters
    ----------
    df : DataFrame with columns home_team, away_team, home_goals, away_goals
    weights : array of per-match importance weights (EWM decay)
    initial_params : warm-start from previous fit (optional)

    Returns
    -------
    DCParams with fitted attack, defense, home_advantage, rho
    """
    if df.empty:
        raise ValueError("Cannot fit model on empty DataFrame")

    teams = sorted(set(df["home_team"].tolist()) | set(df["away_team"].tolist()))
    n_teams = len(teams)
    team_idx = {t: i for i, t in enumerate(teams)}

    if weights is None:
        weights = np.ones(len(df))

    # Normalize weights
    weights = weights / weights.sum() * len(df)

    home_goals = df["home_goals"].values.astype(int)
    away_goals = df["away_goals"].values.astype(int)
    home_idx = np.array([team_idx[t] for t in df["home_team"]])
    away_idx = np.array([team_idx[t] for t in df["away_team"]])

    def neg_log_likelihood(x: np.ndarray) -> float:
        attack = x[:n_teams]
        defense = x[n_teams : 2 * n_teams]
        mu = x[2 * n_teams]       # home advantage
        rho = x[2 * n_teams + 1]  # DC correction

        ll = 0.0
        for k in range(len(df)):
            lam_h = np.exp(attack[home_idx[k]] - defense[away_idx[k]] + mu)
            lam_a = np.exp(attack[away_idx[k]] - defense[home_idx[k]])
            ll += _log_likelihood_single(
                home_goals[k], away_goals[k], lam_h, lam_a, rho, weights[k]
            )

        # Identifiability: penalize deviation of mean attack from 0
        penalty = 1000.0 * (np.mean(attack) ** 2)
        return -(ll - penalty)

    # Initial guess
    if initial_params is not None:
        x0 = initial_params.pack()
    else:
        x0 = np.zeros(2 * n_teams + 2)
        x0[2 * n_teams] = 0.3    # home advantage
        x0[2 * n_teams + 1] = -0.1  # rho

    # Bounds: rho in (-1, 0.2), home_advantage > 0
    bounds = (
        [(None, None)] * n_teams          # attack: unbounded
        + [(None, None)] * n_teams        # defense: unbounded
        + [(0.0, 1.0)]                    # home advantage
        + [(-0.99, 0.2)]                  # rho
    )

    logger.info("Fitting Dixon-Coles on %d matches, %d teams...", len(df), n_teams)
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
    params.unpack(result.x)
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

        # Need team IDs
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
