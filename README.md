# Probity — Probability Pricing Engine & Market Mispricing Detector

A statistically rigorous system for estimating true outcome probabilities for Eredivisie matches,
comparing those prices to bookmaker odds, and detecting +EV opportunities.

## Architecture

```
probity/
├── core/               # Shared DB models, config, utilities
├── ingestion/          # Historical data fetcher + live odds scraper
├── features/           # Feature engineering layer
├── model/              # Dixon-Coles probabilistic model (MLE)
├── market/             # Market comparison & edge detection
├── evaluation/         # ROI, CLV, Kelly, Monte Carlo
├── api/                # FastAPI REST API
├── dashboard/          # HTML dashboard (Jinja2)
└── scripts/            # CLI runner scripts
```

## Services

| Service | Purpose | Schedule |
|---|---|---|
| `ingestion` | Pulls historical data, polls live odds | Every 90 sec |
| `model` | Nightly MLE re-fit, precomputes fair odds | Daily 02:00 |
| `edge_scanner` | Compares live vs model odds, logs triggers | Every 90 sec |
| `api` | REST API serving probabilities + edges | Always on |

## Setup

### 1. Prerequisites
- Python 3.10+
- PostgreSQL 15+

### 2. Environment
```bash
cp .env.example .env
# Fill in DB credentials and API keys
```

### 3. Install
```bash
pip install -r requirements.txt
```

### 4. Initialize Database
```bash
python scripts/init_db.py
```

### 5. Ingest Historical Data (3 seasons)
```bash
python scripts/ingest_historical.py --leagues NL1 --seasons 3
```

### 6. Run Model Fit
```bash
python scripts/run_model.py
```

### 7. Start API
```bash
uvicorn api.main:app --reload --port 8000
```

### 8. Start Dashboard
```bash
python scripts/run_dashboard.py
```

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /matches/upcoming` | Upcoming fixtures |
| `GET /model/fair-odds/{match_id}` | Fair probabilities + fair odds |
| `GET /edges/live` | Detected +EV opportunities |
| `GET /performance/metrics` | ROI, CLV, drawdown stats |

## Model: Dixon-Coles

- **Attack_i / Defense_i** per team (MLE via scipy.optimize)
- **Home advantage** parameter
- **rho** correction for low-scoring outcomes (0-0, 1-0, 0-1, 1-1)
- Rolling 3-year calibration window
- Exponentially weighted xG features (λ tuned via cross-validation)

## Edge Detection Formula

```
Edge = (Model_Prob × Book_Odds) − 1
Trigger: Edge > 2.5%
```

Margin removal: **Power method**

## Risk

- 0.25 fractional Kelly staking
- Max drawdown tracking
- Monte Carlo: 10,000 seasons × 500 bets
