"""
scripts/run_api.py
-------------------
Convenience launcher for the FastAPI API.

Usage:
    python scripts/run_api.py
    python scripts/run_api.py --port 8080 --reload
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    import uvicorn
    from core.config import settings

    uvicorn.run(
        "api.main:app",
        host=args.host or settings.API_HOST,
        port=args.port or settings.API_PORT,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
