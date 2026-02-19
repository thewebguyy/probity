"""
dashboard/app.py
-----------------
Standalone FastAPI dashboard app with Jinja2 HTML templates.
Serves the HTML performance dashboard at GET /dashboard.

Can run standalone:
    uvicorn dashboard.app:app --port 8080

Or mount into main API at /dashboard.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

_here = Path(__file__).parent
_static = _here / "static"
_static.mkdir(exist_ok=True)

app = FastAPI(title="Probity Dashboard")
app.mount("/static", StaticFiles(directory=str(_static)), name="static")


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    html = (_here / "templates" / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)
