"""
api/routers/dashboard.py
Mounts the HTML dashboard at GET /dashboard
"""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

_template = Path(__file__).resolve().parent.parent.parent / "dashboard" / "templates" / "index.html"


@router.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    if _template.exists():
        return HTMLResponse(content=_template.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Dashboard not found. Check dashboard/templates/index.html</h1>", status_code=404)
