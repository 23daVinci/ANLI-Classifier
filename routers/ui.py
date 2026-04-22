from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["ui"])

_static = Path(__file__).parent.parent / "static"


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
def root():
    html_path = _static / "index.html"
    if html_path.exists():
        return html_path.read_text()
    return HTMLResponse(
        "<h3>ANLI R2 NLI Classifier</h3>"
        "<p>API is running. Visit <a href='/docs'>/docs</a> for Swagger UI.</p>"
    )


@router.get("/presentation", response_class=HTMLResponse, include_in_schema=False)
def presentation():
    html_path = _static / "presentation.html"
    if html_path.exists():
        return html_path.read_text()
    raise HTTPException(status_code=404, detail="Presentation not found")
