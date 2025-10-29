from typing import Any, Dict
from urllib.parse import urlencode

def urlq(request, name: str, **query: Dict[str, Any]) -> str:
    """
    Build a URL by route name and optional query params:
    {{ urlq(request, routes.sim_select, mode='run') }}
    """
    base = request.url_for(name)
    return f"{base}?{urlencode(query, doseq=True)}" if query else str(base)
