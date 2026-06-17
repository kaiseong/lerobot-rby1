from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def json_ready(value: Any) -> Any:
    """Convert common non-JSON scalar/container values into JSON-safe values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    if hasattr(value, "tolist"):
        return json_ready(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    return value


def emit_report(report: dict[str, Any], report_path: str | Path | None = None) -> dict[str, Any]:
    payload = json_ready(report)
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if report_path:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    return payload
