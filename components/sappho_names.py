import json
import random
from pathlib import Path
from typing import Any

_HERE = Path(__file__).parent
DB_PATH = _HERE.parent / "data" / "sappho_names.json"
_FALLBACK_NAMES = ["Sappho", "Atthis", "Anactoria", "Kleis", "Aphrodite"]


def _load_names(path: Path = DB_PATH) -> list[str]:
    if not path.exists():
        return _FALLBACK_NAMES

    with open(path, "r", encoding="utf-8") as file:
        payload: Any = json.load(file)

    names: list[str] = []
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                name = str(item.get("name", "")).strip()
                if name:
                    names.append(name)
            elif isinstance(item, str):
                name = item.strip()
                if name:
                    names.append(name)

    return names or _FALLBACK_NAMES


def get_random_sappho_name() -> str:
    return random.choice(_load_names())
