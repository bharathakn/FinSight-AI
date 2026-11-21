"""Name normalisation shared by the parser (offline) and the screener (serving); no lxml import."""
from __future__ import annotations

import re
import unicodedata

_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")


def normalise_name(name: str) -> str:
    """Casefold, strip accents/punctuation, collapse whitespace: 'Müller-Ödön, J.' → 'muller odon j'."""
    if not isinstance(name, str):
        return ""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = _NON_ALNUM.sub(" ", s.lower())
    return _WS.sub(" ", s).strip()
