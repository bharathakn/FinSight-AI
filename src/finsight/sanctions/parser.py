"""OFAC SDN (headerless CSV) + UN Consolidated List (XML) → one unified entity table.

Output schema (one row per name variant):
    entity_uid, source, source_id, primary_name, name, name_norm, name_kind,
    entity_type, program, nationality
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd
from lxml import etree

from finsight import config
from finsight.sanctions.text import normalise_name  # noqa: F401  (re-exported)

OFAC_COLUMNS = ["ent_num", "sdn_name", "sdn_type", "program", "title", "call_sign",
                "vess_type", "tonnage", "grt", "vess_flag", "vess_owner", "remarks"]
OFAC_NULL = "-0-"
_AKA = re.compile(r"a\.k\.a\.\s*'([^']+)'", re.IGNORECASE)
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")


def _ofac_display_name(raw: str, sdn_type: str) -> str:
    """OFAC writes individuals as 'LAST, First Middle' → 'First Middle LAST'."""
    if sdn_type == "individual" and "," in raw:
        last, first = raw.split(",", 1)
        return f"{first.strip()} {last.strip()}"
    return raw.strip()


def parse_ofac(path: Path | None = None) -> pd.DataFrame:
    path = Path(path or config.RAW_DIR / "sanctions" / "ofac" / "ofac_sdn.csv")
    df = pd.read_csv(path, header=None, names=OFAC_COLUMNS, dtype=str,
                     skipinitialspace=True, keep_default_na=False, on_bad_lines="warn")
    df = df.apply(lambda c: c.str.strip())
    df = df.replace({OFAC_NULL: None, "": None})
    df = df[df["ent_num"].notna() & df["sdn_name"].notna()]
    df["entity_type"] = df["sdn_type"].fillna("entity")

    rows = []
    for r in df.itertuples(index=False):
        primary = _ofac_display_name(r.sdn_name, r.entity_type)
        base = dict(entity_uid=f"OFAC-{r.ent_num}", source="OFAC", source_id=r.ent_num,
                    primary_name=primary, entity_type=r.entity_type, program=r.program, nationality=None)
        rows.append({**base, "name": primary, "name_kind": "primary"})
        for alias in _AKA.findall(r.remarks or ""):
            rows.append({**base, "name": alias.strip(), "name_kind": "alias"})
    return pd.DataFrame(rows)


def _text(el, tag: str) -> str | None:
    found = el.find(tag)
    if found is None or found.text is None:
        return None
    t = found.text.strip()
    return t or None


def parse_un(path: Path | None = None) -> pd.DataFrame:
    path = Path(path or config.RAW_DIR / "sanctions" / "un" / "un_consolidated_sanctions.xml")
    root = etree.parse(str(path)).getroot()
    rows = []
    for kind, alias_tag in (("INDIVIDUAL", "INDIVIDUAL_ALIAS"), ("ENTITY", "ENTITY_ALIAS")):
        for el in root.iter(kind):
            parts = [_text(el, t) for t in ("FIRST_NAME", "SECOND_NAME", "THIRD_NAME", "FOURTH_NAME")]
            primary = " ".join(p for p in parts if p)
            if not primary:
                continue
            nat = el.find("NATIONALITY/VALUE")
            base = dict(entity_uid=f"UN-{_text(el, 'DATAID')}", source="UN", source_id=_text(el, "DATAID"),
                        primary_name=primary, entity_type=kind.lower(), program=_text(el, "UN_LIST_TYPE"),
                        nationality=(nat.text.strip() if nat is not None and nat.text else None))
            rows.append({**base, "name": primary, "name_kind": "primary"})
            for a in el.iter(alias_tag):
                alias = _text(a, "ALIAS_NAME")
                if alias:
                    rows.append({**base, "name": alias, "name_kind": "alias"})
            original = _text(el, "NAME_ORIGINAL_SCRIPT")
            if original:
                rows.append({**base, "name": original, "name_kind": "original_script"})
    return pd.DataFrame(rows)


def build_entities() -> pd.DataFrame:
    ents = pd.concat([parse_ofac(), parse_un()], ignore_index=True)
    ents["name_norm"] = ents["name"].map(normalise_name)
    ents = ents[ents["name_norm"].str.len() > 1]
    ents = ents.drop_duplicates(subset=["entity_uid", "name_norm"]).reset_index(drop=True)
    return ents
