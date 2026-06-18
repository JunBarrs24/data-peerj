#!/usr/bin/env python3
"""Compute retrieval metrics (recall@k, precision@k, MRR) for ValladolIA.

This is a *proxy* IR evaluation. The gold set per question is the set of articles
cited by the human judges in the `referencias_<judge>` columns of the consolidated
dataset, parsed into normalized (group_id, clave) pairs and unioned across judges.
Judge-cited articles are a proxy for relevance, not a curated retrieval benchmark.

The retrieved set per question is the service `referencias` captured by the
collector (`run_validation.py`, `service.referencias`), ranked by retrieval
`score` (descending), because the service reorders references by order of
appearance in the answer, not by score.

Caveats (also emitted in the report `caveats` block):
- Gold is judge-cited articles, a proxy, not a labeled relevance set.
- Article-clave recall applies to `articulos` / `infracciones` questions;
  `definiciones` questions route to definition units (no article clave), so cases
  whose judge references yield no parseable article clave are excluded and counted.
- Reform and base regulation share a group_id; normalization is on (group_id,
  clave), not on the source file, so a transit article matches whether it was
  retrieved from the base PDF or the 2024 reform PDF.
- Free-text reference parsing is lossy. Reference cells that name a regulation but
  no article (page-only citations) contribute no gold clave.
- A judge cell that cites an article without naming a regulation yields an
  unknown-group gold pair, matched as a wildcard against any group with the same
  clave; the count of such pairs is reported.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

VALIDATOR_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    VALIDATOR_ROOT / "cases_to_validate" / "respuestas_jueces_consolidadas_semantico.csv"
)
DEFAULT_RESPONSES = VALIDATOR_ROOT / "output" / "latest_llm_responses.json"
DEFAULT_OUTPUT_DIR = VALIDATOR_ROOT / "output"
DEFAULT_K_VALUES = [1, 3, 5, 10]

# Source-file -> normative group_id (reform and base transit share a group_id).
ARCHIVO_TO_GROUP = {
    "Reglamento-de-Transito-y-Vialidad-Mpal.pdf": "transito-morelia",
    "reforma_transito_2024.pdf": "transito-morelia",
    "Reglamento-De-Orden-y-Justicia-Civica.pdf": "orden-justicia-civica-morelia",
}

# Regulation keyword -> group_id, applied to judge free-text reference cells.
# Order matters: more specific markers first.
REG_MARKERS = [
    ("orden y justicia", "orden-justicia-civica-morelia"),
    ("justicia civica", "orden-justicia-civica-morelia"),
    ("rojc", "orden-justicia-civica-morelia"),
    ("civica", "orden-justicia-civica-morelia"),
    ("orden", "orden-justicia-civica-morelia"),
    ("transito", "transito-morelia"),
    ("vialidad", "transito-morelia"),
    ("rtvm", "transito-morelia"),
    ("rtv", "transito-morelia"),
]

# "articulo 20", "Artículo 87.", "Art. 5", "Art 19", "art. 106"
ARTICLE_RE = re.compile(r"\bart(?:[íi]culos?|s?\.?)?\s*0*(\d{1,3})", re.IGNORECASE)


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def _norm_clave(num: int) -> str:
    return f"Artículo {num}."


def _archivo_to_group(archivo: str) -> str | None:
    if archivo in ARCHIVO_TO_GROUP:
        return ARCHIVO_TO_GROUP[archivo]
    low = _strip_accents(archivo).lower()
    if "transito" in low or "vialidad" in low:
        return "transito-morelia"
    if "orden" in low or "civica" in low or "justicia" in low:
        return "orden-justicia-civica-morelia"
    return None


def _parse_gold_from_cell(cell: str) -> list[tuple[str | None, str]]:
    """Parse one judge reference cell into (group_id|None, clave) pairs.

    Each article number is associated with the most recent regulation marker that
    appears before it in the text; if no marker precedes it, the first marker in
    the cell is used; if the cell names no regulation, group is None (wildcard).
    """
    if not cell.strip():
        return []
    low = _strip_accents(cell).lower()

    # marker positions: (start_index, group_id)
    markers: list[tuple[int, str]] = []
    for kw, gid in REG_MARKERS:
        for m in re.finditer(re.escape(kw), low):
            markers.append((m.start(), gid))
    markers.sort()

    first_group = markers[0][1] if markers else None

    pairs: list[tuple[str | None, str]] = []
    for am in ARTICLE_RE.finditer(low):
        num = int(am.group(1))
        pos = am.start()
        group: str | None = first_group
        for mpos, gid in markers:
            if mpos <= pos:
                group = gid
            else:
                break
        pairs.append((group, _norm_clave(num)))
    return pairs


def _load_gold(dataset_path: Path, limit: int | None) -> dict[str, dict[str, Any]]:
    """case_id -> {question, gold:set[(group,clave)], unknown_group:int}."""
    out: dict[str, dict[str, Any]] = {}
    with dataset_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Dataset has no headers: {dataset_path}")
        ref_cols = [h for h in reader.fieldnames if h.startswith("referencias_")]
        kept = 0
        for idx, row in enumerate(reader, start=1):
            q = (row.get("Pregunta") or "").strip()
            if not q:
                continue
            kept += 1
            case_id = f"case_{idx:04d}"
            gold: set[tuple[str | None, str]] = set()
            for col in ref_cols:
                for pair in _parse_gold_from_cell(row.get(col) or ""):
                    gold.add(pair)
            unknown = sum(1 for g, _ in gold if g is None)
            out[case_id] = {"question": q, "gold": gold, "unknown_group": unknown}
            if limit is not None and kept >= limit:
                break
    return out


def _load_retrieved(responses_path: Path) -> dict[str, list[tuple[str | None, str | None, float]]]:
    """case_id -> ranked list of (group_id, clave|None, score), sorted by score desc.

    clave is None for non-article referencias (definicion/infraccion rows without
    an article clave); they still occupy a rank slot.
    """
    payload = json.load(responses_path.open("r", encoding="utf-8"))
    out: dict[str, list[tuple[str | None, str | None, float]]] = {}
    for case in payload.get("cases", []):
        cid = str(case.get("case_id"))
        refs = case.get("service", {}).get("referencias", []) or []
        ranked: list[tuple[str | None, str | None, float]] = []
        for r in refs:
            if not isinstance(r, dict):
                continue
            try:
                score = float(r.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            group = _archivo_to_group(str(r.get("archivo", "")))
            art = str(r.get("articulo", ""))
            m = re.search(r"Art[íi]culo\s*0*(\d{1,3})", art)
            clave = _norm_clave(int(m.group(1))) if m else None
            ranked.append((group, clave, score))
        ranked.sort(key=lambda x: x[2], reverse=True)
        out[cid] = ranked
    return out


def _matches(gold: set[tuple[str | None, str]], group: str | None, clave: str) -> bool:
    """A retrieved (group, clave) hits gold if clave matches and groups agree, or
    the gold pair has an unknown group (wildcard)."""
    if (group, clave) in gold:
        return True
    return (None, clave) in gold


def _hit_set_at_k(
    ranked: list[tuple[str | None, str | None, float]],
    gold: set[tuple[str | None, str]],
    k: int,
) -> set[str]:
    """Distinct gold claves hit within the top-k ranked referencias."""
    hits: set[str] = set()
    for group, clave, _ in ranked[:k]:
        if clave is not None and _matches(gold, group, clave):
            hits.add(clave)
    return hits


def _gold_claves(gold: set[tuple[str | None, str]]) -> set[str]:
    return {c for _, c in gold}


def _mrr(
    ranked: list[tuple[str | None, str | None, float]],
    gold: set[tuple[str | None, str]],
) -> float:
    for rank, (group, clave, _) in enumerate(ranked, start=1):
        if clave is not None and _matches(gold, group, clave):
            return 1.0 / rank
    return 0.0


def compute(
    dataset_path: Path,
    responses_path: Path,
    k_values: list[int],
    limit: int | None,
) -> dict[str, Any]:
    gold_by_case = _load_gold(dataset_path, limit)
    retrieved_by_case = _load_retrieved(responses_path)

    n_total = len(gold_by_case)
    included: list[str] = []
    excluded_no_gold: list[str] = []
    zero_article_retrieval: list[str] = []
    per_case: list[dict[str, Any]] = []
    unknown_group_pairs = 0

    recall_sums = {k: 0.0 for k in k_values}
    precision_sums = {k: 0.0 for k in k_values}
    mrr_sum = 0.0

    for cid, ginfo in gold_by_case.items():
        gold = ginfo["gold"]
        gold_claves = _gold_claves(gold)
        if not gold_claves:
            excluded_no_gold.append(cid)
            continue
        included.append(cid)
        unknown_group_pairs += ginfo["unknown_group"]
        ranked = retrieved_by_case.get(cid, [])
        n_article_retrieved = sum(1 for _, c, _ in ranked if c is not None)
        if n_article_retrieved == 0:
            zero_article_retrieval.append(cid)

        case_recall: dict[int, float] = {}
        case_precision: dict[int, float] = {}
        for k in k_values:
            hits = _hit_set_at_k(ranked, gold, k)
            case_recall[k] = len(hits) / len(gold_claves)
            case_precision[k] = len(hits) / k
            recall_sums[k] += case_recall[k]
            precision_sums[k] += case_precision[k]
        rr = _mrr(ranked, gold)
        mrr_sum += rr

        per_case.append(
            {
                "case_id": cid,
                "question": ginfo["question"],
                "gold_claves": sorted(gold_claves),
                "gold_pairs": sorted([f"{g}|{c}" for g, c in gold]),
                "n_retrieved": len(ranked),
                "n_article_retrieved": n_article_retrieved,
                "recall_at_k": case_recall,
                "precision_at_k": case_precision,
                "reciprocal_rank": rr,
            }
        )

    n_inc = len(included)

    def _avg(d: dict[int, float]) -> dict[str, float]:
        return {str(k): round(d[k] / n_inc, 4) if n_inc else 0.0 for k in k_values}

    summary_retrieval = {
        "metric_label": "retrieval recall against judge-cited articles (proxy, llama3-free)",
        "k_values": k_values,
        "n_cases_total": n_total,
        "n_cases_with_gold": n_inc,
        "n_cases_excluded_no_parseable_gold": len(excluded_no_gold),
        "excluded_case_ids": excluded_no_gold,
        "gold_parse_coverage": round(n_inc / n_total, 4) if n_total else 0.0,
        "n_included_with_zero_article_retrieval": len(zero_article_retrieval),
        "zero_article_retrieval_case_ids": zero_article_retrieval,
        "unknown_group_gold_pairs": unknown_group_pairs,
        "recall_at_k": _avg(recall_sums),
        "precision_at_k": _avg(precision_sums),
        "mrr": round(mrr_sum / n_inc, 4) if n_inc else 0.0,
        "caveats": [
            "Gold = judge-cited articles parsed from referencias_<judge> (proxy for "
            "relevance, not a curated IR benchmark).",
            "Scope is articulos/infracciones cases; cases with no parseable article "
            "gold (incl. definiciones and page-only citations) are excluded and counted.",
            "Normalization is on (group_id, clave); reform and base transit share a " "group_id.",
            "Referencias are ranked by service score (descending) because the API "
            "reorders them by answer appearance.",
            "Unknown-group gold pairs (article cited without a named regulation) are "
            "matched as a wildcard against any group with the same clave.",
            "precision@k uses k as denominator including non-article retrieved slots, "
            "so it under-counts for infraction questions that also retrieve "
            "infraction-row units.",
        ],
    }
    return {"summary": {"retrieval": summary_retrieval}, "per_case": per_case}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute retrieval metrics for ValladolIA")
    p.add_argument("--dataset", default=str(DEFAULT_DATASET))
    p.add_argument("--responses-file", default=str(DEFAULT_RESPONSES))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "retrieval_metrics_latest.json"))
    p.add_argument("--k", default=",".join(str(k) for k in DEFAULT_K_VALUES))
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    k_values = [int(x) for x in str(args.k).split(",") if x.strip()]
    result = compute(Path(args.dataset), Path(args.responses_file), k_values, args.limit)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    r = result["summary"]["retrieval"]
    print(f"Retrieval metrics written to: {out_path}")
    print(
        f"cases total={r['n_cases_total']} with_gold={r['n_cases_with_gold']} "
        f"excluded_no_gold={r['n_cases_excluded_no_parseable_gold']} "
        f"zero_article_retrieval={r['n_included_with_zero_article_retrieval']}"
    )
    print(f"gold_parse_coverage={r['gold_parse_coverage']}")
    print(f"recall@k={r['recall_at_k']}")
    print(f"precision@k={r['precision_at_k']}")
    print(f"MRR={r['mrr']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
