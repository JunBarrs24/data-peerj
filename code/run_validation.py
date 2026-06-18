#!/usr/bin/env python3
"""Run regression validation against judge references using local Ollama."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import HTTPCookieProcessor, Request, build_opener, urlopen

PROMPT_VERSION = "v1.0"
VALIDATOR_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    VALIDATOR_ROOT / "cases_to_validate" / "respuestas_jueces_consolidadas_semantico.csv"
)
DEFAULT_THRESHOLDS = VALIDATOR_ROOT / "scripts" / "thresholds.json"
DEFAULT_BASELINE = VALIDATOR_ROOT / "output" / "baseline_report.json"
DEFAULT_OUTPUT_DIR = VALIDATOR_ROOT / "output"


@dataclass
class Case:
    case_id: str
    question: str
    judge_answers: list[dict[str, str]]
    status_label: str


@dataclass
class ServiceCallResult:
    ok: bool
    latency_ms: float
    error: str | None
    status: str
    candidate_answer: str
    clarification: dict[str, Any] | None
    referencias: list[dict[str, Any]]


def _now_hms() -> str:
    return datetime.now(UTC).strftime("%H:%M:%S")


def _log(msg: str, *, enabled: bool = True) -> None:
    if enabled:
        print(f"[{_now_hms()}] {msg}", flush=True)


def _post_json(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=timeout_s) as resp:  # nosec B310 - controlled URLs from CLI
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Session-aware opener: acquires session_id cookie from the homepage
# ---------------------------------------------------------------------------

_session_opener: Any = None


def _get_session_opener(service_url: str, timeout_s: float = 10.0) -> Any:
    """Return a urllib opener that carries the session cookie from the service."""
    global _session_opener
    if _session_opener is not None:
        return _session_opener
    jar = CookieJar()
    opener = build_opener(HTTPCookieProcessor(jar))
    # GET the homepage to acquire the session_id cookie
    base_url = service_url.rsplit("/api/", 1)[0]
    opener.open(base_url + "/", timeout=timeout_s)
    _session_opener = opener
    return opener


def _post_json_with_session(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    """POST JSON using the session-aware opener (carries session_id cookie)."""
    opener = _get_session_opener(url, timeout_s)
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with opener.open(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def _load_thresholds(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_cases(dataset_path: Path, limit: int | None = None) -> list[Case]:
    cases: list[Case] = []
    with dataset_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Dataset has no headers: {dataset_path}")

        response_cols = [
            h
            for h in reader.fieldnames
            if h.startswith("respuesta_") and h != "respuesta_consolidada"
        ]

        for idx, row in enumerate(reader, start=1):
            q = (row.get("Pregunta") or "").strip()
            if not q:
                continue

            judge_answers: list[dict[str, str]] = []
            for col in response_cols:
                answer = (row.get(col) or "").strip()
                if not answer:
                    continue
                judge = col.replace("respuesta_", "").strip() or "judge"
                refs = (row.get(f"referencias_{judge}") or "").strip()
                judge_answers.append({"judge": judge, "answer": answer, "references": refs})

            consolidated = (row.get("respuesta_consolidada") or "").strip()
            if consolidated:
                judge_answers.append(
                    {
                        "judge": "consolidada",
                        "answer": consolidated,
                        "references": "",
                    }
                )

            if not judge_answers:
                continue

            case = Case(
                case_id=f"case_{idx:04d}",
                question=q,
                judge_answers=judge_answers,
                status_label=(row.get("estado_respuesta") or "").strip(),
            )
            cases.append(case)
            if limit is not None and len(cases) >= limit:
                break

    if not cases:
        raise ValueError(f"No valid cases loaded from: {dataset_path}")
    return cases


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("Empty judge response")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    chunk = text[start : i + 1]
                    try:
                        return json.loads(chunk)
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)

    raise ValueError(f"Could not parse JSON from judge output: {text[:200]}")


def _judge_system_prompt() -> str:
    return (
        "You are a strict legal-answer evaluator. Compare candidate answer to reference answers. "
        "CRITICAL: If the candidate refuses to answer, says it lacks information, or gives a generic "
        "non-answer while the references contain a clear answer, that is a FAILURE — score it as "
        "verdict 'worse' with overall <= 1.0, accuracy 0, coverage 0, consistency 0. "
        "A non-answer is NEVER 'better' or 'similar' to an actual answer. "
        "Output JSON only, no markdown, no extra text."
    )


def _judge_user_prompt(question: str, references: list[str], candidate: str) -> str:
    refs = "\n\n".join([f"Reference {i + 1}: {ref}" for i, ref in enumerate(references)])
    return (
        "Evaluate the candidate answer against references for legal correctness and completeness.\n"
        "Scoring scale for numeric fields is 0..10.\n"
        "- accuracy: factual/legal correctness\n"
        "- coverage: key points included\n"
        "- consistency: agreement with references\n"
        "- contradiction_risk: risk of contradiction (0 best, 10 worst)\n"
        "- hallucination_risk: unsupported claims risk (0 best, 10 worst)\n"
        "- overall: weighted overall quality\n"
        "verdict must be one of: better, similar, worse\n"
        "reason fields must be concise arrays of strings.\n\n"
        "REFUSAL RULE: If the candidate says it cannot answer, does not have enough information, "
        "or provides only a generic deflection — while the references clearly contain an answer — "
        "you MUST score: accuracy=0, coverage=0, consistency=0, overall<=1, verdict='worse'. "
        "Not answering a question that HAS an answer is the worst possible outcome.\n\n"
        "Return exactly this JSON object keys:\n"
        "accuracy, coverage, consistency, contradiction_risk, hallucination_risk, overall, verdict, "
        "reasons, missing_points, incorrect_claims, matched_reference_points, confidence\n\n"
        f"Question:\n{question}\n\n"
        f"References:\n{refs}\n\n"
        f"Candidate:\n{candidate}\n"
    )


def _clamp_score(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(10.0, x))


def _normalize_eval(raw: dict[str, Any]) -> dict[str, Any]:
    verdict = str(raw.get("verdict", "worse")).strip().lower()
    if verdict not in {"better", "similar", "worse"}:
        verdict = "worse"

    conf = raw.get("confidence", 0.0)
    try:
        conf_f = max(0.0, min(1.0, float(conf)))
    except (TypeError, ValueError):
        conf_f = 0.0

    def _to_list(key: str) -> list[str]:
        value = raw.get(key)
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    return {
        "accuracy": _clamp_score(raw.get("accuracy")),
        "coverage": _clamp_score(raw.get("coverage")),
        "consistency": _clamp_score(raw.get("consistency")),
        "contradiction_risk": _clamp_score(raw.get("contradiction_risk")),
        "hallucination_risk": _clamp_score(raw.get("hallucination_risk")),
        "overall": _clamp_score(raw.get("overall")),
        "verdict": verdict,
        "reasons": _to_list("reasons"),
        "missing_points": _to_list("missing_points"),
        "incorrect_claims": _to_list("incorrect_claims"),
        "matched_reference_points": _to_list("matched_reference_points"),
        "confidence": conf_f,
    }


def _ollama_judge(
    judge_url: str,
    model: str,
    question: str,
    references: list[str],
    candidate: str,
    timeout_s: float,
) -> tuple[dict[str, Any], str]:
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": _judge_system_prompt()},
            {"role": "user", "content": _judge_user_prompt(question, references, candidate)},
        ],
        "options": {"temperature": 0},
    }
    raw_resp = _post_json(judge_url, payload, timeout_s=timeout_s)
    content = raw_resp.get("message", {}).get("content", "") if isinstance(raw_resp, dict) else ""
    parsed = _extract_json_object(str(content))
    return _normalize_eval(parsed), str(content)


def _call_service(
    service_url: str, question: str, timeout_s: float
) -> tuple[dict[str, Any], float, str | None]:
    start = time.monotonic()
    try:
        out = _post_json_with_session(service_url, {"question": question}, timeout_s=timeout_s)
        elapsed_ms = (time.monotonic() - start) * 1000.0
        return out, elapsed_ms, None
    except HTTPError as e:
        elapsed_ms = (time.monotonic() - start) * 1000.0
        try:
            body = e.read().decode("utf-8", errors="ignore")
        except Exception:
            body = ""
        return {}, elapsed_ms, f"HTTP {e.code}: {body[:500]}"
    except URLError as e:
        elapsed_ms = (time.monotonic() - start) * 1000.0
        return {}, elapsed_ms, f"URL error: {e.reason}"
    except Exception as e:  # noqa: BLE001
        elapsed_ms = (time.monotonic() - start) * 1000.0
        return {}, elapsed_ms, f"Unhandled error: {e}"


def _service_result_from_raw(
    service_raw: dict[str, Any],
    latency_ms: float,
    service_error: str | None,
) -> ServiceCallResult:
    candidate = str(service_raw.get("respuesta", "")).strip() if service_raw else ""
    clarification = service_raw.get("clarification") if isinstance(service_raw, dict) else None
    raw_refs = service_raw.get("referencias") if isinstance(service_raw, dict) else None
    referencias = [r for r in raw_refs if isinstance(r, dict)] if isinstance(raw_refs, list) else []
    return ServiceCallResult(
        ok=service_error is None,
        latency_ms=round(latency_ms, 2),
        error=service_error,
        status="ok" if service_error is None else "error",
        candidate_answer=candidate,
        clarification=clarification if isinstance(clarification, dict) else None,
        referencias=referencias,
    )


def _collect_one_response(
    case: Case,
    service_url: str,
    timeout_s: float,
    verbose: bool,
    total: int,
) -> tuple[str, ServiceCallResult, dict[str, Any]]:
    """Collect a single service response (called in parallel)."""
    _log(
        f"{case.case_id} sending question to service: {case.question[:90]}",
        enabled=verbose,
    )
    service_raw, latency_ms, service_error = _call_service(service_url, case.question, timeout_s)
    result = _service_result_from_raw(service_raw, latency_ms, service_error)
    if result.error:
        _log(f"{case.case_id} service error: {result.error}", enabled=verbose)
    else:
        _log(
            f"{case.case_id} service response received in {result.latency_ms:.0f}ms | chars={len(result.candidate_answer)}",
            enabled=verbose,
        )
    case_data = {
        "case_id": case.case_id,
        "question": case.question,
        "status_label": case.status_label,
        "service": {
            "ok": result.ok,
            "latency_ms": result.latency_ms,
            "error": result.error,
            "status": result.status,
            "candidate_answer": result.candidate_answer,
            "clarification": result.clarification,
            "referencias": result.referencias,
        },
    }
    return case.case_id, result, case_data


def _collect_llm_responses(
    cases: list[Case],
    service_url: str,
    output_dir: Path,
    timeout_s: float,
    date_str: str,
    run_id: str,
    verbose: bool,
    workers: int = 4,
) -> tuple[dict[str, ServiceCallResult], Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    responses_path = output_dir / f"llm_responses_{date_str}.json"
    latest_path = output_dir / "latest_llm_responses.json"

    service_map: dict[str, ServiceCallResult] = {}
    case_data_map: dict[str, dict[str, Any]] = {}

    _log(
        f"Stage 1/2 collecting service responses for {len(cases)} cases (workers={workers})",
        enabled=verbose,
    )

    done_count = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _collect_one_response, case, service_url, timeout_s, verbose, len(cases)
            ): case.case_id
            for case in cases
        }
        for future in as_completed(futures):
            case_id, result, case_data = future.result()
            service_map[case_id] = result
            case_data_map[case_id] = case_data
            done_count += 1
            _log(f"Stage 1 progress: {done_count}/{len(cases)}", enabled=verbose)

    # Reconstruct list in original case order
    response_cases = [case_data_map[case.case_id] for case in cases]

    payload = {
        "metadata": {
            "run_id": run_id,
            "date": date_str,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "service_url": service_url,
            "total_cases": len(cases),
        },
        "cases": response_cases,
    }
    _write_json(responses_path, payload)
    _write_json(latest_path, payload)
    _log(f"Saved LLM responses: {responses_path}", enabled=verbose)
    _log(f"Updated latest responses: {latest_path}", enabled=verbose)
    return service_map, responses_path


def _load_responses_map(
    path: Path, cases: list[Case]
) -> tuple[dict[str, ServiceCallResult], dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    raw_cases = payload.get("cases", [])
    by_id = {
        str(item.get("case_id")): item
        for item in raw_cases
        if isinstance(item, dict) and item.get("case_id")
    }
    out: dict[str, ServiceCallResult] = {}
    for case in cases:
        row = by_id.get(case.case_id, {})
        service = row.get("service", {}) if isinstance(row, dict) else {}
        if not isinstance(service, dict):
            service = {}
        out[case.case_id] = ServiceCallResult(
            ok=bool(service.get("ok", False)),
            latency_ms=float(service.get("latency_ms", 0.0) or 0.0),
            error=str(service.get("error")) if service.get("error") is not None else None,
            status=str(service.get("status", "error")),
            candidate_answer=str(service.get("candidate_answer", "")),
            clarification=service.get("clarification")
            if isinstance(service.get("clarification"), dict)
            else None,
            referencias=[r for r in service.get("referencias", []) if isinstance(r, dict)]
            if isinstance(service.get("referencias"), list)
            else [],
        )
    return out, payload.get("metadata", {})


def _resolve_latest_responses_path(output_dir: Path) -> Path | None:
    latest = output_dir / "latest_llm_responses.json"
    if latest.exists():
        return latest
    candidates = sorted(output_dir.glob("llm_responses_*.json"), reverse=True)
    return candidates[0] if candidates else None


def _verdict_counts(cases: list[dict[str, Any]]) -> dict[str, int]:
    out = {"better": 0, "similar": 0, "worse": 0}
    for c in cases:
        v = c["overall_eval"].get("verdict", "worse")
        if v not in out:
            v = "worse"
        out[v] += 1
    return out


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _score_stddev(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) >= 2 else 0.0


def _apply_regression(
    current_cases: list[dict[str, Any]],
    baseline_report: dict[str, Any] | None,
    thresholds: dict[str, Any],
) -> tuple[list[dict[str, Any]], float]:
    if not baseline_report:
        return current_cases, 0.0

    baseline_map: dict[str, dict[str, Any]] = {
        str(c.get("case_id")): c for c in baseline_report.get("cases", [])
    }

    deltas: list[float] = []
    max_drop = float(thresholds.get("max_score_drop_per_case", 1.0))

    for case in current_cases:
        cid = case["case_id"]
        reg = {
            "baseline_overall": None,
            "current_overall": case["overall_eval"].get("overall", 0.0),
            "delta_overall": None,
            "regression_flags": [],
        }
        b = baseline_map.get(cid)
        if b:
            baseline_overall = float(b.get("overall_eval", {}).get("overall", 0.0))
            current_overall = float(case["overall_eval"].get("overall", 0.0))
            delta = current_overall - baseline_overall
            deltas.append(delta)
            reg["baseline_overall"] = baseline_overall
            reg["delta_overall"] = delta

            if delta < -max_drop:
                reg["regression_flags"].append("score_drop")

            baseline_verdict = str(b.get("overall_eval", {}).get("verdict", "similar"))
            current_verdict = str(case["overall_eval"].get("verdict", "worse"))
            if baseline_verdict in {"better", "similar"} and current_verdict == "worse":
                reg["regression_flags"].append("verdict_regressed")

            base_contra = float(b.get("overall_eval", {}).get("contradiction_risk", 0.0))
            curr_contra = float(case["overall_eval"].get("contradiction_risk", 0.0))
            if base_contra <= float(
                thresholds.get("max_contradiction_risk", 4.0)
            ) and curr_contra > float(thresholds.get("max_contradiction_risk", 4.0)):
                reg["regression_flags"].append("new_contradiction")

            base_hall = float(b.get("overall_eval", {}).get("hallucination_risk", 0.0))
            curr_hall = float(case["overall_eval"].get("hallucination_risk", 0.0))
            if base_hall <= float(
                thresholds.get("max_hallucination_risk", 4.0)
            ) and curr_hall > float(thresholds.get("max_hallucination_risk", 4.0)):
                reg["regression_flags"].append("new_hallucination")

        case["regression"] = reg

    avg_delta = _mean(deltas)
    return current_cases, avg_delta


def _evaluate_case(
    case: Case,
    case_index: int,
    total_cases: int,
    judge_url: str,
    judge_model: str,
    timeout_s: float,
    evaluate_by_judge: bool,
    verbose: bool,
    service_result: ServiceCallResult,
) -> dict[str, Any]:
    _log(f"{case.case_id} ({case_index}/{total_cases}) validation phase started", enabled=verbose)
    candidate = service_result.candidate_answer
    if service_result.error:
        _log(
            f"{case.case_id} using response with service error: {service_result.error}",
            enabled=verbose,
        )
    else:
        _log(
            f"{case.case_id} using response latency={service_result.latency_ms:.0f}ms | chars={len(candidate)}",
            enabled=verbose,
        )

    service_data: dict[str, Any] = {
        "ok": service_result.ok,
        "latency_ms": service_result.latency_ms,
        "error": service_result.error,
        "status": service_result.status,
        "candidate_answer": candidate,
        "clarification": service_result.clarification,
        "referencias": service_result.referencias,
    }

    reference_texts = [j["answer"] for j in case.judge_answers if j["answer"]]
    overall_eval: dict[str, Any]
    overall_raw = ""
    by_judge: list[dict[str, Any]] = []

    if not candidate:
        _log(f"{case.case_id} empty candidate answer, marking as worse", enabled=verbose)
        overall_eval = {
            "accuracy": 0.0,
            "coverage": 0.0,
            "consistency": 0.0,
            "contradiction_risk": 10.0,
            "hallucination_risk": 10.0,
            "overall": 0.0,
            "verdict": "worse",
            "reasons": ["service_empty_answer"],
            "missing_points": [],
            "incorrect_claims": [],
            "matched_reference_points": [],
            "confidence": 1.0,
        }
    else:
        try:
            _log(f"{case.case_id} running overall judge with model={judge_model}", enabled=verbose)
            overall_eval, overall_raw = _ollama_judge(
                judge_url=judge_url,
                model=judge_model,
                question=case.question,
                references=reference_texts,
                candidate=candidate,
                timeout_s=timeout_s,
            )
            _log(
                f"{case.case_id} overall judged: score={overall_eval['overall']:.2f}, verdict={overall_eval['verdict']}",
                enabled=verbose,
            )
        except Exception as e:  # noqa: BLE001
            _log(f"{case.case_id} overall judge error: {e}", enabled=verbose)
            overall_eval = {
                "accuracy": 0.0,
                "coverage": 0.0,
                "consistency": 0.0,
                "contradiction_risk": 10.0,
                "hallucination_risk": 10.0,
                "overall": 0.0,
                "verdict": "worse",
                "reasons": [f"judge_error: {e}"],
                "missing_points": [],
                "incorrect_claims": [],
                "matched_reference_points": [],
                "confidence": 0.0,
            }

    if evaluate_by_judge and candidate:
        total_judges = len(case.judge_answers)
        for j_idx, ref in enumerate(case.judge_answers, start=1):
            try:
                _log(
                    f"{case.case_id} by-judge {j_idx}/{total_judges}: {ref['judge']}",
                    enabled=verbose,
                )
                single_eval, raw_text = _ollama_judge(
                    judge_url=judge_url,
                    model=judge_model,
                    question=case.question,
                    references=[ref["answer"]],
                    candidate=candidate,
                    timeout_s=timeout_s,
                )
                by_judge.append(
                    {
                        "judge": ref["judge"],
                        "eval": single_eval,
                        "raw_output": raw_text,
                        "reference": ref["answer"],
                    }
                )
            except Exception as e:  # noqa: BLE001
                _log(
                    f"{case.case_id} by-judge error ({ref['judge']}): {e}",
                    enabled=verbose,
                )
                by_judge.append(
                    {
                        "judge": ref["judge"],
                        "eval": {
                            "accuracy": 0.0,
                            "coverage": 0.0,
                            "consistency": 0.0,
                            "contradiction_risk": 10.0,
                            "hallucination_risk": 10.0,
                            "overall": 0.0,
                            "verdict": "worse",
                            "reasons": [f"judge_error: {e}"],
                            "missing_points": [],
                            "incorrect_claims": [],
                            "matched_reference_points": [],
                            "confidence": 0.0,
                        },
                        "raw_output": "",
                        "reference": ref["answer"],
                    }
                )

    score_values = [x["eval"]["overall"] for x in by_judge]
    by_judge_stddev = _score_stddev(score_values)
    _log(
        f"{case.case_id} done | overall={overall_eval['overall']:.2f} "
        f"verdict={overall_eval['verdict']} by_judge_sd={by_judge_stddev:.2f}",
        enabled=verbose,
    )

    return {
        "case_id": case.case_id,
        "question": case.question,
        "status_label": case.status_label,
        "reference_answers": case.judge_answers,
        "service": service_data,
        "overall_eval": overall_eval,
        "overall_raw_output": overall_raw,
        "by_judge": by_judge,
        "by_judge_overall_stddev": by_judge_stddev,
        "regression": {
            "baseline_overall": None,
            "current_overall": overall_eval.get("overall", 0.0),
            "delta_overall": None,
            "regression_flags": [],
        },
    }


def _load_baseline(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _gate_result(
    cases: list[dict[str, Any]],
    thresholds: dict[str, Any],
    avg_delta_from_baseline: float,
) -> tuple[bool, list[str], dict[str, Any], dict[str, int]]:
    overall_scores = [float(c["overall_eval"].get("overall", 0.0)) for c in cases]
    contradiction_risks = [float(c["overall_eval"].get("contradiction_risk", 10.0)) for c in cases]
    hallucination_risks = [float(c["overall_eval"].get("hallucination_risk", 10.0)) for c in cases]
    stddevs = [float(c.get("by_judge_overall_stddev", 0.0)) for c in cases]
    verdict_counts = _verdict_counts(cases)

    counts = {
        "total": len(cases),
        "service_errors": sum(1 for c in cases if not c["service"]["ok"]),
        "worse": verdict_counts["worse"],
        "similar": verdict_counts["similar"],
        "better": verdict_counts["better"],
        "critical_failures": 0,
        "regression_flagged_cases": sum(
            1 for c in cases if c.get("regression", {}).get("regression_flags")
        ),
    }

    averages = {
        "overall": _mean(overall_scores),
        "accuracy": _mean([float(c["overall_eval"].get("accuracy", 0.0)) for c in cases]),
        "coverage": _mean([float(c["overall_eval"].get("coverage", 0.0)) for c in cases]),
        "consistency": _mean([float(c["overall_eval"].get("consistency", 0.0)) for c in cases]),
        "contradiction_risk": _mean(contradiction_risks),
        "hallucination_risk": _mean(hallucination_risks),
        "by_judge_stddev": _mean(stddevs),
        "avg_delta_from_baseline": avg_delta_from_baseline,
    }

    critical_ids = set(str(x) for x in thresholds.get("critical_case_ids", []))
    for c in cases:
        if c["case_id"] in critical_ids and c["overall_eval"].get("verdict") == "worse":
            counts["critical_failures"] += 1

    fail_reasons: list[str] = []
    if averages["overall"] < float(thresholds.get("min_avg_overall", 0.0)):
        fail_reasons.append("avg_overall_below_threshold")

    min_case_overall = float(thresholds.get("min_case_overall", 0.0))
    low_case_count = sum(1 for s in overall_scores if s < min_case_overall)
    if low_case_count > 0:
        fail_reasons.append(f"{low_case_count}_cases_below_min_case_overall")

    max_worse = int(thresholds.get("max_worse_cases", 0))
    if counts["worse"] > max_worse:
        fail_reasons.append("too_many_worse_cases")

    max_contra = float(thresholds.get("max_contradiction_risk", 10.0))
    if any(x > max_contra for x in contradiction_risks):
        fail_reasons.append("case_contradiction_risk_too_high")

    max_hall = float(thresholds.get("max_hallucination_risk", 10.0))
    if any(x > max_hall for x in hallucination_risks):
        fail_reasons.append("case_hallucination_risk_too_high")

    max_stddev = float(thresholds.get("max_by_judge_stddev", math.inf))
    if any(x > max_stddev for x in stddevs):
        fail_reasons.append("by_judge_variance_too_high")

    if counts["critical_failures"] > int(thresholds.get("max_critical_failures", 0)):
        fail_reasons.append("critical_cases_failed")

    if averages["avg_delta_from_baseline"] < -float(
        thresholds.get("max_avg_drop_from_baseline", 0.0)
    ):
        fail_reasons.append("avg_drop_from_baseline_too_large")

    if any(c["regression"]["regression_flags"] for c in cases):
        fail_reasons.append("regression_flags_detected")

    return (len(fail_reasons) == 0), fail_reasons, averages, counts


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run LLM validation against Morelia judge dataset")
    p.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Path to CSV dataset",
    )
    p.add_argument(
        "--service-url",
        default="http://localhost:8000/api/question",
        help="Service endpoint that answers questions",
    )
    p.add_argument(
        "--judge-url",
        default="http://localhost:11434/api/chat",
        help="Ollama chat endpoint",
    )
    p.add_argument("--judge-model", default="llama3", help="Ollama model name")
    p.add_argument("--timeout", type=float, default=120.0, help="Timeout seconds per network call")
    p.add_argument("--limit", type=int, default=None, help="Optional number of cases to run")
    p.add_argument(
        "--thresholds",
        default=str(DEFAULT_THRESHOLDS),
        help="Threshold policy JSON file",
    )
    p.add_argument(
        "--baseline",
        default=str(DEFAULT_BASELINE),
        help="Baseline report for regression comparison",
    )
    p.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for report output",
    )
    p.add_argument(
        "--skip-by-judge",
        action="store_true",
        help="Skip per-judge breakdown and evaluate only overall",
    )
    p.add_argument(
        "--print-failures",
        action="store_true",
        help="Print per-case failure details in terminal",
    )
    p.add_argument(
        "--responses-file",
        default=None,
        help="Use an existing llm_responses JSON file instead of calling service",
    )
    p.add_argument(
        "--use-latest-responses",
        action="store_true",
        help="Use latest_llm_responses.json (or newest llm_responses_*.json) from output-dir",
    )
    p.add_argument(
        "--collect-only",
        action="store_true",
        help="Only collect service responses and save llm_responses_{date}.json (skip Ollama validation)",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Print only per-case summary lines (less verbose)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers for service calls and evaluation (default: 4)",
    )
    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    t0 = time.monotonic()
    now = datetime.now(UTC)
    date_str = now.strftime("%Y-%m-%d_%H%M%S")
    timestamp = now.isoformat()
    run_id = str(uuid.uuid4())

    dataset_path = Path(args.dataset)
    thresholds_path = Path(args.thresholds)
    output_dir = Path(args.output_dir)
    baseline_path = Path(args.baseline) if args.baseline else None

    thresholds = _load_thresholds(thresholds_path)
    _log(f"Loaded thresholds from {thresholds_path}", enabled=not args.quiet)
    cases = _load_cases(dataset_path, limit=args.limit)
    _log(f"Loaded {len(cases)} cases from {dataset_path}", enabled=not args.quiet)
    baseline = _load_baseline(baseline_path)
    if baseline:
        _log(f"Loaded baseline report from {baseline_path}", enabled=not args.quiet)
    else:
        _log("No baseline report found; regression deltas will be empty", enabled=not args.quiet)

    response_source = "fresh_service_calls"
    response_source_meta: dict[str, Any] = {}
    service_results_by_case: dict[str, ServiceCallResult] = {}
    if args.responses_file:
        responses_path = Path(args.responses_file)
        if not responses_path.exists():
            raise SystemExit(f"Responses file not found: {responses_path}")
        service_results_by_case, response_source_meta = _load_responses_map(responses_path, cases)
        response_source = f"responses_file:{responses_path}"
        _log(f"Using pre-collected responses from {responses_path}", enabled=not args.quiet)
    elif args.use_latest_responses:
        resolved = _resolve_latest_responses_path(output_dir)
        if resolved is None:
            raise SystemExit(
                f"No latest responses file found in {output_dir}. Run without --use-latest-responses first."
            )
        service_results_by_case, response_source_meta = _load_responses_map(resolved, cases)
        response_source = f"latest_responses:{resolved}"
        _log(f"Using latest pre-collected responses from {resolved}", enabled=not args.quiet)
    else:
        service_results_by_case, saved_responses_path = _collect_llm_responses(
            cases=cases,
            service_url=args.service_url,
            output_dir=output_dir,
            timeout_s=args.timeout,
            date_str=date_str,
            run_id=run_id,
            verbose=not args.quiet,
            workers=args.workers,
        )
        response_source_meta = {"responses_path": str(saved_responses_path)}
        _log("Stage 1/2 finished (service response collection)", enabled=not args.quiet)

    if args.collect_only:
        _log("Collect-only mode enabled; skipping Ollama validation stage", enabled=True)
        return 0

    _log(
        f"Stage 2/2 starting Ollama validation (workers={args.workers})",
        enabled=not args.quiet,
    )
    _missing = ServiceCallResult(
        ok=False,
        latency_ms=0.0,
        error="missing_response_for_case",
        status="error",
        candidate_answer="",
        clarification=None,
        referencias=[],
    )
    eval_results: dict[int, dict[str, Any]] = {}
    done_count = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _evaluate_case,
                case=case,
                case_index=idx,
                total_cases=len(cases),
                judge_url=args.judge_url,
                judge_model=args.judge_model,
                timeout_s=args.timeout,
                evaluate_by_judge=not args.skip_by_judge,
                verbose=not args.quiet,
                service_result=service_results_by_case.get(case.case_id, _missing),
            ): idx
            for idx, case in enumerate(cases, start=1)
        }
        for future in as_completed(futures):
            idx = futures[future]
            result = future.result()
            eval_results[idx] = result
            done_count += 1
            print(
                f"[{done_count}/{len(cases)}] {result['case_id']} "
                f"overall={result['overall_eval']['overall']:.2f} "
                f"verdict={result['overall_eval']['verdict']}"
            )

    # Reconstruct list in original case order
    evaluated: list[dict[str, Any]] = [eval_results[i] for i in range(1, len(cases) + 1)]

    evaluated, avg_delta = _apply_regression(evaluated, baseline, thresholds)
    passed, fail_reasons, averages, counts = _gate_result(evaluated, thresholds, avg_delta)

    runtime_s = round(time.monotonic() - t0, 2)

    report: dict[str, Any] = {
        "metadata": {
            "run_id": run_id,
            "timestamp_utc": timestamp,
            "date": date_str,
            "service_url": args.service_url,
            "dataset_path": str(dataset_path),
            "judge_model": args.judge_model,
            "judge_url": args.judge_url,
            "prompt_version": PROMPT_VERSION,
            "response_source": response_source,
            "response_source_metadata": response_source_meta,
            "thresholds_path": str(thresholds_path),
            "baseline_path": str(baseline_path) if baseline_path else None,
            "total_cases": len(cases),
            "evaluated_cases": len(evaluated),
            "runtime_seconds": runtime_s,
        },
        "summary": {
            "pass": passed,
            "fail_reasons": fail_reasons,
            "counts": counts,
            "averages": averages,
        },
        "cases": evaluated,
    }

    report_path = output_dir / f"report_test_{date_str}.json"
    latest_path = output_dir / "latest_report.json"
    _write_json(report_path, report)
    _write_json(latest_path, report)

    print(f"Report: {report_path}")
    print(f"Latest: {latest_path}")
    print(f"PASS: {passed}")
    if fail_reasons:
        print("Fail reasons:")
        for reason in fail_reasons:
            print(f" - {reason}")

    if args.print_failures:
        for c in evaluated:
            if c["overall_eval"]["verdict"] == "worse" or c["regression"]["regression_flags"]:
                print(f"\n{c['case_id']} | verdict={c['overall_eval']['verdict']}")
                print(f"Question: {c['question']}")
                print(f"Reasons: {c['overall_eval'].get('reasons', [])}")
                print(f"Regression: {c['regression']['regression_flags']}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
