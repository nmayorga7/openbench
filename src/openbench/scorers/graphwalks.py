# src/openbench/scorers/graphwalks.py
from __future__ import annotations

import re
from typing import Set, Iterable, Tuple

from inspect_ai.scorer import (
    scorer,
    Score,
    Target,
    mean,
    stderr,
    Metric,
    Value,
    SampleScore,
    metric,
)
from openbench.utils.text import get_token_count

# Match "Final Answer: [a, b, c]" at the end of output
_FINAL_LINE_RE = re.compile(r"Final Answer:\s*\[(.*)\]\s*$", re.IGNORECASE)


def _parse_nodes(text: str) -> Tuple[list[str], bool]:
    """Return (nodes, parse_error_flag). Dedup while preserving order."""
    if not text:
        return [], True
    last_line = text.strip().splitlines()[-1]
    m = _FINAL_LINE_RE.search(last_line)
    if not m:
        return [], True
    inner = m.group(1)
    raw = [t.strip() for t in inner.split(",")]
    seen: Set[str] = set()
    out: list[str] = []
    for t in raw:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out, False


def _prf1(pred: Iterable[str], gold: Iterable[str]) -> Tuple[float, float, float]:
    sp, sg = set(pred), set(gold)
    inter = len(sp & sg)
    p = inter / len(sp) if sp else 0.0
    r = inter / len(sg) if sg else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


GRAPHWALKS_BINS = [
    (0, 512),
    (512, 1024),
    (1024, 2048),
    (2048, 4096),
    (4096, 8192),
    (8192, 16384),
    (16384, 32768),
    (32768, 65536),
]


@metric
def graphwalks_metrics() -> Metric:
    """Report mean precision / recall / f1 per token bin."""
    def metric_calculator(scores: list[SampleScore]) -> Value:
        f1_by_bin = {f"{L}-{R}": 0.0 for (L, R) in GRAPHWALKS_BINS}
        p_by_bin = {f"{L}-{R}": 0.0 for (L, R) in GRAPHWALKS_BINS}
        r_by_bin = {f"{L}-{R}": 0.0 for (L, R) in GRAPHWALKS_BINS}
        counts = {f"{L}-{R}": 0 for (L, R) in GRAPHWALKS_BINS}

        for s in scores:
            md = s.score.metadata or {}
            tok = int(md.get("raw_input_tok_cnt", 0))
            p = float(md.get("precision", 0.0))
            r = float(md.get("recall", 0.0))
            f1 = float(md.get("f1", 0.0))

            for i, (L, R) in enumerate(GRAPHWALKS_BINS):
                if (i == 0 and L <= tok <= R) or (i > 0 and L < tok <= R):
                    k = f"{L}-{R}"
                    f1_by_bin[k] += f1
                    p_by_bin[k] += p
                    r_by_bin[k] += r
                    counts[k] += 1
                    break

        # average per bin
        for k in f1_by_bin.keys():
            if counts[k] > 0:
                f1_by_bin[k] /= counts[k]
                p_by_bin[k] /= counts[k]
                r_by_bin[k] /= counts[k]

        return {
            "f1_by_token_count": f1_by_bin,
            "precision_by_token_count": p_by_bin,
            "recall_by_token_count": r_by_bin,
            "samples_per_bin": counts,
        }

    return metric_calculator


@scorer(metrics=[mean(), stderr(), graphwalks_metrics()])
def graphwalks_scorer():
    async def score(state, target: Target) -> Score:
        out = ""
        if getattr(state, "output", None) is not None:
            out = (
                getattr(state.output, "completion", None)
                or getattr(state.output, "text", "")
                or ""
            )

        pred, parse_err = _parse_nodes(out)
        gold = list(target)
        p, r, f1 = _prf1(pred, gold)

        # use precomputed token count if available, else compute on the fly
        md = getattr(state, "metadata", None) or {}
        tok_cnt = int(md.get("raw_input_tok_cnt") or get_token_count(str(state.input)))

        return Score(
            value=f1,
            answer=str(pred),
            metadata={
                "precision": p,
                "recall": r,
                "f1": f1,
                "parsed_ok": (not parse_err),
                "pred": pred,
                "gold": gold,
                "raw_input_tok_cnt": tok_cnt,
            },
        )

    return score