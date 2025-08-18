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
    """Mean F1 by token-count bin (MRCR-style; flat mapping)."""

    def metric_calculator(scores: list[SampleScore]) -> Value:
        f1_by_token_count_bin: dict[str, float] = {}
        bin_counts: dict[str, int] = {}

        # init bins
        for left_bin, right_bin in GRAPHWALKS_BINS:
            key = f"{left_bin}-{right_bin}"
            f1_by_token_count_bin[key] = 0.0
            bin_counts[key] = 0

        if not scores:
            # MRCR returns the flat mapping (zeros) when empty
            return f1_by_token_count_bin

        for sample_score in scores:
            md = sample_score.score.metadata or {}
            bin_index = md.get("bin_index")
            if (
                not isinstance(bin_index, int)
                or bin_index < 0
                or bin_index >= len(GRAPHWALKS_BINS)
            ):
                continue

            left_bin, right_bin = GRAPHWALKS_BINS[bin_index]
            key = f"{left_bin}-{right_bin}"

            try:
                f1_val = float(sample_score.score.as_float())  # per-sample scalar
            except Exception:
                continue

            f1_by_token_count_bin[key] += f1_val
            bin_counts[key] += 1

        # average per bin
        for key in f1_by_token_count_bin:
            cnt = bin_counts[key]
            if cnt > 0:
                f1_by_token_count_bin[key] /= cnt

        return f1_by_token_count_bin

    return metric_calculator

@metric
def graphwalks_token_counts() -> Metric:
    def calc(scores: list[SampleScore]) -> Value:
        counts = {f"{L}-{R}": 0 for (L, R) in GRAPHWALKS_BINS}
        for s in scores:
            md = s.score.metadata or {}
            bidx = md.get("bin_index")
            if isinstance(bidx, int) and 0 <= bidx < len(GRAPHWALKS_BINS):
                L, R = GRAPHWALKS_BINS[bidx]
                counts[f"{L}-{R}"] += 1
        # flat dict; numeric values
        return {f"samples_per_bin[{k}]": float(v) for k, v in counts.items()}
    return calc


@scorer(metrics=[mean(), graphwalks_metrics(), graphwalks_token_counts()])
def graphwalks_scorer():
    async def score(state, target: Target) -> Score:
        # 1) get output text
        out = ""
        if getattr(state, "output", None) is not None:
            out = (
                getattr(state.output, "completion", None)
                or getattr(state.output, "text", "")
                or ""
            )

        # 2) parse prediction + compute PRF1
        pred, parse_err = _parse_nodes(out)
        gold = list(target)
        p, r, f1 = _prf1(pred, gold)

        # 3) token counts (MRCR-style total = input + output)
        md_in = getattr(state, "metadata", None) or {}
        input_tok_cnt = int(md_in.get("raw_input_tok_cnt", 0))

        # Serialize gold to a compact string for counting (no nesting)
        try:
            gold_str = ",".join(map(str, gold))
        except Exception:
            gold_str = ""
        output_tok_cnt = int(get_token_count(gold_str))
        total_tok_cnt = input_tok_cnt + output_tok_cnt

        # 4) compute bin_index inline (mirror MRCR’s boundary handling)
        bin_index = 0
        for i, (left_bin, right_bin) in enumerate(GRAPHWALKS_BINS):
            if i == 0 or i == len(GRAPHWALKS_BINS) - 1:
                # First and last bins inclusive on both ends
                if left_bin <= total_tok_cnt <= right_bin:
                    bin_index = i
                    break
            else:
                # Middle bins: [left, right)
                if left_bin <= total_tok_cnt < right_bin:
                    bin_index = i
                    break
                
        print(f"tok={total_tok_cnt} -> bin={bin_index}")

        # 5) return per-sample score
        return Score(
            value=float(f1),
            answer=str(pred),
            metadata={
                "precision": float(p),
                "recall": float(r),
                "f1": float(f1),
                "parsed_ok": (not parse_err),
                "pred": pred,
                "gold": gold,
                "raw_input_tok_cnt": input_tok_cnt,
                "total_tok_cnt": total_tok_cnt,
                "bin_index": bin_index,  # <-- MRCR pattern
            },
        )
    return score