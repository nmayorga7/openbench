# src/openbench/scorers/graphwalks.py
from __future__ import annotations

import re
from typing import Set

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

# Parse ONLY the very last line, which must look like:
#   Final Answer: [a, b, c]
_FINAL_LINE_RE = re.compile(r"Final Answer:\s*\[(.*)\]\s*$", re.IGNORECASE)


def _parse_nodes(text: str) -> tuple[list[str], bool]:
    """Return (nodes, parse_error_flag). Dedup while preserving order."""
    if not text:
        return [], True
    last_line = text.strip().splitlines()[-1]
    m = _FINAL_LINE_RE.search(last_line)
    if not m:
        return [], True
    inner = m.group(1)
    # split by commas only; trim; drop empties; dedup preserving order
    raw = [t.strip() for t in inner.split(",")]
    seen: Set[str] = set()
    out: list[str] = []
    for t in raw:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out, False


def _prf1(pred: list[str], gold: list[str]) -> tuple[float, float, float]:
    sp, sg = set(pred), set(gold)
    inter = len(sp & sg)
    p = inter / len(sp) if sp else 0.0
    r = inter / len(sg) if sg else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


# Token count bins for Graphwalks (can be adjusted based on your needs)
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
    """Calculate Graphwalks specific metrics: F1 score by token count bin.
    
    Bin boundaries are:
    [0, 512], (512, 1024], (1024, 2048], (2048, 4096], 
    (4096, 8192], (8192, 16384], (16384, 32768], (32768, 65536]
    """
    
    def metric_calculator(scores: list[SampleScore]) -> Value:
        f1_by_token_count_bin: dict[str, float] = {}
        precision_by_token_count_bin: dict[str, float] = {}
        recall_by_token_count_bin: dict[str, float] = {}
        bin_counts: dict[str, int] = {}
        
        for left_bin, right_bin in GRAPHWALKS_BINS:
            bin_key = f"{left_bin}-{right_bin}"
            f1_by_token_count_bin[bin_key] = 0.0
            precision_by_token_count_bin[bin_key] = 0.0
            recall_by_token_count_bin[bin_key] = 0.0
            bin_counts[bin_key] = 0
        
        if not scores:
            return {
                "f1_by_token_count": f1_by_token_count_bin,
                "precision_by_token_count": precision_by_token_count_bin,
                "recall_by_token_count": recall_by_token_count_bin,
            }
        
        for sample_score in scores:
            if sample_score.score.metadata is None:
                continue
            
            # Get token count and metrics from metadata
            raw_input_tok_cnt = sample_score.score.metadata.get("raw_input_tok_cnt", 0)
            precision = sample_score.score.metadata.get("precision", 0.0)
            recall = sample_score.score.metadata.get("recall", 0.0)
            f1 = sample_score.score.metadata.get("f1", 0.0)
            
            # Find the appropriate bin
            for i, (left_bin, right_bin) in enumerate(GRAPHWALKS_BINS):
                if i == 0:  # First bin includes left boundary
                    if left_bin <= raw_input_tok_cnt <= right_bin:
                        bin_key = f"{left_bin}-{right_bin}"
                        f1_by_token_count_bin[bin_key] += f1
                        precision_by_token_count_bin[bin_key] += precision
                        recall_by_token_count_bin[bin_key] += recall
                        bin_counts[bin_key] += 1
                        break
                else:  # Other bins exclude left boundary
                    if left_bin < raw_input_tok_cnt <= right_bin:
                        bin_key = f"{left_bin}-{right_bin}"
                        f1_by_token_count_bin[bin_key] += f1
                        precision_by_token_count_bin[bin_key] += precision
                        recall_by_token_count_bin[bin_key] += recall
                        bin_counts[bin_key] += 1
                        break
        
        # Calculate averages for each bin
        for bin_key in f1_by_token_count_bin:
            if bin_counts[bin_key] > 0:
                f1_by_token_count_bin[bin_key] /= bin_counts[bin_key]
                precision_by_token_count_bin[bin_key] /= bin_counts[bin_key]
                recall_by_token_count_bin[bin_key] /= bin_counts[bin_key]
        
        return {
            "f1_by_token_count": f1_by_token_count_bin,
            "precision_by_token_count": precision_by_token_count_bin,
            "recall_by_token_count": recall_by_token_count_bin,
            "samples_per_bin": bin_counts,
        }
    
    return metric_calculator


@scorer(metrics=[mean(), stderr(), graphwalks_metrics()])  # UI will show Mean (and stderr) of F1, plus binned metrics
def graphwalks_scorer():
    async def score(state, target: Target) -> Score:
        # Inspect model output: prefer .completion, fall back to .text if needed
        out = ""
        if getattr(state, "output", None) is not None:
            out = (
                getattr(state.output, "completion", None)
                or getattr(state.output, "text", "")
                or ""
            )

        pred, parse_err = _parse_nodes(out)
        gold = list(target)  # Target is a sequence of gold node strings

        p, r, f1 = _prf1(pred, gold)
        
        # Get token count from metadata if available
        raw_input_tok_cnt = 0
        if hasattr(state, "metadata") and state.metadata:
            raw_input_tok_cnt = state.metadata.get("raw_input_tok_cnt", 0)
        
        return Score(
            value=f1,  # Mean in the UI = mean F1
            answer=str(pred),
            metadata={
                "precision": p,
                "recall": r,
                "f1": f1,
                "parsed_ok": (not parse_err),
                "pred": pred,
                "gold": gold,
                "raw_input_tok_cnt": raw_input_tok_cnt,
            },
        )

    return score
