# src/openbench/datasets/graphwalks.py
from __future__ import annotations
from typing import Any, Optional, Callable
from inspect_ai.dataset import Dataset, Sample, hf_dataset, FieldSpec
from openbench.utils.text import get_token_count

_ALLOWED = {"bfs", "parents"}


def record_to_sample(
    allowed: Optional[set[str]] = None,
    max_context_size: Optional[int] = None,
) -> FieldSpec | Callable[[dict[str, Any]], Sample | list[Sample]]:
    """
    Map one HF row to an Inspect Sample.
    - If `allowed` is provided, drop rows whose problem_type isn't in it (return []).
    - If `max_context_size` is provided, compute token count and drop rows that exceed it.
    """

    def _record_to_sample(record: dict[str, Any]) -> Sample | list[Sample]:
        problem_type = (record.get("problem_type") or "").strip().lower()

        # Type filter (fast) — do this first
        if allowed is not None and problem_type not in allowed:
            return []

        prompt: str = record["prompt"]

        tok_cnt: Optional[int] = None
        if max_context_size is not None:
            # Only compute tokens when we actually need gating
            try:
                tok_cnt = int(get_token_count(prompt))
            except Exception:
                # Never block the pipeline on tokenization; default to a large value so it won't be dropped by mistake
                tok_cnt = 0

            if tok_cnt > max_context_size:
                return []

        gold = record.get("answer", record.get("answer_nodes", []))

        md = {
            "problem_type": problem_type,
            "prompt_chars": record.get("prompt_chars"),
        }
        if tok_cnt is not None:
            md["raw_input_tok_cnt"] = tok_cnt

        return Sample(input=prompt, target=gold, metadata=md)

    return _record_to_sample


def get_dataset(
    split: str = "train",
    task_type: str = "both",
    max_context_size: Optional[int] = None,
) -> Dataset:
    """
    task_type: 'bfs' | 'parents' | 'both' (default: keep all)
    """
    task = (task_type or "both").strip().lower()
    if task in ("both", "all", "*"):
        allowed = None
    elif task in _ALLOWED:
        allowed = {task}
    else:
        raise ValueError("task_type must be one of 'bfs', 'parents', 'both'")

    return hf_dataset(
        path="openai/graphwalks",
        split=split,
        sample_fields=record_to_sample(allowed=allowed, max_context_size=max_context_size),
    )
