# src/openbench/datasets/graphwalks.py
from __future__ import annotations
from typing import Any, Optional, Callable
from inspect_ai.dataset import Dataset, Sample, hf_dataset, FieldSpec
from openbench.utils.text import get_token_count  # plain text token counter

_ALLOWED = {"bfs", "parents"}


def record_to_sample(
    allowed: Optional[set[str]] = None,
    max_context_size: Optional[int] = None,
) -> FieldSpec | Callable[[dict[str, Any]], Sample | list[Sample]]:
    """
    Create a mapper from GraphWalks records to Inspect Samples.

    - If `allowed` is provided, drop rows whose problem_type isn't in it by returning [].
    - If `max_context_size` is provided, drop rows whose prompt token count exceeds it.
    """

    def _record_to_sample(record: dict[str, Any]) -> Sample | list[Sample]:
        problem_type = (record.get("problem_type") or "").strip().lower()

        # Keep/skip by problem type (early drop)
        if allowed is not None and problem_type not in allowed:
            return []

        prompt: str = record["prompt"]
        # Count tokens in the same format you will send to the model (plain text here)
        tok_cnt = get_token_count(prompt)

        # Early drop by token budget (MRCR-style gating)
        if max_context_size is not None and tok_cnt > max_context_size:
            return []

        gold = record.get("answer", record.get("answer_nodes", []))

        return Sample(
            input=prompt,
            target=gold,
            metadata={
                "problem_type": problem_type,
                "prompt_chars": record.get("prompt_chars"),
                "raw_input_tok_cnt": tok_cnt,  # consumed by scorer/metrics for binning
            },
        )

    return _record_to_sample


def get_dataset(
    split: str = "train",
    task_type: str = "both",
    max_context_size: Optional[int] = None,
) -> Dataset:
    """
    task_type: 'bfs' | 'parents' | 'both' (default: keep all)
    max_context_size: if set, drop samples whose prompt token count exceeds this value
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
