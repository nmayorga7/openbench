# src/openbench/datasets/graphwalks.py
from __future__ import annotations
from typing import Any, Optional, Callable
from inspect_ai.dataset import Dataset, Sample, hf_dataset, FieldSpec
from openbench.utils.text import get_token_count

_ALLOWED = {"bfs", "parents"}


def record_to_sample(
    max_context_size: Optional[int] = None,
    allowed: Optional[set[str]] = None,
) -> FieldSpec | Callable[[dict[str, Any]], Sample | list[Sample]]:
    """Create a mapper from Graphwalks records to Inspect Samples.
    
    Args:
        max_context_size: Maximum context size in tokens. Defaults to None.
        allowed: Set of allowed problem types. If provided, drop rows whose 
                problem_type isn't in it by returning empty list.
    
    Returns:
        A function that maps HF records to Samples, filtering by context size and problem type.
    """
    
    def _record_to_sample(record: dict[str, Any]) -> Sample | list[Sample]:
        problem_type = (record.get("problem_type") or "").strip().lower()
        
        # Filter by problem type
        if allowed is not None and problem_type not in allowed:
            return []
        
        prompt = record["prompt"]
        
        # Calculate token count and filter by max_context_size
        if max_context_size is not None:
            input_tok_cnt = get_token_count(prompt)
            if input_tok_cnt > max_context_size:
                return []
        else:
            input_tok_cnt = get_token_count(prompt)
        
        gold = record.get("answer", record.get("answer_nodes", []))
        
        return Sample(
            input=prompt,
            target=gold,
            metadata={
                "problem_type": problem_type,
                "prompt_chars": record.get("prompt_chars"),
                "raw_input_tok_cnt": input_tok_cnt,
            },
        )
    
    return _record_to_sample


def get_dataset(
    split: str = "train", 
    task_type: str = "both",
    max_context_size: Optional[int] = None
) -> Dataset:
    """Load the Graphwalks dataset.
    
    Args:
        split: Dataset split to load ('train', 'validation', 'test').
        task_type: 'bfs' | 'parents' | 'both' (default: keep all).
        max_context_size: Maximum context size in tokens. Defaults to None.
    
    Returns:
        Dataset filtered by task type and context size.
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
        sample_fields=record_to_sample(max_context_size=max_context_size, allowed=allowed),
    )
