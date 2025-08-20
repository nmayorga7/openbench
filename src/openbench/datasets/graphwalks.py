from __future__ import annotations
from typing import Any, Optional, Callable
from inspect_ai.dataset import Dataset, Sample, hf_dataset, FieldSpec
from openbench.utils.text import get_token_count

_ALLOWED = {"bfs", "parents"}

def _estimate_tokens_from_chars(chars: int) -> int:
    # Simple, conservative chars→tokens conversion
    return max(1, (chars + 3) // 4)

def record_to_sample(
    allowed: Optional[set[str]] = None,
    max_context_size: Optional[int] = None,
) -> FieldSpec | Callable[[dict[str, Any]], Sample | list[Sample]]:

    def _record_to_sample(record: dict[str, Any]) -> Sample | list[Sample]:
        problem_type = (record.get("problem_type") or "").strip().lower()

        # 1) Cheap type filter
        if allowed is not None and problem_type not in allowed:
            return []

        prompt: str = record["prompt"]

        # 2) Always compute (or estimate) token count for binning/metrics
        try:
            tok_cnt = int(get_token_count(prompt))
        except Exception:
            # Fall back to a reasonable estimate based on characters
            tok_cnt = _estimate_tokens_from_chars(int(record.get("prompt_chars") or len(prompt)))

        # 3) Optional gating by max_context_size (use exact count only;
        #    if we only had an estimate due to error, do NOT drop)
        if max_context_size is not None:
            try:
                exact_tok_cnt = int(get_token_count(prompt))  # re-try once for gating accuracy
                if exact_tok_cnt > max_context_size:
                    return []
                tok_cnt = exact_tok_cnt  # keep exact value in metadata
            except Exception:
                # If tokenizer still fails, do not drop on an estimate
                pass

        gold = record.get("answer", record.get("answer_nodes", []))

        return Sample(
            input=prompt,
            target=gold,
            metadata={
                "problem_type": problem_type,
                "prompt_chars": record.get("prompt_chars"),
                "raw_input_tok_cnt": tok_cnt,   # <- always set
            },
        )

    return _record_to_sample


def get_dataset(
    split: str = "train",
    task_type: str = "both",
    max_context_size: Optional[int] = None,
) -> Dataset:
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
