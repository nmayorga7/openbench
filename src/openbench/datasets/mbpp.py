from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from inspect_ai.dataset import Dataset, Sample, hf_dataset, FieldSpec
from inspect_ai.model import ChatMessageUser

MBPP_INSTRUCTION = """
    You will be given a short Python programming task.
    Write a single Python function that satisfies the prompt.
    Your response should only contain the code for this function.
    """.strip()

# Normalize keys that commonly appear across MBPP variants.
_TEST_LIST_KEYS = ("test_list", "challenge_test_list")
_SETUP_KEYS = ("test_imports", "test_setup_code", "imports", "setup_code")
_PROMPT_KEYS = ("prompt", "text", "task_description")
_CODE_KEYS = ("code", "solution", "canonical_solution", "reference_code")


def _get_first(record: Dict[str, Any], *candidates: str, default: Any = None) -> Any:
    """Return the first present, non-empty value among candidate keys."""
    for k in candidates:
        if k in record and record[k] not in (None, ""):
            return record[k]
    return default


def record_to_sample(
    instruction_prompt: str = MBPP_INSTRUCTION,
    include_reference_in_target: bool = True,
) -> FieldSpec | Callable[[Dict[str, Any]], Sample]:
    """
    Mapper from MBPP records to Inspect Samples.

    Parameters:
        instruction_prompt : str
            A prefix instruction prepended to the MBPP problem statement. This helps
            constrain the model to emit a single function with no extra text.
        include_reference_in_target : bool
            If True, set `Sample.target` to the dataset's reference code (when available).
            Scorers for code-exec usually don't *use* target, but it's handy for debugging.
    """

    def _map(record: Dict[str, Any]) -> Sample:
        # Identify an ID if present (task_id is common in MBPP)
        task_id = str(_get_first(record, "task_id", "id", default="")).strip() or None

        # Problem statement (prompt)
        prompt_text = str(_get_first(record, *_PROMPT_KEYS, default="")).strip()
        if not prompt_text:
            # Defensive: some variants may store the prompt under 'question'
            prompt_text = str(record.get("question", "")).strip()

        # Tests and optional setup/imports
        # test_list is commonly a list[str] of assert statements
        tests = _get_first(record, *_TEST_LIST_KEYS, default=None)
        # test setup/imports may be a string block with imports or helper defs
        setup = _get_first(record, *_SETUP_KEYS, default=None)

        # Optional reference code (not executed by the scorer, but useful)
        reference = _get_first(record, *_CODE_KEYS, default=None)

        # Build the chat-style input: a single user message with instruction + prompt
        user_msg = ChatMessageUser(content=f"{instruction_prompt}\n\n{prompt_text}".strip())

        # Metadata that the scorer will use to assemble the harness
        metadata: Dict[str, Any] = {
            "task_id": task_id,
            "prompt": prompt_text,
            "tests": tests,                 # expected: list[str] of asserts (or None)
            "setup": setup,                 # expected: str with imports/setup (or None)
            "reference_code": reference,    # optional debugging aid
        }

        # Target can hold the reference code or be empty
        target = reference if include_reference_in_target and isinstance(reference, str) else ""

        return Sample(
            id=task_id,
            input=[user_msg],
            target=target,
            metadata=metadata,
        )

    return _map


def get_dataset(
    config: str = "sanitized",
    split: str = "test",
    limit: Optional[int] = None,
    instruction_prompt: str = MBPP_INSTRUCTION,
    name: Optional[str] = None,
) -> Dataset:
    """
    Load MBPP from Hugging Face.

    Parameters: 
        config : str
            - "sanitized"  (fields: prompt, test_list, test_imports, code (often removed))
            - "full" or "" (dataset default, fields vary but include test_list/test_setup_code)
        split : str
            Dataset split to load (e.g., "test", "validation", "train").
        limit : Optional[int]
            Max number of samples (useful for quick local runs).
        instruction_prompt : str
            Instruction prefix appended before each problem statement.
        name : Optional[str]
            Name for the resulting Dataset (defaults to "mbpp_<config>_<split>").

    Returns:
        Dataset
    """
    ds_name = name or f"mbpp_{config}_{split}".strip("_")

    return hf_dataset(
        path="Muennighoff/mbpp",
        split=split,
        sample_fields=record_to_sample(instruction_prompt=instruction_prompt),
        limit=limit,
        name=ds_name,
    )
