# src/openbench/datasets/graphwalks.py
from __future__ import annotations
from typing import Any
from inspect_ai.dataset import Dataset, Sample, hf_dataset

# Optional: pin for reproducibility
# _DATASET_REV = "6fe75ac25ccf55853294fe7995332d4f59d91bfb"

def record_to_sample(record: dict[str, Any]) -> Sample:
    """
    HF schema:
      - prompt: str
      - answer (or answer_nodes): list[str]
      - problem_type: str
      - prompt_chars: int
    """
    gold = record.get("answer", record.get("answer_nodes", []))
    return Sample(
        input=record["prompt"],
        target=gold,
        metadata={
            "problem_type": record.get("problem_type"),
            "prompt_chars": record.get("prompt_chars"),
        },
    )

def get_dataset(split: str = "train") -> Dataset:
    return hf_dataset(
        path="openai/graphwalks",
        split=split,
        sample_fields=record_to_sample,
        # revision=_DATASET_REV,  # ← uncomment to pin
    )
