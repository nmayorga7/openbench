# src/openbench/evals/mbpp.py
"""
MBPP (Mostly Basic Python Problems) evaluation for OpenBench (built on Inspect).

Generates Python solutions for MBPP prompts and verifies them by executing
unit tests inside a sandbox (Docker strongly recommended).

Requires:
  - datasets/mbpp.py   -> get_dataset(...)
  - scorers/mbpp.py    -> mbpp_verify(...)

CLI example:
  bench eval mbpp --model openai/gpt-4o-mini --temperature 0 \
    -T config=sanitized split=test limit=25 timeout=6 epochs=1 use_docker=true
"""

from __future__ import annotations

from typing import Optional

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate
from inspect_ai.util import Epochs

from openbench.datasets.mbpp import get_dataset
from openbench.scorers.mbpp import mbpp_verify


@task
def mbpp(
    *,
    # ---- Dataset controls ----
    config: str = "sanitized",        # HF subset: "sanitized" or ""/"full"
    split: str = "test",
    limit: Optional[int] = None,
    instruction_prompt: Optional[str] = None,  # override adapter default if provided

    # ---- Execution / scoring ----
    timeout: int = 6,                 # seconds for python -c execution in sandbox
    epochs: int = 1,                  # >1 approximates pass@k
    temperature: float = 0.0,         # determinism for code generation

) -> Task:

    if instruction_prompt is None:
        dataset = get_dataset(
            config=config,
            split=split,
            limit=limit,
            name=f"mbpp_{config}_{split}".strip("_"),
        )
    else:
        dataset = get_dataset(
            config=config,
            split=split,
            limit=limit,
            instruction_prompt=instruction_prompt,      # custom instruction prompt
            name=f"mbpp_{config}_{split}".strip("_"),
        )
    
    return Task(
        dataset=dataset,
        solver=generate(),
        scorer=mbpp_verify(timeout=timeout),
        sandbox="local",
        config=GenerateConfig(temperature=temperature),
        epochs=Epochs(epochs),
        name="mbpp",
    )
