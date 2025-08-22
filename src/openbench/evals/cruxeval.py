# src/openbench/evals/cruxeval.py

from __future__ import annotations

from typing import Literal, Optional

from inspect_ai import Task, task, Epochs
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate

from openbench.datasets.cruxeval import get_dataset
from openbench.scorers.cruxeval import cruxeval_verify

Mode = Literal["input", "output"]
Variant = Literal["direct", "cot"]


@task
def cruxeval(
    mode: Mode = "output",
    variant: Variant = "direct",
    epochs: int = 10,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
) -> Task:
    """
    CRUXEval task for OpenBench.

    Args:
        mode: "output" (CRUXEval-O) or "input" (CRUXEval-I).
        variant: "direct" or "cot" (chain-of-thought prompt variant).
        epochs: number of samples per prompt (controls pass@k headroom).
        temperature: decoding temperature for generations.
        max_tokens: override max tokens (defaults chosen per variant).

    Returns:
        Inspect Task configured for CRUXEval.
    """
    # Pick sensible defaults per variant
    if max_tokens is None:
        max_tokens = 1000 if variant == "cot" else 200

    dataset = get_dataset(mode=mode, variant=variant)

    # We always stop on the closing answer tag; prompts are built to include it.
    stop = ["[/ANSWER]"]

    return Task(
        name=f"cruxeval_{mode}_{variant}",
        dataset=dataset,
        solver=generate(),
        scorer=cruxeval_verify(),
        sandbox="local",
        config=GenerateConfig(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        ),
        # Request N generations per sample; reducers compute mean + pass@k
        epochs=Epochs(epochs, reducer=["mean", "pass_at_1", "pass_at_5"]),
    )


# -----------------------------
# Convenience wrappers (optional)
# -----------------------------

@task
def cruxeval_input_direct(
    epochs: int = 10,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
) -> Task:
    return cruxeval(
        mode="input",
        variant="direct",
        epochs=epochs,
        temperature=temperature,
        max_tokens=max_tokens,
    )


@task
def cruxeval_input_cot(
    epochs: int = 10,
    temperature: float = 0.5,
    max_tokens: Optional[int] = None,
) -> Task:
    return cruxeval(
        mode="input",
        variant="cot",
        epochs=epochs,
        temperature=temperature,
        max_tokens=max_tokens,
    )


@task
def cruxeval_output_direct(
    epochs: int = 10,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
) -> Task:
    return cruxeval(
        mode="output",
        variant="direct",
        epochs=epochs,
        temperature=temperature,
        max_tokens=max_tokens,
    )


@task
def cruxeval_output_cot(
    epochs: int = 10,
    temperature: float = 0.5,
    max_tokens: Optional[int] = None,
) -> Task:
    return cruxeval(
        mode="output",
        variant="cot",
        epochs=epochs,
        temperature=temperature,
        max_tokens=max_tokens,
    )
