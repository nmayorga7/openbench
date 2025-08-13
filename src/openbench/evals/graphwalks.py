# src/openbench/evals/graphwalks.py
from __future__ import annotations

from inspect_ai import task, Task
from inspect_ai.model import ChatMessage, GenerateConfig
from inspect_ai.solver import generate

from openbench.datasets.graphwalks import get_dataset
from openbench.scorers.graphwalks import graphwalks_scorer


@task
def graphwalks(split: str = "train") -> Task:
    """
    OpenAI GraphWalks eval:
      - loads HF dataset (prompt -> input, answer(s) -> target)
      - sends the prompt verbatim as a single user message
      - scores outputs with set Precision/Recall/F1 (reports mean F1)
    """
    return Task(
        dataset=get_dataset(split=split),
        # Single-turn: pass the dataset prompt exactly as the user message
        solver=[generate(messages=lambda row: [ChatMessage.user(row.input)])],
        scorer=graphwalks_scorer(),
        name="graphwalks",
        config=GenerateConfig(
            temperature=0.0,
            top_p=1.0,
            max_tokens=256,
        ),
    )
