# src/openbench/scorers/mbpp.py
import re
from typing import Any, Callable, List, Optional

from inspect_ai.scorer import (
    Score,
    Scorer,
    Target,
    accuracy,
    CORRECT,
    INCORRECT,
    stderr,
    scorer,
)
from inspect_ai.solver import TaskState
from inspect_ai.util import ExecResult, sandbox

TIMEOUT_DEFAULT = 6
SENTINEL = "__MBPP_OK__"


def find_code(completion: str) -> str:
    """
    Extract Python code from a completion. Prefer the first ```python fenced block,
    else return the raw completion.
    """
    if not completion:
        return ""
    pattern = re.compile(r"```python\s*(.*?)```", re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(completion)
    extracted = matches[0] if matches else completion
    return str(extracted).strip()


def _normalize_tests(tests: Any) -> List[str]:
    """Accept list[str] or str; return list[str] of test statements."""
    if tests is None:
        return []
    if isinstance(tests, list):
        return [t for t in (str(x).rstrip() for x in tests) if t]
    if isinstance(tests, str):
        return [ln.rstrip() for ln in tests.splitlines() if ln.strip()]
    return []


def _assemble_program(
    *,
    setup: Optional[str],
    candidate_code: str,
    tests: List[str],
    add_sentinel: bool = True,
) -> str:
    parts: List[str] = []
    if setup and setup.strip():
        parts.append(setup.strip())
    parts.append(candidate_code.strip())
    if tests:
        parts.append("\n".join(tests))
    if add_sentinel:
        parts.append(f'print("{SENTINEL}")')
    return "\n\n".join(parts)


@scorer(metrics=[accuracy(), stderr()])
def mbpp_verify(timeout: int = TIMEOUT_DEFAULT, require_sentinel: bool = True) -> Scorer:
    """
    Execute model-generated MBPP solutions and unit tests inside the sandbox.

    - Builds one Python program: [setup] + [candidate] + [tests] + sentinel
    - Runs: python -c "<program>"
    - Pass iff exit code == 0 and (optionally) sentinel appears in stdout.
    """

    async def score(state: TaskState, target: Target) -> Score:
        md = state.metadata or {}
        task_id = md.get("task_id")
        prompt = md.get("prompt", "")
        setup = md.get("setup")
        tests = _normalize_tests(md.get("tests"))

        # 1) Extract candidate in HumanEval style
        raw = state.output.completion if state.output else ""
        candidate = find_code(raw)

        # Guard rail: empty candidate
        if not candidate.strip():
            prog = _assemble_program(setup=setup, candidate_code="", tests=tests, add_sentinel=True)
            explanation = (
                "No candidate code found.\n\n"
                "The following verification code was executed:\n\n```python\n"
                + prog + "\n```\n"
            )
            return Score(value=INCORRECT, answer="", explanation=explanation, metadata={"task_id": task_id})

        # 2) Assemble the verification program
        program = _assemble_program(setup=setup, candidate_code=candidate, tests=tests, add_sentinel=True)

        # 3) Execute in sandbox
        try:
            result = await sandbox().exec(
                cmd=["python", "-c", program],
                timeout=timeout,
            )
        except TimeoutError:
            result = ExecResult(False, 1, "", "Verification timed out.")
        except Exception as e:
            result = ExecResult(False, 1, "", f"{type(e).__name__}: {e}")

        # 4) Decide pass/fail
        passed = bool(result.success) and ((SENTINEL in (result.stdout or "")) if require_sentinel else True)
        value = CORRECT if passed else INCORRECT

        # 5) HumanEval-like explanation blob (show exactly what ran)
        explanation_parts = [
            "The following verification code was executed:\n\n```python\n",
            program,
            "\n```\n",
        ]
        if not passed:
            explanation_parts.extend(
                [
                    "The submission was incorrect.\n\n",
                    (result.stderr or ""),
                ]
            )
        explanation = "".join(explanation_parts)

        return Score(
            value=value,
            answer=candidate,
            explanation=explanation,
            metadata={
                "task_id": task_id,
                "prompt": prompt,
                "tests_count": len(tests),
                "timeout_s": timeout,
                "exit_success": bool(result.success),
                "sentinel_seen": (SENTINEL in (result.stdout or "")) if require_sentinel else None,
            },
        )

    return score