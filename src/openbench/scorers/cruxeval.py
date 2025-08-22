from __future__ import annotations

import re
from typing import Optional

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


# How long to allow per assertion check (seconds)
TIMEOUT = 3


@scorer(metrics=[accuracy(), stderr()])
def cruxeval_verify() -> Scorer:
    """
    Functional-correctness scorer for CRUXEval (input- and output-prediction).

    Expects Sample.metadata to include:
      - mode: "input" | "output"
      - variant: "direct" | "cot"     (only affects parsing; scorer is uniform)
      - code: the Python source defining f(...)
      - ref_input: the canonical input string from the dataset (e.g. "'abc', 3")
      - ref_output: the canonical output literal string (e.g. '"hello"' or "[1,2]")

    The model must produce either:
      - OUTPUT mode: literal RHS to satisfy: assert f(ref_input) == <candidate>
      - INPUT  mode: an f(...) call so that: assert ref_output == <candidate>
    """

    async def score(state: TaskState, _: Target) -> Score:
        # Pull required metadata
        mode = str(state.metadata.get("mode", "")).strip()
        variant = str(state.metadata.get("variant", "")).strip()
        code = str(state.metadata.get("code", ""))
        ref_input = str(state.metadata.get("ref_input", ""))
        ref_output = str(state.metadata.get("ref_output", ""))

        # 1) Extract candidate from completion
        raw = state.output.completion or ""
        candidate = extract_candidate(raw, mode=mode, variant=variant)

        # Minimal structural sanity checks (mirror upstream behavior)
        if mode == "input":
            # Should be an f(...) expression
            if "f(" not in candidate:
                return _incorrect(
                    explanation=_fmt_expl(
                        code, ref_input, ref_output, candidate, "Missing f(…) call in input-prediction."
                    )
                )
        elif mode == "output":
            # Should *not* be a full assert or an f(...) call
            if f"f({ref_input})" in candidate:
                return _incorrect(
                    explanation=_fmt_expl(
                        code, ref_input, ref_output, candidate, "Output mode expects a literal RHS, not a function call."
                    )
                )
        else:
            return _incorrect(explanation="Unknown mode; expected 'input' or 'output'.")

        # 2) Build verification program
        #    Both modes reduce to: assert <expected> == <candidate>
        #    - OUTPUT: expected = f(ref_input), candidate = model literal
        #    - INPUT : expected = ref_output   , candidate = model f(...)
        if mode == "output":
            expected = f"f({ref_input})"
        else:  # "input"
            expected = ref_output

        verify_code = [
            code,
            "\n",
            f"assert {expected} == {candidate}",
            "\n",
        ]

        # 3) Execute in sandbox
        try:
            result: ExecResult = await sandbox().exec(
                cmd=["python", "-c", "".join(verify_code)],
                timeout=TIMEOUT,
            )
        except:
            result = ExecResult(False, 1, "", "Error arose.")

        if result.success:
            return Score(
                value=CORRECT,
                answer=candidate,
                explanation=_fmt_expl(code, ref_input, ref_output, candidate, "Assertion passed."),
            )
        else:
            return _incorrect(
                explanation=_fmt_expl(
                    code,
                    ref_input,
                    ref_output,
                    candidate,
                    f"Assertion failed.\n\n{result.stderr or ''}".strip(),
                )
            )

    return score


# -----------------------
# Parsing helper routines
# -----------------------

ANSWER_OPEN = "[ANSWER]"
ANSWER_CLOSE = "[/ANSWER]"
THOUGHT_OPEN = "[THOUGHT]"
THOUGHT_CLOSE = "[/THOUGHT]"


def extract_candidate(completion: str, mode: str, variant: str) -> str:
    """
    Extract the model's *evaluatable* candidate from its completion.

    Strategy:
      1) If [ANSWER]…[/ANSWER] exists, use that span.
      2) Else, take the last non-empty line.
      3) Normalize from "assert f(...) == X" form to the required expression:
         - OUTPUT mode -> take RHS literal after '=='
         - INPUT  mode -> extract the 'f(...)' expression (LHS before '==' or bare f-call)
    """
    text = completion.strip()

    # Prefer explicit answer tags
    inside = _between(text, ANSWER_OPEN, ANSWER_CLOSE)
    if inside is None:
        # strip any trailing reasoning if CoT forgot tags; take last non-empty line
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        inside = lines[-1] if lines else ""

    # If the model echoed an "assert ..." line, keep it; else keep raw
    s = inside

    # Normalize whitespace
    s = s.strip()

    # If there are code fences, remove them
    s = _strip_first_fenced_block(s)

    # If it's a full assert line, split on '=='
    if s.startswith("assert "):
        # E.g., "assert f(1,2) == [3,4]"
        parts = s[len("assert ") :].split("==")
        parts = [p.strip() for p in parts]
        if len(parts) >= 2:
            lhs, rhs = parts[0], parts[1]
            if mode == "output":
                # Need the RHS literal
                return rhs
            else:
                # Need the f(...) call (prefer LHS if it already is f(...))
                return lhs if "f(" in lhs else _first_f_call(s)
        else:
            # malformed assert; fall through to best-effort
            pass

    # Not an assert — try to recover
    if mode == "output":
        # Expect a literal; if the model printed something like: f(x) == 42, grab RHS
        if "==" in s:
            return s.split("==", 1)[1].strip()
        # Else assume the whole thing is the literal
        return s
    else:
        # INPUT mode — try to find an f(...) call anywhere
        call = _first_f_call(s)
        return call if call else s


def _first_f_call(text: str) -> Optional[str]:
    # Capture a best-effort 'f(<anything balanced-ish>)'
    # This is permissive: matches the shortest 'f(... )' group.
    m = re.search(r"f\s*\((.*)\)", text)
    if not m:
        return None
    # Reconstruct "f(<args>)" from the match
    # Try to trim trailing comments or hash-completions
    candidate = "f(" + m.group(1).strip()
    # If the string contains extra closing parens, keep as-is; the assert will tell us if it's valid Python.
    return candidate


def _between(s: str, open_tag: str, close_tag: str) -> Optional[str]:
    i = s.find(open_tag)
    if i == -1:
        return None
    j = s.find(close_tag, i + len(open_tag))
    if j == -1:
        return None
    return s[i + len(open_tag) : j].strip()


def _strip_first_fenced_block(s: str) -> str:
    # Remove leading ```...``` block if present
    fence = re.compile(r"^```(?:python)?\s*\n(.*?)\n```$", re.DOTALL | re.IGNORECASE)
    m = fence.match(s)
    return m.group(1).strip() if m else s


# -----------------------
# Small helpers
# -----------------------

def _fmt_expl(code: str, ref_input: str, ref_output: str, candidate: str, note: str) -> str:
    return "".join(
        [
            "The following verification code was executed:\n\n",
            "```python\n",
            code,
            "\n",
            f"assert {'f(' + ref_input + ')' if ref_input and note.startswith('Assertion passed') else (f'f({ref_input})' if ref_input else ref_output)} == {candidate}",
            "\n```",
            f"\n\n{note}",
        ]
    )


def _incorrect(explanation: str) -> Score:
    return Score(
        value=INCORRECT,
        answer="",
        explanation=explanation,
    )
