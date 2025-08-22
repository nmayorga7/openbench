# src/openbench/datasets/cruxeval.py

from typing import Any, Callable, Literal
from inspect_ai.dataset import Sample, Dataset, hf_dataset

Mode = Literal["input", "output"]
Variant = Literal["direct", "cot"]

# -------------------------
# Prompt builders (4 only)
# -------------------------

def _make_direct_output_prompt(code: str, inp: str) -> str:
    return f"""You are given a Python function and an assertion containing an input to the function.
Complete the assertion with a literal (no expressions or function calls) equal to the output of executing the code on that input.
Provide ONLY the full assertion in [ANSWER] and [/ANSWER] tags.

[PYTHON]
{code}
assert f({inp}) == ??
[/PYTHON]
[ANSWER]
"""

def _make_cot_output_prompt(code: str, inp: str) -> str:
    return f"""You are given a Python function and an assertion containing an input to the function.
Think step by step in a [THOUGHT] block, then provide ONLY the full assertion in [ANSWER] and [/ANSWER] tags.

[PYTHON]
{code}
assert f({inp}) == ??
[/PYTHON]
[THOUGHT]
"""
    # model should continue and then finish with:
    # [ANSWER]
    # assert f({inp}) == <literal>
    # [/ANSWER]

def _make_direct_input_prompt(code: str, outp: str) -> str:
    return f"""You will be given a function f and an output in the form f(??) == output.
Find any input such that executing f on the input leads to the given output.
Provide ONLY the full assertion in [ANSWER] and [/ANSWER] tags.

[PYTHON]
{code}
assert f(??) == {outp}
[/PYTHON]
[ANSWER]
"""

def _make_cot_input_prompt(code: str, outp: str) -> str:
    return f"""You will be given a function f and an output in the form f(??) == output.
Think step by step in a [THOUGHT] block, then provide ONLY the full assertion in [ANSWER] and [/ANSWER] tags.

[PYTHON]
{code}
assert f(??) == {outp}
[/PYTHON]
[THOUGHT]
"""
    # model should continue and then finish with:
    # [ANSWER]
    # assert f(<any valid input>) == {outp}
    # [/ANSWER]

# ------------------------------------------------
# OpenBench dataset API: record_to_sample + getter
# ------------------------------------------------

def record_to_sample(
    mode: Mode = "output",
    variant: Variant = "direct",
) -> Callable[[dict[str, Any]], Sample]:
    """
    Mapper from HF CRUXEval records to Inspect Samples.

    HF record fields: code (str), input (str), output (str), id (str)
    - mode="output": CRUXEval-O (predict literal RHS for assert f(input) == ??)
    - mode="input":  CRUXEval-I (find input so assert f(??) == output holds)
    - variant: "direct" | "cot"
    """

    if mode not in ("input", "output"):
        raise ValueError("mode must be 'input' or 'output'")
    if variant not in ("direct", "cot"):
        raise ValueError("variant must be 'direct' or 'cot'")

    def _build_prompt(code: str, inp: str, outp: str) -> str:
        if mode == "output":
            return (
                _make_cot_output_prompt(code, inp)
                if variant == "cot"
                else _make_direct_output_prompt(code, inp)
            )
        else:  # mode == "input"
            return (
                _make_cot_input_prompt(code, outp)
                if variant == "cot"
                else _make_direct_input_prompt(code, outp)
            )

    def _record_to_sample(rec: dict[str, Any]) -> Sample:
        code: str = rec["code"]
        inp: str = rec["input"]
        outp: str = rec["output"]
        sid: str = rec.get("id", "")

        return Sample(
            id=sid,
            input=_build_prompt(code, inp, outp),
            target="",  # functional-correctness scorer uses metadata + exec, not target
            metadata={
                "mode": mode,
                "variant": variant,
                "code": code,
                "ref_input": inp,
                "ref_output": outp,
            },
        )

    return _record_to_sample


def get_dataset(
    mode: Mode = "output",
    variant: Variant = "direct",
    split: str = "test",
) -> Dataset:
    """
    Load CRUXEval from HF and return a Dataset of Samples.
    Source: https://huggingface.co/datasets/cruxeval-org/cruxeval
    """
    return hf_dataset(
        path="cruxeval-org/cruxeval",
        split=split,
        sample_fields=record_to_sample(mode=mode, variant=variant),
        #name=f"cruxeval_{mode}_{variant}",
    )
