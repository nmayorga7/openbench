import json
import re
import math
import ast
from collections import OrderedDict
from statistics import median
from typing import Any

from inspect_ai.scorer import Score, scorer, accuracy, stderr
from inspect_ai.scorer._metric import metric, SampleScore, Value, Metric
from inspect_ai.solver import TaskState

# mapping of expected fields by task
FIELDS_BY_TASK = {
    "time_fields": ["valid", "hours", "minutes", "seconds", "date", "month", "weekday"],
    "shift_fields": ["valid", "hours", "minutes", "seconds"],
    "angle_fields": ["valid", "hours", "minutes", "seconds"],
    "zone_fields": ["valid", "hours", "minutes", "seconds"],
}


# helper functions for detailed analysis
def calculate_percentage(numerator, denominator):
    """Calculate percentage, handling division by zero."""
    return None if denominator == 0 else round(numerator / denominator, 4)


def calculate_fraction(numerator, denominator):
    """Calculate fraction, handling division by zero."""
    return None if denominator == 0 else round(numerator / denominator, 4)


def calculate_range_midpoint(range_low, range_high):
    """Get midpoint for ranges."""
    return int(round((int(range_low) + int(range_high)) / 2.0))


def convert_expected_value_to_scalar(expected_value):
    """Convert expected value to scalar, handling false/none as zero and ranges as midpoint."""
    if expected_value is False or expected_value is None:
        return 0

    expected_int = convert_to_int_or_none(expected_value)
    if expected_int is not None:
        return expected_int

    # handle range values like [4, 5] by taking midpoint
    if (
        isinstance(expected_value, list)
        and len(expected_value) == 2
        and convert_to_int_or_none(expected_value[0]) is not None
        and convert_to_int_or_none(expected_value[1]) is not None
    ):
        return calculate_range_midpoint(
            convert_to_int_or_none(expected_value[0]),
            convert_to_int_or_none(expected_value[1]),
        )
    return None


def convert_predicted_value_to_scalar(predicted_value):
    """Convert model prediction to scalar."""
    if predicted_value is False or predicted_value is None:
        return 0
    return convert_to_int_or_none(predicted_value)


def determine_clock_period_hours(sample_key, ground_truth_hours):
    """Determine if clock uses 24h or 12h format for wrap-around calculations."""
    hours_int = convert_to_int_or_none(ground_truth_hours)
    if hours_int is not None and hours_int >= 13:
        return 24

    key_lower = str(sample_key).lower()
    if ("24" in key_lower) and ("hour" in key_lower):
        return 24
    return 12


def convert_time_to_seconds(hours, minutes, seconds, period_hours):
    """Convert time components to total seconds within period."""
    hours_normalized = (int(hours) if hours is not None else 0) % period_hours
    minutes_normalized = int(minutes) if minutes is not None else 0
    seconds_normalized = int(seconds) if seconds is not None else 0

    # map onto [0, period_hours*3600) for circular time comparison
    return (hours_normalized * 3600 + minutes_normalized * 60 + seconds_normalized) % (
        period_hours * 3600
    )


def convert_sec_to_hours_min(total_sec):
    """Convert seconds to hours and minutes format."""
    if total_sec is None:
        return None
    total_sec_int = int(round(total_sec))
    hours = total_sec_int // 3600
    min_val = (total_sec_int % 3600) // 60
    return {"hours": hours, "minutes": min_val}


def compare_gt_pred(ground_truth_obj, predicted_obj, required_fields):
    """Compare ground truth and predicted answer objects field by field."""

    ground_truth = normalize_answer_fields(ground_truth_obj, required_fields)
    predicted = normalize_answer_fields(predicted_obj, required_fields)

    comparison_details = OrderedDict()
    comparison_details["valid"] = (ground_truth.get("valid"), predicted.get("valid"))

    # validity comparison - must agree on valid/invalid
    if ground_truth.get("valid") is not predicted.get("valid"):
        return False, {**comparison_details, "reason": "validity_mismatch"}

    # if ground truth says invalid, other fields don't matter, return true
    if ground_truth.get("valid") is False:
        return True, comparison_details

    # if valid time, check all other fields
    all_fields_correct = True
    for field_name in required_fields:
        if field_name == "valid":
            continue

        field_matches = match_value(
            ground_truth.get(field_name), predicted.get(field_name)
        )
        comparison_details[field_name] = (
            ground_truth.get(field_name),
            predicted.get(field_name),
            field_matches,
        )
        all_fields_correct = all_fields_correct and field_matches

    return all_fields_correct, comparison_details


def parse_obj(value):
    """Parse potentially messy JSON from model responses."""
    if isinstance(value, dict):
        return value

    text = str(value).strip()
    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json|javascript|js)?\s*|\s*```$", "", text, flags=re.I | re.S
        )

    json_match = re.search(r"\{.*\}", text, flags=re.S)
    if json_match:
        text = json_match.group(0)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # fix trailing commas
    text_fixed_commas = re.sub(r",(\s*[}\]])", r"\1", text)
    text_fixed_commas = re.sub(
        r"(?m)(?<=\{|,)\s*([A-Za-z_]\w*)\s*:", r'"\1":', text_fixed_commas
    )
    try:
        return json.loads(text_fixed_commas)
    except json.JSONDecodeError:
        pass

    # convert JS booleans to Python
    text_python_bools = re.sub(r"\btrue\b", "True", text_fixed_commas, flags=re.I)
    text_python_bools = re.sub(r"\bfalse\b", "False", text_python_bools, flags=re.I)
    text_python_bools = re.sub(r"\bnull\b", "None", text_python_bools, flags=re.I)
    return ast.literal_eval(text_python_bools)


def is_finite_number(value):
    """Check if value is a finite number (not boolean, not NaN, not infinity)."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def convert_to_int_or_none(value):
    """Convert value to integer if possible, otherwise return None."""
    if is_finite_number(value):
        return int(value)
    if isinstance(value, str):
        value_stripped = value.strip()
        if re.fullmatch(r"-?\d+", value_stripped):
            return int(value_stripped)
    return None


def match_value(expected_value, actual_value):
    """Compare expected and actual values with flexible matching rules."""

    # strings: case-insensitive comparison
    if isinstance(expected_value, str):
        return (
            isinstance(actual_value, str)
            and expected_value.strip().casefold()
            == str(actual_value).strip().casefold()
        )

    # booleans and None: exact comparison
    if isinstance(expected_value, (bool, type(None))):
        return expected_value == actual_value

    # numeric comparison
    if is_finite_number(expected_value):
        actual_int = convert_to_int_or_none(actual_value)
        return actual_int is not None and actual_int == int(expected_value)

    # list comparison with inclusive range or choices
    if isinstance(expected_value, list) and expected_value:
        if len(expected_value) == 2 and all(
            is_finite_number(x) for x in expected_value
        ):
            # range comparison: [4, 5] means 4 <= actual <= 5
            actual_int = convert_to_int_or_none(actual_value)
            if actual_int is None:
                return False
            range_low, range_high = int(expected_value[0]), int(expected_value[1])
            return range_low <= actual_int <= range_high

        # multiple choice comparison: [4, 5, 6] means actual must be one of these
        valid_choices = {
            int(choice)
            for choice in expected_value
            if is_finite_number(choice)
            or (isinstance(choice, str) and re.fullmatch(r"-?\d+", choice))
        }
        actual_int = convert_to_int_or_none(actual_value)
        return actual_int is not None and actual_int in valid_choices

    # dictionary alternatives comparison
    if isinstance(expected_value, dict) and expected_value:
        choice_set = set()
        for dict_value in expected_value.values():
            if is_finite_number(dict_value):
                choice_set.add(int(dict_value))
            elif isinstance(dict_value, str) and re.fullmatch(
                r"-?\d+", dict_value.strip()
            ):
                choice_set.add(int(dict_value.strip()))
            elif (
                isinstance(dict_value, list)
                and len(dict_value) == 2
                and all(is_finite_number(x) for x in dict_value)
            ):
                range_low, range_high = int(dict_value[0]), int(dict_value[1])
                choice_set.update(range(range_low, range_high + 1))

        actual_int = convert_to_int_or_none(actual_value)
        return (
            (actual_int is not None and actual_int in choice_set)
            if choice_set
            else (expected_value == actual_value)
        )

    # fallback: exact comparison
    return expected_value == actual_value


def normalize_answer_fields(answer_dict, required_fields):
    """Extract and normalize required fields from answer dictionary."""
    normalized = {}
    for field_name in required_fields:
        normalized[field_name] = answer_dict.get(field_name, None)
    return normalized


@metric
def compute_detailed_scores() -> Metric:
    """
    Computes clockbench metrics using the original scoring logic.

    Args:
        scores: list of SampleScore

    Returns:
        dict of clockbench metrics (including per-task accuracy, validity breakdown, time delta)
    """

    def metric_calculator(scores: list[SampleScore]) -> Value:
        # reconstruct all_results structure from sample metadata
        all_results: OrderedDict = OrderedDict()

        for sample_score in scores:
            metadata = sample_score.score.metadata or {}
            sample_id = metadata.get("sample_id", f"sample_{len(all_results)}")

            if "detailed_results" in metadata:
                all_results[sample_id] = metadata["detailed_results"]

        if not all_results:
            return {}

        question_types = ["time", "shift", "angle", "zone"]
        sample_ids = list(all_results.keys())

        # basic accuracy breakdown
        totals = {q_type: {"correct": 0, "total": 0} for q_type in question_types}

        for sample_id in sample_ids:
            for q_type in question_types:
                if q_type in all_results[sample_id]:
                    totals[q_type]["total"] += 1
                    if all_results[sample_id][q_type]["correct"]:
                        totals[q_type]["correct"] += 1

        # validity breakdown (using time as base task)
        base_task = "time"

        valid_total = sum(
            1
            for sample_id in sample_ids
            if all_results[sample_id][base_task]["expected"].get("valid") is True
        )
        invalid_total = sum(
            1
            for sample_id in sample_ids
            if all_results[sample_id][base_task]["expected"].get("valid") is False
        )
        total_correct_base = sum(
            1
            for sample_id in sample_ids
            if all_results[sample_id][base_task]["correct"]
        )
        valid_correct = sum(
            1
            for sample_id in sample_ids
            if (
                all_results[sample_id][base_task]["expected"].get("valid") is True
                and all_results[sample_id][base_task]["correct"]
            )
        )
        invalid_correct = sum(
            1
            for sample_id in sample_ids
            if (
                all_results[sample_id][base_task]["expected"].get("valid") is False
                and all_results[sample_id][base_task]["correct"]
            )
        )

        validity_breakdown: OrderedDict = OrderedDict(
            [
                ("task", base_task),
                ("total_items", len(sample_ids)),
                ("total_correct", total_correct_base),
                (
                    "valid",
                    OrderedDict(
                        [
                            ("correct", valid_correct),
                            ("total", valid_total),
                            (
                                "accuracy",
                                calculate_percentage(valid_correct, valid_total),
                            ),
                        ]
                    ),
                ),
                (
                    "invalid",
                    OrderedDict(
                        [
                            ("correct", invalid_correct),
                            ("total", invalid_total),
                            (
                                "accuracy",
                                calculate_percentage(invalid_correct, invalid_total),
                            ),
                        ]
                    ),
                ),
            ]
        )

        # follow-up questions breakdown
        followup_types = ["shift", "angle", "zone"]

        valid_time_correct_ids = [
            sample_id
            for sample_id in sample_ids
            if all_results[sample_id]["time"]["correct"]
            and all_results[sample_id]["time"]["expected"].get("valid") is True
        ]
        valid_time_correct_count = len(valid_time_correct_ids)

        cond_accuracy: OrderedDict = OrderedDict()
        cond_accuracy["denominator_valid_time_correct"] = valid_time_correct_count

        for followup_type in followup_types:
            correct_count = sum(
                1
                for sample_id in valid_time_correct_ids
                if all_results[sample_id][followup_type]["correct"]
            )
            cond_accuracy[f"{followup_type}_given_valid_time_correct"] = {
                "numerator": correct_count,
                "denominator": valid_time_correct_count,
                "accuracy": calculate_fraction(correct_count, valid_time_correct_count),
            }

        # time delta breakdown
        circular_deltas = []
        excluded_alternatives = 0
        skipped_incomplete = 0

        for sample_id in sample_ids:
            gt_time = all_results[sample_id]["time"]["expected"]
            pred_time = all_results[sample_id]["time"]["got"]

            # leep only valid times
            if gt_time.get("valid") is not True:
                continue

            # exclude alternatives
            if any(
                isinstance(gt_time.get(field), dict)
                for field in ("hours", "minutes", "seconds")
            ):
                excluded_alternatives += 1
                continue

            # skip if already correct
            if all_results[sample_id]["time"]["correct"]:
                continue

            expected_h, expected_m, expected_s = (
                convert_expected_value_to_scalar(gt_time.get("hours")),
                convert_expected_value_to_scalar(gt_time.get("minutes")),
                convert_expected_value_to_scalar(gt_time.get("seconds")),
            )
            pred_h, pred_m, pred_s = (
                convert_predicted_value_to_scalar(pred_time.get("hours")),
                convert_predicted_value_to_scalar(pred_time.get("minutes")),
                convert_predicted_value_to_scalar(pred_time.get("seconds")),
            )

            if None in (expected_h, expected_m, expected_s, pred_h, pred_m, pred_s):
                skipped_incomplete += 1
                continue

            period_hours = determine_clock_period_hours(sample_id, gt_time.get("hours"))
            expected_total_sec = convert_time_to_seconds(
                expected_h, expected_m, expected_s, period_hours
            )
            pred_total_sec = convert_time_to_seconds(
                pred_h, pred_m, pred_s, period_hours
            )

            period_total_sec = period_hours * 3600
            raw_diff = abs(pred_total_sec - expected_total_sec)
            circular_deltas.append(min(raw_diff, period_total_sec - raw_diff))

        avg_delta_sec = (
            round(sum(circular_deltas) / len(circular_deltas), 2)
            if circular_deltas
            else None
        )
        median_delta_sec = (
            round(median(circular_deltas), 2) if circular_deltas else None
        )
        avg_delta_hm = convert_sec_to_hours_min(avg_delta_sec)
        median_delta_hm = convert_sec_to_hours_min(median_delta_sec)

        # invalid predictions breakdown
        predicted_invalid_count = 0
        for sample_id in sample_ids:
            predicted_time = all_results[sample_id]["time"]["got"]
            predicted_validity = (
                predicted_time.get("valid")
                if isinstance(predicted_time, dict)
                else None
            )
            if predicted_validity is False:
                predicted_invalid_count += 1

        predicted_invalid_percentage = (
            round(100 * predicted_invalid_count / len(sample_ids), 2)
            if sample_ids
            else 0.0
        )

        # final scores (following orig structure)
        per_task_accuracy = {
            f"{q_type}_accuracy": round(
                totals[q_type]["correct"] / max(1, totals[q_type]["total"]), 4
            )
            for q_type in question_types
        }

        result_scores = OrderedDict([("per_task_accuracy_abs", per_task_accuracy)])
        result_scores["time_validity_breakdown"] = validity_breakdown
        result_scores["predicted_invalid"] = {
            "count": predicted_invalid_count,
            "percent_of_all_items": predicted_invalid_percentage,
        }
        result_scores["conditional_accuracy_given_valid_answer_time_correct"] = (
            cond_accuracy
        )
        result_scores["time_delta_seconds_on_incorrect_valid_circular"] = OrderedDict(
            [
                ("count_items", len(circular_deltas)),
                ("average_delta_seconds", float(avg_delta_sec or 0.0)),
                ("median_delta_seconds", float(median_delta_sec or 0.0)),
                ("average_delta_hm", avg_delta_hm),
                ("median_delta_hm", median_delta_hm),
                ("excluded_due_to_alternatives", excluded_alternatives),
                ("skipped_incomplete_after_normalization", skipped_incomplete),
            ]
        )

        # extract values for metrics output
        return {
            "time_reading_accuracy": per_task_accuracy.get("time_accuracy", 0.0),
            "shift_accuracy": per_task_accuracy.get("shift_accuracy", 0.0),
            "angle_accuracy": per_task_accuracy.get("angle_accuracy", 0.0),
            "zone_accuracy": per_task_accuracy.get("zone_accuracy", 0.0),
            "predicted_invalid_time_percent": float(predicted_invalid_percentage),
            "average_time_error_seconds": float(avg_delta_sec or 0.0),
            "median_time_error_seconds": float(median_delta_sec or 0.0),
            # validity breakdown metrics (readable vs broken clocks)
            "readable_clocks_accuracy": validity_breakdown["valid"]["accuracy"] or 0.0,
            "broken_clocks_accuracy": validity_breakdown["invalid"]["accuracy"] or 0.0,
            # conditional accuracy (follow-up performance when initial time reading was correct)
            "correct_valid_time": cond_accuracy["denominator_valid_time_correct"],
            "conditional_shift_accuracy": cond_accuracy[
                "shift_given_valid_time_correct"
            ]["accuracy"]
            or 0.0,
            "conditional_angle_accuracy": cond_accuracy[
                "angle_given_valid_time_correct"
            ]["accuracy"]
            or 0.0,
            "conditional_zone_accuracy": cond_accuracy["zone_given_valid_time_correct"][
                "accuracy"
            ]
            or 0.0,
            # time error analysis details
            "predicted_incorrect_time": len(circular_deltas),
            "excluded_multiple_answers": excluded_alternatives,
            "skipped_incomplete_data": skipped_incomplete,
        }

    return metric_calculator


@scorer(metrics=[accuracy(), stderr(), compute_detailed_scores()])
def clockbench_scorer():
    """
    Detailed clockbench scorer that stores full comparison results for sophisticated analysis.

    This scorer:
    1. Parses target fields from metadata and model response from solver output
    2. Uses original compare_gt_pred logic with detailed comparison tracking
    3. Stores full results structure for compute_detailed_scores analysis
    4. Returns sample-level accuracy with detailed metadata
    """

    async def score(state: TaskState, _target: Any) -> Score:
        try:
            # Parse target fields from metadata and model response from solver output
            model_responses = json.loads(state.output.completion)
            target_dict = state.metadata.get("target", {})

            question_types = ["time", "shift", "angle", "zone"]
            detailed_results = OrderedDict()
            per_task_scores = {}

            for question_type in question_types:
                # Extract expected fields based on task type
                task_fields = f"{question_type}_fields"
                required_fields = FIELDS_BY_TASK[task_fields]

                # Get ground truth from metadata and model response from solver output
                gt_data = target_dict.get(question_type, {})
                model_data = model_responses.get(question_type, {})

                # Parse model response - already parsed from JSON
                parsed_model_data = model_data

                # Use original detailed comparison logic
                is_correct, comparison_details = compare_gt_pred(
                    gt_data, parsed_model_data, required_fields
                )

                # Store detailed results in the format expected by compute_detailed_scores
                detailed_results[question_type] = {
                    "expected": gt_data,
                    "got": parsed_model_data,
                    "correct": is_correct,
                    "details": comparison_details,
                }

                # Store per-task score (1.0 for correct, 0.0 for incorrect)
                per_task_scores[question_type] = 1.0 if is_correct else 0.0

            # Overall accuracy for the sample
            sample_accuracy = sum(per_task_scores.values()) / len(question_types)

            # Store sample ID and detailed results
            sample_id = getattr(state, "sample_id", None) or f"sample_{id(state)}"

            return Score(
                value=sample_accuracy,
                metadata={
                    "sample_id": sample_id,
                    "detailed_results": detailed_results,
                    **per_task_scores,  # Include individual task scores in metadata
                },
            )

        except Exception as e:
            return Score(
                value=0.0,
                metadata={
                    "error": str(e),
                    "sample_id": getattr(state, "sample_id", "unknown"),
                },
            )

    return score
