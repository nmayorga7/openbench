"""Common, reusable metrics."""

from openbench.metrics.grouped import grouped
from openbench.metrics.simpleqa import simpleqa_metrics
from openbench.metrics.drop import drop_metrics, get_drop_metrics
from openbench.metrics.mmlu import (
    category_accuracy_metrics as mmlu_category_accuracy_metrics,
    SUBJECT_TO_CATEGORY,
)
from openbench.metrics.mrcr import mrcr_metrics, OPENAI_MRCR_BINS
from openbench.metrics.mmlu_pro import (
    category_accuracy_metrics as mmlu_pro_category_accuracy_metrics,
)
from openbench.metrics.mgsm import language_accuracy
from openbench.metrics.json_schema import (
    json_validity,
    schema_compliance,
    api_success_rate,
)
from openbench.metrics.hle import hle_metrics
from openbench.metrics.healthbench import healthbench_metrics
from openbench.metrics.graphwalks import (
    graphwalks_metrics,
    graphwalks_token_counts,
    GRAPHWALKS_BINS,
)
from openbench.metrics.cti_bench import (
    technique_precision,
    technique_recall,
    technique_f1,
    exact_match_accuracy,
    mean_absolute_deviation,
    accuracy_within_threshold,
)
from openbench.metrics.scicode import sub_problem_correctness
from openbench.metrics.clockbench import compute_detailed_scores

__all__ = [
    # Common metrics
    "grouped",
    # SimpleQA metrics
    "simpleqa_metrics",
    # DROP metrics
    "drop_metrics",
    "get_drop_metrics",
    # MMLU metrics
    "mmlu_category_accuracy_metrics",
    "SUBJECT_TO_CATEGORY",
    # MRCR metrics
    "mrcr_metrics",
    "OPENAI_MRCR_BINS",
    # MMLU Pro metrics
    "mmlu_pro_category_accuracy_metrics",
    # MGSM metrics
    "language_accuracy",
    # JSON Schema metrics
    "json_validity",
    "schema_compliance",
    "api_success_rate",
    # HLE metrics
    "hle_metrics",
    # HealthBench metrics
    "healthbench_metrics",
    # GraphWalks metrics
    "graphwalks_metrics",
    "graphwalks_token_counts",
    "GRAPHWALKS_BINS",
    # CTI-Bench metrics
    "technique_precision",
    "technique_recall",
    "technique_f1",
    "exact_match_accuracy",
    "mean_absolute_deviation",
    "accuracy_within_threshold",
    # SciCode metrics
    "sub_problem_correctness",
    # ClockBench metrics
    "compute_detailed_scores",
]
