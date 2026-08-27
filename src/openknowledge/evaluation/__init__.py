"""Evaluation harness: accuracy and cost, measured together."""

from .dataset import Case, DatasetError, filter_cases, load_cases, parse_cases
from .report import Comparison, compare, format_report
from .runner import CaseResult, EvalReport, run_case, run_eval

__all__ = [
    "Case",
    "CaseResult",
    "Comparison",
    "DatasetError",
    "EvalReport",
    "compare",
    "filter_cases",
    "format_report",
    "load_cases",
    "parse_cases",
    "run_case",
    "run_eval",
]
