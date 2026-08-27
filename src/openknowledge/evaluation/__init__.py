"""Evaluation harness: accuracy and cost, measured together."""

from .conflicts import (
    ConflictCase,
    ConflictReport,
    ConflictSetError,
    format_conflict_report,
    load_conflict_cases,
    parse_conflict_cases,
    run_conflict_eval,
)
from .dataset import Case, DatasetError, filter_cases, load_cases, parse_cases
from .report import Comparison, compare, format_report
from .runner import CaseResult, EvalReport, run_case, run_eval

__all__ = [
    "Case",
    "ConflictCase",
    "ConflictReport",
    "ConflictSetError",
    "CaseResult",
    "Comparison",
    "DatasetError",
    "EvalReport",
    "compare",
    "format_conflict_report",
    "load_conflict_cases",
    "parse_conflict_cases",
    "run_conflict_eval",
    "filter_cases",
    "format_report",
    "load_cases",
    "parse_cases",
    "run_case",
    "run_eval",
]
