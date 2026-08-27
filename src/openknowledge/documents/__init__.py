"""Turning company files into structured, citable blocks."""

from .blocks import Block, BlockKind, ParsedDocument, normalise, table_row_text
from .registry import (
    LEGACY_SUFFIXES,
    SUPPORTED_SUFFIXES,
    TEXT_SUFFIXES,
    is_supported,
    parse_bytes,
    parse_file,
    skip_reason,
)
from .text import parse_text

__all__ = [
    "LEGACY_SUFFIXES",
    "SUPPORTED_SUFFIXES",
    "TEXT_SUFFIXES",
    "Block",
    "BlockKind",
    "ParsedDocument",
    "is_supported",
    "normalise",
    "parse_bytes",
    "parse_file",
    "parse_text",
    "skip_reason",
    "table_row_text",
]
