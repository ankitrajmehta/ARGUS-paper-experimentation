"""Model-agnostic types.

Kept free of torch so the CLI, scoring and dry-run path import without a GPU
stack installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class Reply:
    """One model response to a DriveLM scene."""

    long: str  # first-turn chain-of-thought
    short: str  # second-turn one-word summary
    action: str | None  # parsed action ("a)"/"b)"), or None if unreadable/ambiguous
    retried: bool = False


@runtime_checkable
class DriveLMAgent(Protocol):
    """What run_eval needs from a model."""

    def ask(self, images: list[np.ndarray], retry_on_unreadable: bool = True) -> Reply: ...
