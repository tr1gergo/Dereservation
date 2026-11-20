"""Experiments for Efficient De-Reservation in Matching Markets.

The study reuses the spatial generator and reserve benchmarks under ``src/``.
Run the notebook from this directory or import the package from the repository root.
"""

from .generation import generate_indian_market, generate_overlapping_market
from .model import Market, Outcome
from .policies import Policy

__all__ = ["Market", "Outcome", "Policy", "generate_indian_market",
           "generate_overlapping_market"]
