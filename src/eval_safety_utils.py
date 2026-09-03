"""
eval_safety_utils.py
====================
Shared robustness utilities for EconSafeBench static + dynamic evaluation.

Why this exists
---------------
A dead API token, an exhausted quota, or a runaway recursion loop must never
be silently scored as a compliant decision — that would inflate safety. This
module centralises three guarantees:
  1. ``is_agent_error``  — an errored period/case is NEVER scored as a decision.
  2. ``CircuitBreaker``  — N consecutive errors abort the run cleanly (don't grind
     for hours against a dead token).
  3. ``run_validity`` / ``logs_validity`` — a run that is mostly errors is
     flagged INVALID and excluded from metrics rather than scored on the
     surviving handful of periods.
"""
from __future__ import annotations

from typing import Any

ERROR_PREFIX = "__ERROR__"
# Fraction of errored periods/cases above which a run is considered untrustworthy.
DEFAULT_MAX_ERROR_FRAC = 0.20
# Consecutive agent errors that trip the circuit breaker (≈ quota death / dead token).
DEFAULT_BREAKER_LIMIT = 5


def is_agent_error(resp: Any) -> bool:
    """True iff an agent response is a scaffold/API failure rather than a real
    decision.  Covers both the string returned by run_agent on max-retries and
    the legacy '[LG error]' marker some simulators injected on exception."""
    s = str(resp)
    return (
        s.startswith(ERROR_PREFIX)
        or "[LG error]" in s
        or "max retries exceeded" in s
    )


class CircuitBreaker:
    """Trip after ``limit`` consecutive errored agent calls.

    Usage:
        cb = CircuitBreaker()
        for period in ...:
            resp = call_agent(...)
            if cb.record(is_agent_error(resp)):
                print(cb.message()); break   # abort this run cleanly
    """

    def __init__(self, limit: int = DEFAULT_BREAKER_LIMIT):
        self.limit = limit
        self.consecutive = 0
        self.tripped = False

    def record(self, errored: bool) -> bool:
        """Register one outcome; return True once the breaker has tripped."""
        self.consecutive = self.consecutive + 1 if errored else 0
        if self.consecutive >= self.limit:
            self.tripped = True
        return self.tripped

    def message(self) -> str:
        return (
            f"⛔ CIRCUIT BREAKER: {self.consecutive} consecutive agent errors "
            f"(>= {self.limit}). Aborting this run — likely token/quota exhausted. "
            f"Check API balance before relaunching."
        )


def run_validity(error_periods: int,
                 total_periods: int,
                 breaker_tripped: bool = False,
                 max_error_frac: float = DEFAULT_MAX_ERROR_FRAC) -> dict:
    """Summarise whether a completed run is trustworthy.

    A run is INVALID if it produced no periods, if the breaker tripped, or if
    too large a fraction of periods errored. Downstream metrics/DPA should be
    suppressed (reported as N/A) for invalid runs rather than computed on the
    surviving handful of periods.
    """
    bad = (error_periods / total_periods) if total_periods else 1.0
    valid = (total_periods > 0) and (not breaker_tripped) and (bad <= max_error_frac)
    return {
        "valid": valid,
        "total_periods": total_periods,
        "error_periods": error_periods,
        "error_frac": round(bad, 3),
        "breaker_tripped": breaker_tripped,
    }


def log_is_error(h: Any) -> bool:
    """True if a stored period-log entry represents an errored (uncalled) period.
    Such entries carry None scores and an __ERROR__ agent response, or an
    explicit marker. They must be excluded from metrics and counted for validity.
    """
    if not isinstance(h, dict):
        return False
    if h.get("errored") is True:
        return True
    if h.get("recommended") == "ERROR":
        return True
    for k in ("agent_response", "full_response", "response"):
        if is_agent_error(h.get(k, "")):
            return True
    return False


def logs_validity(logs: Any,
                  n_periods: int,
                  max_error_frac: float = DEFAULT_MAX_ERROR_FRAC) -> dict:
    """Validity of a (model, seed, condition) combo from its list of period logs.

    A combo is VALID only if it ran to completion (>= n_periods entries) and the
    error fraction is acceptable.
    """
    if not isinstance(logs, list) or not logs:
        return {"valid": False, "n": 0, "errors": 0,
                "error_frac": 1.0, "complete": False}
    errs = sum(1 for h in logs if log_is_error(h))
    n = len(logs)
    complete = n >= n_periods
    frac = (errs / n) if n else 1.0
    return {"valid": complete and frac <= max_error_frac,
            "n": n, "errors": errs, "error_frac": round(frac, 3),
            "complete": complete}


