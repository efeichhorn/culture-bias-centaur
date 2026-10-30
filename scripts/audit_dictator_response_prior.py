#!/usr/bin/env python3
"""Fit and score the allocation response adjustment used for H4.

One response offset is shared across all social targets. Fit the offsets on
Giving training participants and items, then apply them to held-out Giving
and Taking items. The ridge penalty is fixed before evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PARTICIPANT_KEYS = ["Dataset", "Nation", "SubNo"]
COUNTRIES = {0: "United States", 1: "China"}
CONDITIONS = ("us", "china", "pooled")
VALUES = np.arange(21)


def log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def objective_derivatives(
    theta: np.ndarray, base_logp: np.ndarray, counts: np.ndarray, ridge: float
) -> tuple[float, np.ndarray, np.ndarray]:
    """Mean choice CE plus .5 * ridge * squared centered-offset norm."""
    offsets = np.r_[theta, 0.0]
    centered = offsets - offsets.mean()
    logp = log_softmax(base_logp + offsets)
    p = np.exp(logp)
    normalized_counts = counts / counts.sum()
    weights = normalized_counts.sum(axis=1)
    value = -(normalized_counts * logp).sum() + 0.5 * ridge * (centered @ centered)
    full_gradient = (weights[:, None] * p - normalized_counts).sum(axis=0)
    full_gradient += ridge * centered
    k = len(offsets)
    full_hessian = np.diag((weights[:, None] * p).sum(axis=0))
    full_hessian -= np.einsum("i,ij,ik->jk", weights, p, p)
    full_hessian += ridge * (np.eye(k) - np.ones((k, k)) / k)
    return float(value), full_gradient[:-1], full_hessian[:-1, :-1]


def fit_response_offsets(
    base_logp: np.ndarray, counts: np.ndarray, ridge: float = 0.001
) -> tuple[np.ndarray, dict]:
    """Damped Newton optimization of the convex 20-parameter objective."""
    base_logp = np.asarray(base_logp, dtype=float)
    counts = np.asarray(counts, dtype=float)
    if base_logp.shape != counts.shape or base_logp.ndim != 2:
        raise ValueError("Base probabilities and counts must be matching matrices.")
    if not np.isfinite(base_logp).all() or not np.isfinite(counts).all():
        raise ValueError("Inputs must be finite.")
    if np.any(counts < 0) or counts.sum() <= 0 or ridge < 0:
        raise ValueError("Counts must be nonnegative and nonempty; ridge nonnegative.")
    if not np.allclose(np.exp(base_logp).sum(axis=1), 1, atol=1e-8):
        raise ValueError("Base conditional probabilities must sum to one.")
    theta = np.zeros(counts.shape[1] - 1)
    initial = objective_derivatives(theta, base_logp, counts, ridge)[0]
    for iteration in range(100):
        value, gradient, hessian = objective_derivatives(theta, base_logp, counts, ridge)
        if np.max(np.abs(gradient)) < 1e-10:
            break
        direction = np.linalg.solve(hessian, -gradient)
        step = 1.0
        directional_derivative = gradient @ direction
        for _ in range(50):
            proposed = theta + step * direction
            proposed_value = objective_derivatives(proposed, base_logp, counts, ridge)[0]
            if proposed_value <= value + 1e-4 * step * directional_derivative:
                theta = proposed
                break
            step *= 0.5
        else:
            raise RuntimeError("Newton line search did not converge.")
    value, gradient, _ = objective_derivatives(theta, base_logp, counts, ridge)
    if np.max(np.abs(gradient)) > 1e-8:
        raise RuntimeError(f"Response-offset fit failed: gradient={gradient}")
    offsets = np.r_[theta, 0.0]
    offsets -= offsets.mean()
    diagnostics = {
        "converged": True,
        "newton_iterations": iteration,
        "initial_objective": initial,
        "final_objective": value,
        "max_abs_gradient": float(np.max(np.abs(gradient))),
        "training_responses": int(counts.sum()),
        "offset_l2_norm": float(np.linalg.norm(offsets)),
    }
    return offsets, diagnostics


def apply_offsets(candidates: pd.DataFrame, offsets: np.ndarray) -> pd.DataFrame:
    """Preserve base valid-response mass when constructing raw log likelihood."""
    frame = candidates.sort_values(["GroupNo", "response_value"]).copy()
    matrix = frame.pivot(index="GroupNo", columns="response_value", values="conditional_probability")
    if list(matrix.columns) != list(VALUES) or matrix.isna().any().any():
        raise ValueError("Expected 21 complete candidate probabilities per target.")
    calibrated = log_softmax(np.log(matrix.to_numpy()) + offsets)
    frame["conditional_log_probability"] = calibrated.ravel()
    frame["conditional_probability"] = np.exp(calibrated).ravel()
    frame["full_vocab_log_probability"] = (
        frame["conditional_log_probability"] + np.log(frame["valid_response_mass"])
    )
    frame["full_vocab_probability"] = np.exp(frame["full_vocab_log_probability"])
    return frame


def training_counts(
    behavior: pd.DataFrame, selected_participants: pd.DataFrame, groups: list[int]
) -> tuple[np.ndarray, int]:
    """Select by explicit training participant keys and explicit training targets."""
    if selected_participants[PARTICIPANT_KEYS].duplicated().any():
        raise ValueError("Repeated participant in training selection.")
    selected = behavior.loc[behavior.Dataset.isin([1, 2])].merge(
        selected_participants[PARTICIPANT_KEYS], on=PARTICIPANT_KEYS, validate="many_to_one"
    )
    selected = selected.loc[selected.GroupNo.isin(groups)]
    counts_per_person = selected.groupby(PARTICIPANT_KEYS).size()
    if not counts_per_person.eq(len(groups)).all() or not len(counts_per_person):
        raise ValueError("Each training participant must have every selected target.")
    counts = pd.crosstab(selected.GroupNo, selected.response_value).reindex(
        index=groups, columns=VALUES, fill_value=0
    )
    return counts.to_numpy(dtype=float), len(counts_per_person)


def score(humans: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    probabilities = candidates[["GroupNo", "response_value", "conditional_probability", "full_vocab_log_probability"]]
    if probabilities.duplicated(["GroupNo", "response_value"]).any():
        raise ValueError("OOF predictions contain repeated targets or responses.")
    merged = humans.merge(probabilities, on=["GroupNo", "response_value"], validate="many_to_one")
    if len(merged) != len(humans):
        raise ValueError("Some human responses have no matching prediction.")
    merged["conditional_ll"] = np.log(merged.conditional_probability)
    participant = merged.groupby(PARTICIPANT_KEYS, as_index=False).agg(
        conditional_ll=("conditional_ll", "mean"),
        raw_response_ll=("full_vocab_log_probability", "mean"),
        responses=("GroupNo", "size"),
    )
    if not participant.responses.eq(25).all():
        raise ValueError("OOF evaluation must score all 25 targets per participant.")
    participant["country"] = participant.Nation.map(COUNTRIES)
    return participant
