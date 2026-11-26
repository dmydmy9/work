"""B continuation updates that retain the official checkpoint representation."""
from collections import Counter
from copy import deepcopy
import re

import numpy as np


def affine_readout(parent, x, y, weights, penalty=3.0):
    """Fit an anchored gain and offset of the current readout in normalized target units.

    Old batching and Adam state remain available. These are direct least-squares
    updates to the output tensors, with no extra Adam steps.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != y.shape or not np.all(np.isfinite(weights) & (weights > 0)) or penalty <= 0:
        raise ValueError("B readout requires positive finite weights and penalty")
    _, raw = parent.forward(parent.normalized(x))
    scaled_y = (y - parent.meta["target_mean"]) / parent.meta["target_scale"]
    design = np.column_stack((raw[:, 0], np.ones(len(x))))
    weights = weights / weights.mean()
    normal = design.T @ (weights[:, None] * design) + penalty * np.eye(2)
    target = design.T @ (weights * (scaled_y - raw[:, 0]))
    gain_delta, offset = np.linalg.solve(normal, target)
    updated = deepcopy(parent)
    gain = 1 + float(gain_delta)
    updated.params["w2"] *= gain
    updated.params["b2"] = updated.params["b2"] * gain + float(offset)
    updated.validate()
    return updated, {"gain": gain, "offset_normalized_units": float(offset), "normal_equations": 1}


def refine_readout(parent, x, y, weights, penalty=3.0, learning_rate=0.0001):
    """Use an analytic warm start and one exact weighted Adam update on the readout."""
    candidate, fit = affine_readout(parent, x, y, weights, penalty)
    hidden, logits = candidate.forward(candidate.normalized(x))
    labels = (y - candidate.meta["target_mean"]) / candidate.meta["target_scale"]
    weights = np.asarray(weights, dtype=np.float64)
    probability = weights / weights.sum()
    error = logits[:, 0] - labels
    gradient = 2 * error * probability
    directions = {"w2": hidden.T @ gradient[:, None], "b2": np.array([gradient.sum()])}
    norm = np.sqrt(sum(np.sum(value ** 2) for value in directions.values()))
    clip = min(1, 5 / max(norm, 1e-12))
    step = candidate.meta["steps"] + 1
    for name, value in directions.items():
        value *= clip
        candidate.first[name] = .9 * candidate.first[name] + .1 * value
        candidate.second[name] = .999 * candidate.second[name] + .001 * value ** 2
        first = candidate.first[name] / (1 - .9 ** step)
        second = candidate.second[name] / (1 - .999 ** step)
        candidate.params[name] -= learning_rate * first / (np.sqrt(second) + 1e-8)
    candidate.meta.update(steps=step, epochs=parent.meta["epochs"] + 1)
    candidate.validate()
    observation = {**fit, "epoch": candidate.meta["epochs"], "step": step, "learning_rate": learning_rate,
                   "loss": float(np.average(error ** 2, weights=weights)), "frozen_layers": ["w1", "b1"]}
    return candidate, [observation]


def group_regression_gradients(model, x, hidden, derivative):
    hidden_derivative = (derivative @ model.params["w2"].T) * (1 - hidden**2)
    return {"w1": x.T @ hidden_derivative, "b1": hidden_derivative.sum(axis=0),
            "w2": hidden.T @ derivative, "b2": derivative.sum(axis=0)}


def group_adam_step(model, gradients, rate):
    norm = np.sqrt(sum(np.sum(values**2) for values in gradients.values()))
    if not np.isfinite(norm):
        raise ValueError("B group-centered update has a nonfinite gradient")
    clip = min(1, 5 / max(float(norm), 1e-12))
    model.meta["steps"] += 1
    count = model.meta["steps"]
    for key, gradient in gradients.items():
        gradient *= clip
        model.first[key] = .9 * model.first[key] + .1 * gradient
        model.second[key] = .999 * model.second[key] + .001 * gradient**2
        first = model.first[key] / (1 - .9**count)
        second = model.second[key] / (1 - .999**count)
        model.params[key] -= rate * first / (np.sqrt(second) + 1e-8)
    model.meta["epochs"] += 1
    return count


def late_phase_mass(target, codes, counts, *, emphasis, power):
    """Weight labeled fitting rows by the late fraction of their own trajectory."""
    if not 0 <= emphasis < 1 or not 0 < power <= 4:
        raise ValueError("invalid B phase weighting")
    maximum = np.zeros(len(counts))
    np.maximum.at(maximum, codes, target)
    late = 1 - target / np.maximum(maximum[codes], 1.)
    return 1 - emphasis + emphasis * late**power


def within_bearing_pairs(target, groups, fractions):
    """Select fitting-only minute differences at proportional trajectory intervals."""
    groups = np.asarray(groups)
    first, second, mass = [], [], []
    for name in np.unique(groups):
        ids = np.flatnonzero(groups == name)
        if len(ids) < 2:
            raise ValueError("B pair update requires at least two snapshots per fitting bearing")
        ordered = ids[np.argsort(target[ids], kind="stable")]
        for fraction in fractions:
            delta = max(1, min(len(ids) - 1, round(fraction * len(ids))))
            index = np.unique(np.linspace(0, len(ids) - delta - 1, min(384, len(ids) - delta)).astype(int))
            first.extend(ordered[index])
            second.extend(ordered[index + delta])
            mass.extend(np.full(len(index), np.sqrt(len(ids)) / (len(fractions) * len(index))))
    weights = np.asarray(mass)
    return np.asarray(first), np.asarray(second), weights / weights.sum()


def paired_minute_steps(parent, features, y, first, second, weights, *, rate, bound, checkpoints):
    model = deepcopy(parent)
    x = parent.normalized(features)
    target = (y[second] - y[first]) / parent.meta["target_scale"]
    model.meta["learning_rate"] = rate
    for step in range(1, max(checkpoints) + 1):
        hidden, values = model.forward(x)
        error = values[second, 0] - values[first, 0] - target
        contribution = 2 * weights * np.clip(error, -bound, bound)
        rows = (np.bincount(second, weights=contribution, minlength=len(x))
                - np.bincount(first, weights=contribution, minlength=len(x)))
        group_adam_step(model, group_regression_gradients(model, x, hidden, rows[:, None]), rate)
        if step in checkpoints:
            model.validate()
            yield step, model


def continue_group_adam(parent, features, target, codes, counts, center_weight, rate, checkpoints, *,
                        anchor=0., sample_mass=None, group_power=.5):
    """Continue Adam with a separate weight on each bearing's mean squared error."""
    model = deepcopy(parent)
    x = model.normalized(features)
    y = (target - model.meta["target_mean"]) / model.meta["target_scale"]
    row_weight = counts[codes].astype(float) ** -group_power
    if sample_mass is not None:
        sample_mass = np.asarray(sample_mass, dtype=float)
        if sample_mass.shape != y.shape or not np.all(np.isfinite(sample_mass) & (sample_mass > 0)):
            raise ValueError("B fitting weights must be finite positive row masses")
        mass_by_group = np.bincount(codes, weights=sample_mass, minlength=len(counts))
        row_weight *= sample_mass * counts[codes] / mass_by_group[codes]
    row_weight /= row_weight.sum()

    def mean_by_group(values):
        if sample_mass is None:
            return np.bincount(codes, weights=values) / counts
        return np.bincount(codes, weights=sample_mass * values, minlength=len(counts)) / mass_by_group

    if anchor:
        initial = model.forward(x)[1][:, 0]
        initial_mean = mean_by_group(initial)
    model.meta["learning_rate"] = rate
    for i in range(1, max(checkpoints) + 1):
        hidden, output = model.forward(x)
        error = output[:, 0] - y
        offset = mean_by_group(error)[codes]
        gradient_error = error + (center_weight - 1) * offset
        if anchor:
            current_mean = mean_by_group(output[:, 0])
            gradient_error += anchor * (current_mean - initial_mean)[codes]
        derivative = (2 * row_weight * gradient_error)[:, None]
        gradients = group_regression_gradients(model, x, hidden, derivative)
        count = group_adam_step(model, gradients, rate)
        if i in checkpoints:
            model.validate()
            observation = {"step": count, "epoch": model.meta["epochs"], "learning_rate": rate,
                           "within_bearing_mse": float(row_weight @ (error - offset)**2),
                           "mean_bearing_mse": float(row_weight @ offset**2)}
            yield i, model, observation



def training_bearings(rows):
    """Use authenticated B IDs to identify fitting groups only."""
    groups = []
    for row in rows:
        identity = row.get("id")
        if not isinstance(identity, str):
            return None
        match = re.fullmatch(r"xjtu_(Bearing[1-3]_[1-3])_s[0-9]{6}", identity)
        if match is None:
            return None
        bearing = match[1]
        metadata = row.get("meta", {})
        if not isinstance(metadata, dict) or metadata.get("bearing", bearing) != bearing:
            return None
        groups.append(bearing)
    return groups if len(set(groups)) >= 2 else None


def training_bearing_weights(rows):
    groups = training_bearings(rows)
    if groups is None:
        return None
    counts = Counter(groups)
    return np.asarray([1.0 / counts[name] for name in groups])
