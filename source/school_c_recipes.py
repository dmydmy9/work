"""Candidate training actions for the official C-MAPSS parent checkpoint."""

from copy import deepcopy
import re

import numpy as np

from school_c_kernel import METADATA_KEY, RUL_CAP_KEY, fit_regime_kernel, fit_rul_caps


# Keep batch size and optimizer-step budget equal across the first comparisons.
CAP_RECIPE = "regime-kernel-cap-5"
RECIPES = {
    "regime-kernel-5": (5, .0005, False),
    "official-5": (5, .002, False),
    "conservative-5": (5, .0005, False),
    "accelerated-5": (5, .004, False),
    "engine-balanced-5": (5, .002, True),
}
CITY_RECIPES = {**RECIPES, CAP_RECIPE: RECIPES["regime-kernel-5"]}
ENGINE_ID = re.compile(r"(fd00[1-4])_train_(u\d+)_c\d+", re.ASCII)


def group_ids(ids, *, allow_opaque=False):
    engines = []
    for identity in ids:
        match = ENGINE_ID.fullmatch(identity)
        if match is None:
            if allow_opaque:
                return None
            raise ValueError(f"invalid C training identity {identity!r}")
        engines.append("_".join(match.groups()))
    return np.asarray(engines)


def continue_parent(parent, features, targets, groups, recipe, *, epochs=None):
    default_epochs, rate, balance = CITY_RECIPES[recipe]
    if epochs is None:
        epochs = default_epochs
    model = deepcopy(parent)
    model.meta.update(learning_rate=rate, batch_size=256)
    if not balance:
        losses = model.fit(features, targets, epochs)
        if recipe in ("regime-kernel-5", CAP_RECIPE):
            model.meta[METADATA_KEY] = fit_regime_kernel(model, features, targets)
        if recipe == CAP_RECIPE:
            model.meta[RUL_CAP_KEY] = fit_rul_caps(model.meta["paths"], features, targets)
        return model, losses

    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    probabilities = 1.0 / counts[inverse]
    probabilities /= probabilities.sum()
    losses = []
    for _ in range(epochs):
        epoch = np.random.SeedSequence([model.meta["seed"], model.meta["epochs"], 1])
        rng = np.random.default_rng(epoch)
        indices = rng.choice(len(features), len(features), replace=True, p=probabilities)
        losses.extend(model.fit(features[indices], targets[indices], 1))
    return model, losses


def metrics(targets, predictions):
    residual = predictions - targets
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "nasa_score": float(np.expm1(np.where(residual < 0, -residual / 13, residual / 10)).sum()),
    }
