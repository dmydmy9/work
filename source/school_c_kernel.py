"""Fit an operating-regime correction using only C training rows."""

import numpy as np


SETTINGS_PATH = '[["key","settings"],["index",0]]'
METADATA_KEY = "c_regime_kernel_v1"
FEATURE_COUNT = 96


def linear_basis(model, x):
    _, normalized_rul = model.forward(model.normalized(x))
    return np.column_stack((np.ones(len(x)), normalized_rul[:, 0]))


def kernel_basis(x, center, scale, active, frequencies, phase):
    local = np.clip((x[:, active] - center) / scale, -6, 6)
    kernel = np.cos(local @ frequencies + phase)
    return kernel * np.sqrt(2 / kernel.shape[1])


def fit_regime_kernel(model, x, truth, *, width=.7, penalty=.004, strength=1.):
    paths = model.meta["paths"]
    regimes = np.rint(x[:, paths.index(SETTINGS_PATH)]).astype(int)
    sensors = np.asarray([index for index, path in enumerate(paths) if '["key","sensors"]' in path])
    if not len(sensors):
        raise ValueError("C sensor paths missing")
    residual = truth - model.predict(x)
    linear = linear_basis(model, x)
    corrections = {}
    for regime in np.unique(regimes):
        selected = regimes == regime
        if selected.sum() < 256:
            continue
        rows = x[selected]
        sensor_values = rows[:, sensors]
        quantiles = np.quantile(sensor_values, (.25, .50, .75), axis=0)
        spread = quantiles[2] - quantiles[0]
        varies = spread > 1e-9
        if not varies.any():
            continue
        active = sensors[varies]
        center, scale = quantiles[1, varies], spread[varies]
        rng = np.random.default_rng(np.random.SeedSequence([661, int(regime) & 0xffffffff]))
        indices = rng.integers(len(rows), size=(1024, 2))
        sample = np.clip((sensor_values[:, varies] - center) / scale, -6, 6)[indices]
        median = float(np.median(np.linalg.norm(sample[:, 0] - sample[:, 1], axis=1)))
        frequencies = rng.normal(0, 1 / max(median * width, 1e-3), size=(len(active), FEATURE_COUNT))
        phase = rng.uniform(0, 2 * np.pi, FEATURE_COUNT)
        design = np.column_stack((linear[selected], kernel_basis(rows, center, scale, active, frequencies, phase)))
        gram = design.T @ design + len(design) * penalty * np.eye(design.shape[1])
        coefficients = np.linalg.solve(gram, design.T @ residual[selected])
        corrections[str(regime)] = {"active": active.tolist(), "center": center.tolist(), "scale": scale.tolist(),
                                    "frequencies": frequencies.tolist(), "phase": phase.tolist(),
                                    "coefficients": coefficients.tolist()}
    return {"strength": strength, "regimes": corrections}


def compile_regime_kernel(model, metadata=None):
    if metadata is None:
        metadata = model.meta.get(METADATA_KEY)
    if metadata is None:
        return None
    strength = metadata["strength"]
    if type(strength) not in (int, float) or not 0 < strength <= 1:
        raise ValueError("invalid C regime correction strength")
    compiled = {}
    paths = model.meta["paths"]
    for key, data in metadata["regimes"].items():
        active = data["active"]
        if (not active or len(set(active)) != len(active) or any(type(i) is not int or not 0 <= i < len(paths)
                or '["key","sensors"]' not in paths[i] for i in active)):
            raise ValueError("invalid C regime correction sensor indices")
        arrays = [np.asarray(data[name], dtype=np.float64)
                  for name in ("center", "scale", "frequencies", "phase", "coefficients")]
        width = len(active)
        expected = ((width,), (width,), (width, FEATURE_COUNT), (FEATURE_COUNT,), (FEATURE_COUNT + 2,))
        if any(values.shape != shape or not np.isfinite(values).all() for values, shape in zip(arrays, expected, strict=True)):
            raise ValueError("invalid C regime correction weights")
        if (arrays[1] <= 0).any() or int(key) in compiled:
            raise ValueError("invalid C regime correction normalization or regime key")
        compiled[int(key)] = (np.asarray(active, dtype=np.intp), *arrays)
    return strength, paths.index(SETTINGS_PATH), compiled


def predict_regime_kernel(model, x, compiled):
    predicted = model.predict(x)
    if compiled is None:
        return predicted
    strength, settings, corrections = compiled
    regimes = np.rint(x[:, settings]).astype(int)
    linear = linear_basis(model, x)
    for regime in np.unique(regimes):
        if regime not in corrections:
            continue
        active, center, scale, frequencies, phase, coefficients = corrections[regime]
        selected = regimes == regime
        design = np.column_stack((linear[selected], kernel_basis(x[selected], center, scale, active, frequencies, phase)))
        predicted[selected] += strength * np.clip(design @ coefficients, -60, 60)
    return np.maximum(0, predicted)
