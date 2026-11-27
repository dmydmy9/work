"""Train-only vibration transfer and independent prediction for the school B model."""
from base64 import b64decode, b64encode
from copy import deepcopy
from functools import lru_cache
import json
import math
import zlib

import numpy as np

from school_b_adaptation import continue_group_adam, late_phase_mass, paired_minute_steps, refine_readout
from school_b_adaptation import group_adam_step, group_regression_gradients, within_bearing_pairs
from school_b_histograms import fit_histogram_boost, predict_histogram_boost

AMPLITUDE = {"mean_abs", "peak", "peak_to_peak", "rms"}
GAIN_PAIRS = ((.5,.5),(.75,.75),(1.33,1.33),(2.,2.),(.5,2.),(2.,.5),(.75,1.33),(1.33,.75))
SCHEMA = "vibration-transfer-v1"
MIN_TRANSFER_SNAPSHOTS = 6
B_ANCHOR_KEY = "school_b_transfer_anchor"
B_SHORT_PROTECTION_KEY = "school_b_short_protection"
B_TRANSFER_DOSE_KEY = "school_b_transfer_dose"
BAND_ENERGIES = ("band1_energy_0_2k", "band2_energy_2k_6k", "band3_energy_6k_12k")


@lru_cache(maxsize=4)
def channel_layout(paths):
    layout = {}
    for i, path in enumerate(paths):
        parsed = json.loads(path)
        layout[(parsed[-2][1], parsed[-1][1])] = i
    swapped = [layout[("vertical" if c=="horizontal" else "horizontal", f)] for c, f in layout]
    amplitude = tuple(([i for (c,f), i in layout.items() if c==channel and f in AMPLITUDE],
                       [i for (c,f), i in layout.items() if c==channel and f.startswith("band") and "energy" in f])
                      for channel in ("horizontal","vertical"))
    return layout, swapped, amplitude


@lru_cache(maxsize=4)
def spectral_shape_columns(paths):
    layout, _, _ = channel_layout(paths)
    return tuple(layout[axis, key] for axis in ("horizontal", "vertical")
                 for key in (*BAND_ENERGIES, "spectral_centroid", "spectral_rms"))


def spectral_band_response(x, paths, horizontal, vertical):
    """Reweight summary bands at fixed total power and approximate the moment change."""
    outputs = x.copy()
    layout, _, _ = channel_layout(tuple(paths))
    frequency = np.asarray((1000., 4000., 9400.))
    for channel, gain in (("horizontal", horizontal), ("vertical", vertical)):
        columns = [layout[channel, name] for name in BAND_ENERGIES]
        prior = x[:, columns]
        total = prior.sum(axis=1, keepdims=True)
        altered = prior * np.asarray(gain)
        altered *= np.divide(total, altered.sum(axis=1, keepdims=True), out=np.ones_like(total), where=total > 0)
        outputs[:, columns] = altered
        index1 = layout[channel, "spectral_centroid"]
        index2 = layout[channel, "spectral_rms"]
        first = np.divide(altered @ frequency, prior @ frequency, out=np.ones(len(x)), where=total[:, 0] > 0)
        second = np.divide(altered @ frequency**2, prior @ frequency**2, out=np.ones(len(x)), where=total[:, 0] > 0)
        outputs[:, index1] = np.clip(x[:, index1] * first, 0., 12800.)
        outputs[:, index2] = np.clip(x[:, index2] * np.sqrt(second), outputs[:, index1], 12800.)
    return outputs


def dimensionless(x, paths):
    names, _, _ = channel_layout(tuple(paths))

    def col(c, f):
        return x[:, names[c, f]]

    def log(values):
        return np.log(np.maximum(values, 1e-10))

    fields = []
    for channel in ("horizontal","vertical"):
        energies = np.column_stack([col(channel, name) for name in
                                    ("band1_energy_0_2k","band2_energy_2k_6k","band3_energy_6k_12k")])
        fields.extend((energies / np.maximum(energies.sum(axis=1, keepdims=True),1e-10)).T)
        for a,d in (("spectral_centroid","spectral_rms"),("dominant_freq","spectral_rms"),
                    ("peak_to_peak","rms"),("mean_abs","rms")):
            fields.append(log(col(channel,a))-log(col(channel,d)))
        fields.extend(col(channel,f) for f in
                      ("crest_factor","impulse_factor","shape_factor","spectral_kurtosis","skewness"))
    for name in ("rms","peak","mean_abs","dominant_freq","spectral_centroid","spectral_rms"):
        fields.append(log(col("horizontal",name))-log(col("vertical",name)))
    return np.column_stack(fields)


def gain_adjusted(x, amplitude, gains):
    transformed = x.copy()
    for gain, (columns, energy) in zip(gains, amplitude, strict=True):
        transformed[:, columns] *= gain
        transformed[:, energy] *= gain**2
    return transformed


def reciprocal_gain_views(model, x, relative_gain):
    if not 1. < relative_gain <= 2.:
        raise ValueError("invalid B training channel gain")
    _, _, amplitude = channel_layout(tuple(model.meta["paths"]))
    return (x, gain_adjusted(x, amplitude, (1 / relative_gain, relative_gain)),
            gain_adjusted(x, amplitude, (relative_gain, 1 / relative_gain)))


def physical_votes(parent, x):
    _, swapped, amplitude = channel_layout(tuple(parent.meta["paths"]))
    votes = [parent.predict(x), parent.predict(x[:, swapped])]
    for gains in GAIN_PAIRS:
        votes.append(parent.predict(gain_adjusted(x, amplitude, gains)))
    return np.asarray(votes)


def physical_predict(parent, x):
    return physical_votes(parent, x).mean(axis=0)


def standardize(x, knee, center, spread):
    return np.clip((np.arcsinh(x/knee)-center)/spread,-5,5)


def book_distance(x, books, codes=None, book_codes=None):
    output = np.full(len(x), np.inf)
    for i, book in enumerate(books):
        if len(book) < MIN_TRANSFER_SNAPSHOTS:
            raise ValueError("bearing feature book is too small")
        norm = np.einsum("ij,ij->i",book,book)
        for start in range(0,len(x),256):
            rows = x[start:start+256]
            distances = np.maximum(0,np.einsum("ij,ij->i",rows,rows)[:,None] + norm - 2*rows @ book.T)
            lowest = np.partition(distances,5,axis=1)[:,:6].mean(axis=1)
            if codes is not None:
                lowest[codes[start:start+len(rows)]==book_codes[i]]=np.inf
            output[start:start+len(rows)] = np.minimum(output[start:start+len(rows)],lowest)
    return np.log1p(output)


GATE_LEVELS = np.linspace(0, 1, 257)


def bearing_median(target, groups):
    _, codes, counts = np.unique(groups, return_inverse=True, return_counts=True)
    order = np.argsort(target, kind="stable")
    mass = 1. / counts[codes[order]]
    index = int(np.searchsorted(np.cumsum(mass), mass.sum() * .5))
    return float(target[order[index]])


def fit_conditional_shrink(target, groups, *, quantile, width, retain):
    """Fit priors and the crossover from the labeled fitting projection only."""
    values = np.log1p(target)
    return {"low": float(np.log1p(bearing_median(target, groups))),
            "high": float(np.log1p(np.mean(target))), "boundary": float(np.quantile(values, quantile)),
            "width": width, "retain": retain}


def conditional_shrink(prediction, configuration):
    measured = np.log1p(prediction)
    logit = (configuration["boundary"] - measured) / configuration["width"]
    high = 1 / (1 + np.exp(np.clip(logit, -30, 30)))
    prior = (1 - high) * configuration["low"] + high * configuration["high"]
    retain = configuration["retain"]
    return np.expm1(retain * measured + (1 - retain) * prior)


def short_protected_prediction(anchor, candidate, boundary):
    life = np.clip(anchor / max(boundary, 1.), 0, 1)
    return anchor + (.4 + .6 * life) * (candidate - anchor)


def store_transfer_anchor(candidate, parent):
    if B_ANCHOR_KEY not in candidate.meta:
        candidate.meta[B_ANCHOR_KEY] = {name: tensor.tolist() for name, tensor in parent.params.items()}


def fit_short_protection(targets, groups):
    """Use the short-life protection if the fitting window lacks failure endpoints."""
    cohorts = [targets[groups == group] for group in np.unique(groups)]
    if any(rows.min() < 1. for rows in cohorts):
        return None
    return float(np.median([np.median(rows) for rows in cohorts]))


def anchor_prediction(model, x):
    weights = {name: np.asarray(values) for name, values in model.meta[B_ANCHOR_KEY].items()}
    hidden = np.tanh(model.normalized(x) @ weights["w1"] + weights["b1"])
    logits = hidden @ weights["w2"] + weights["b2"]
    return np.maximum(0., logits[:, 0] * model.meta["target_scale"] + model.meta["target_mean"])


def apply_anchor_protection(model, x, prediction):
    boundary = model.meta.get(B_SHORT_PROTECTION_KEY)
    dose = model.meta.get(B_TRANSFER_DOSE_KEY)
    if boundary is None and dose is None:
        return prediction
    anchor = anchor_prediction(model, x)
    if boundary is not None:
        prediction = short_protected_prediction(anchor, prediction, boundary)
    if dose is not None:
        prediction = anchor + dose * (prediction - anchor)
    return prediction


def empirical_rank(knots, values):
    return np.interp(values, knots, GATE_LEVELS)


def rank_gate(vibration, observed, response, quantile, temperature):
    disagreement = np.quantile(response, GATE_LEVELS)
    risk = np.maximum(empirical_rank(vibration, observed), empirical_rank(disagreement, response))
    return {"novelty": np.asarray(vibration).tolist(), "disagreement": disagreement.tolist(),
            "threshold": float(np.quantile(risk, quantile)), "temperature": temperature}


def fit_transfer(candidate, x, y, groups, *, two_signal=False, parent_normalized=False):
    groups = np.asarray(groups)
    names, group_indices, counts = np.unique(groups,return_counts=True,return_inverse=True)
    if len(names) < 2 or int(counts.min()) < MIN_TRANSFER_SNAPSHOTS:
        raise ValueError("need at least two fitting bearings with six snapshots")
    centered = y - np.array([y[group_indices==i].mean() for i in range(len(names))])[group_indices]
    book_length = min(96, int(counts.min()))
    weights = counts[group_indices].astype(float) ** -.5
    residual = fit_histogram_boost(dimensionless(x,candidate.meta["paths"]), centered,weights,stages=110)
    geometry_input = candidate.normalized(x) if parent_normalized else x
    knee = np.maximum(np.median(np.abs(geometry_input),axis=0),1e-12)
    logx = np.arcsinh(geometry_input/knee)
    center = np.median(logx,axis=0)
    spread = np.maximum(np.quantile(np.abs(logx-center),.75,axis=0),1e-6)
    z = standardize(geometry_input,knee,center,spread)
    books = np.asarray([z[np.random.default_rng(42).choice(np.flatnonzero(group_indices==i),size=book_length,
                        replace=False)] for i in range(len(names))],dtype="<f4")
    observed = book_distance(z,books,group_indices,np.arange(len(names)))
    med,p85,threshold = np.quantile(observed,[.5,.85,.95])
    packed = b64encode(zlib.compress(books.tobytes(order="C"))).decode("ascii")
    config = {"format": SCHEMA, "prior": float(np.mean(y)), "histogram": residual,
              "threshold": float(threshold), "temperature": max(.05, float(p85-med)/4),
              "geometry": [p.tolist() for p in (knee, center, spread)],
              "book_shape": list(books.shape), "book_packed": packed}
    if parent_normalized:
        config["geometry_input"] = "parent-normalized"
    if two_signal or parent_normalized:
        response = physical_votes(candidate, x).std(axis=0)
        vibration = np.quantile(observed, GATE_LEVELS)
        quantile, temperature = (.7, .025) if parent_normalized else (.8, .02)
        config["rank_gate"] = rank_gate(vibration, observed, response, quantile, temperature)
    return config


@lru_cache(maxsize=4)
def unpack_books(payload, shape):
    if len(shape)!=3 or not 2<=shape[0]<=9 or not 6<=shape[1]<=96 or shape[2]!=34:
        raise ValueError("invalid B feature book shape")
    raw_size=math.prod(shape)*4
    decoder=zlib.decompressobj()
    contents=decoder.decompress(b64decode(payload,validate=True),raw_size+1)
    if not decoder.eof or decoder.unused_data or len(contents)!=raw_size:
        raise ValueError("invalid B feature book data")
    result=np.frombuffer(contents,dtype="<f4").reshape(shape).astype(np.float64)
    if not np.all(np.isfinite(result)) or np.max(np.abs(result))>5:
        raise ValueError("invalid B feature book values")
    result.setflags(write=False)
    return result


def check_histogram(hist):
    if not math.isfinite(hist["anchor"]) or not 1 <= len(hist["trees"]) <= 150:
        raise ValueError("invalid B histogram")
    for tree in hist["trees"]:
        if not 1 <= len(tree) <= 23:
            raise ValueError("invalid B tree")
        for index, (feature, value, left, right) in enumerate(tree):
            if not math.isfinite(value) or type(feature) is not int:
                raise ValueError("invalid B split or leaf value")
            good_leaf = feature == -1 and left == right == 0
            good_split = (0 <= feature < 30 and type(left) is type(right) is int
                          and index < left < right < len(tree))
            if not (good_leaf or good_split):
                raise ValueError("invalid B transfer tree topology")


def check_transfer(parent):
    meta=parent.meta
    anchor = meta.get(B_ANCHOR_KEY)
    if anchor is not None:
        if not isinstance(anchor, dict) or set(anchor) != set(parent.params):
            raise ValueError("B transfer anchor tensors missing")
        for name, values in anchor.items():
            tensor = np.asarray(values, dtype=float)
            if tensor.shape != parent.params[name].shape or not np.isfinite(tensor).all():
                raise ValueError("invalid B transfer anchor tensor")
    protection = meta.get(B_SHORT_PROTECTION_KEY)
    if protection is not None and (anchor is None or type(protection) not in (int, float)
                                   or not math.isfinite(protection) or protection < 1.):
        raise ValueError("invalid B short-life protection")
    dose = meta.get(B_TRANSFER_DOSE_KEY)
    if dose is not None and (anchor is None or type(dose) not in (int, float) or not 0 < dose <= 1):
        raise ValueError("invalid B transfer dose")
    strategy=meta.get("school_b_prediction")
    shrink = meta.get("school_b_conditional_shrink")
    if shrink is not None:
        expected = {"low", "high", "boundary", "width", "retain"}
        if (strategy != SCHEMA or not isinstance(shrink, dict) or set(shrink) != expected
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in shrink.values())
                or min(shrink["low"], shrink["high"], shrink["boundary"]) < 0
                or shrink["width"] <= 0 or not 0 <= shrink["retain"] <= 1):
            raise ValueError("invalid B conditional shrinkage")
    if strategy is None or strategy=="physical-average":
        return
    if strategy!=SCHEMA:
        raise ValueError("unknown B prediction strategy")
    conf=meta["school_b_transfer"]
    if conf["format"]!=SCHEMA or not math.isfinite(conf["prior"]) or conf["prior"]<0:
        raise ValueError("invalid B transfer prior")
    if (not math.isfinite(conf["threshold"]) or conf["threshold"]<0
            or not math.isfinite(conf["temperature"]) or conf["temperature"]<=0):
        raise ValueError("invalid B transfer novelty gate")
    geom=np.asarray(conf["geometry"],dtype=float)
    if geom.shape!=(3,34) or not np.all(np.isfinite(geom)) or np.any(geom[[0,2]]<=0):
        raise ValueError("invalid B transfer normalization")
    if conf.get("geometry_input", "raw") not in ("raw", "parent-normalized"):
        raise ValueError("invalid B transfer feature space")
    rank_gate = conf.get("rank_gate")
    if rank_gate is not None:
        for name in ("novelty", "disagreement"):
            observable = np.asarray(rank_gate[name], dtype=float)
            if observable.shape != GATE_LEVELS.shape or not np.all(np.isfinite(observable)):
                raise ValueError("invalid B fitted rank basis")
            if np.any(np.diff(observable) < 0):
                raise ValueError("B rank basis is not ordered")
        if not 0 <= rank_gate["threshold"] <= 1 or not 0 < rank_gate["temperature"] <= 1:
            raise ValueError("invalid B fitted rank gate")
    check_histogram(conf["histogram"])
    offset = conf.get("out_of_bearing_offset_minutes")
    if offset is not None:
        if (shrink is None or rank_gate is None or conf.get("geometry_input") != "parent-normalized"
                or type(offset) not in (int, float) or not math.isfinite(offset)
                or abs(offset) > .35 * conf["prior"] + 1e-9):
            raise ValueError("invalid B out-of-bearing offset")
    spectral = conf.get("spectral_residual")
    if spectral is not None:
        knots = np.asarray(spectral["novelty"], dtype=float)
        if (shrink is None or rank_gate is None or conf.get("geometry_input") != "parent-normalized"
                or knots.shape != GATE_LEVELS.shape or not np.all(np.isfinite(knots))
                or np.any(knots < 0) or np.any(np.diff(knots) < 0) or not 0 <= spectral["threshold"] < 1):
            raise ValueError("invalid B spectral residual gate")
        check_histogram(spectral["histogram"])
    unpack_books(conf["book_packed"],tuple(conf["book_shape"]))


def predict_model(model, x):
    strategy=model.meta.get("school_b_prediction")
    if strategy is None:
        return model.predict(x)
    votes = physical_votes(model, x)
    current = votes.mean(axis=0)
    if strategy == "physical-average":
        return current
    conf=model.meta["school_b_transfer"]
    geometry_input = model.normalized(x) if conf.get("geometry_input") == "parent-normalized" else x
    z=standardize(geometry_input,*np.asarray(conf["geometry"]))
    books=unpack_books(conf["book_packed"],tuple(conf["book_shape"]))
    novelty=book_distance(z,books)
    rank_gate = conf.get("rank_gate")
    if rank_gate is None:
        logit = (novelty - conf["threshold"]) / conf["temperature"]
    else:
        risk = np.maximum(empirical_rank(rank_gate["novelty"], novelty),
                          empirical_rank(rank_gate["disagreement"], votes.std(axis=0)))
        logit = (risk - rank_gate["threshold"]) / rank_gate["temperature"]
    confidence = 1 / (1 + np.exp(np.clip(-logit, -30, 30)))
    vibration=dimensionless(x,model.meta["paths"])
    forecast=np.maximum(0,conf["prior"]+predict_histogram_boost(conf["histogram"],vibration))
    result = current + confidence * (forecast - current)
    shrink = model.meta.get("school_b_conditional_shrink")
    if shrink is None:
        return apply_anchor_protection(model, x, result)
    prediction = conditional_shrink(result, shrink)
    spectral = conf.get("spectral_residual")
    if spectral is not None:
        columns = spectral_shape_columns(tuple(model.meta["paths"]))
        frequency_distance = book_distance(z[:, columns], books[:, :, columns])
        ranks = empirical_rank(spectral["novelty"], frequency_distance)
        selection = ranks > spectral["threshold"]
        if np.any(selection):
            alternative = np.maximum(0, conf["prior"] + predict_histogram_boost(spectral["histogram"], vibration[selection]))
            raw = current[selection] + confidence[selection] * (alternative - current[selection])
            robust = conditional_shrink(raw, shrink)
            weight = (ranks[selection] - spectral["threshold"]) / (1 - spectral["threshold"])
            prediction[selection] += weight * (robust - prediction[selection])
    offset = conf.get("out_of_bearing_offset_minutes")
    if offset is not None:
        prediction = np.maximum(0., prediction + offset)
    return apply_anchor_protection(model, x, prediction)


def update_prediction(parent,x,y,groups,weights,recipe):
    trained, trace=refine_readout(parent,x,y,weights)
    trained.meta.pop("school_b_conditional_shrink", None)
    if recipe in ("transfer-physical", "transfer-two-signal", "transfer-parent-coverage"):
        trained.meta["school_b_prediction"] = SCHEMA
        trained.meta["school_b_transfer"] = fit_transfer(
            trained, x, y, groups, two_signal=recipe == "transfer-two-signal",
            parent_normalized=recipe == "transfer-parent-coverage")
        trace[0]["trained_residual_trees"]=len(trained.meta["school_b_transfer"]["histogram"]["trees"])
    elif recipe=="readout-smoothing":
        trained.meta["school_b_prediction"]="physical-average"
        trained.meta.pop("school_b_transfer",None)
    else:
        raise ValueError("unavailable B prediction adaptation")
    check_transfer(trained)
    return trained,trace


def rebase_group_gate(model, x, codes, counts):
    conf = model.meta["school_b_transfer"]
    normalized = standardize(model.normalized(x), *np.asarray(conf["geometry"]))
    books = unpack_books(conf["book_packed"], tuple(conf["book_shape"]))
    if len(counts) != len(books):
        raise ValueError("B group-centered update requires the original fitting bearings")
    observed = book_distance(normalized, books, codes, np.arange(len(counts)))
    response = physical_votes(model, x).std(axis=0)
    conf["rank_gate"] = rank_gate(conf["rank_gate"]["novelty"], observed, response, .7, .025)
    check_transfer(model)


def update_group_shape(parent, x, y, groups, *, steps=40, rate=.001, relative_gain=1.,
                       late_emphasis=0., late_power=1, group_power=.5):
    original = parent.meta.get("school_b_transfer", {})
    if original.get("geometry_input") != "parent-normalized" or "rank_gate" not in original:
        raise ValueError("B group-centered update requires the inherited coverage model")
    _, codes, counts = np.unique(groups, return_inverse=True, return_counts=True)
    fit_x, fit_y, fit_codes, fit_counts = x, y, codes, counts
    view_count = 1
    if relative_gain != 1.:
        views = reciprocal_gain_views(parent, x, relative_gain)
        view_count = len(views)
        fit_x, fit_y = np.concatenate(views), np.tile(y, view_count)
        fit_codes, fit_counts = np.tile(codes, view_count), counts * view_count
    mass = None
    if late_emphasis:
        mass = np.tile(late_phase_mass(y, codes, counts, emphasis=late_emphasis, power=late_power), view_count)
    _, trained, trace = next(continue_group_adam(
        parent, fit_x, fit_y, fit_codes, fit_counts, 0., rate, (steps,), sample_mass=mass, group_power=group_power))
    if relative_gain != 1.:
        trace.update(relative_gain=relative_gain, training_views=view_count)
    if late_emphasis:
        trace.update(late_emphasis=late_emphasis, late_power=late_power, trajectory_weight_power=group_power)
    rebase_group_gate(trained, x, codes, counts)
    return trained, [trace]


def update_paired_minutes(parent, x, y, groups, *, steps, rate, fractions, relative_gain, bound):
    if "school_b_conditional_shrink" not in parent.meta:
        raise ValueError("B paired update requires a conditional-shrinkage checkpoint")
    _, codes, counts = np.unique(groups, return_inverse=True, return_counts=True)
    first, second, weights = within_bearing_pairs(y, groups, fractions)
    views = reciprocal_gain_views(parent, x, relative_gain)
    view_count = len(views)
    low = np.concatenate([first + i * len(x) for i in range(view_count)])
    high = np.concatenate([second + i * len(x) for i in range(view_count)])
    _, trained = next(paired_minute_steps(parent, np.concatenate(views), np.tile(y, view_count), low, high,
                                         np.tile(weights / view_count, view_count), rate=rate, bound=bound,
                                         checkpoints=(steps,)))
    rebase_group_gate(trained, x, codes, counts)
    trace = {"step": trained.meta["steps"], "epoch": trained.meta["epochs"], "learning_rate": rate,
             "pair_fractions": list(fractions), "pairs": len(low), "training_views": view_count,
             "relative_gain": relative_gain, "huber_bound": bound}
    return trained, [trace]


def duration_mean_weight(y, codes, counts, threshold, low, high):
    """Choose one error weight from the proportion of long fitting trajectories."""
    maximum = np.zeros(len(counts))
    np.maximum.at(maximum, codes, y)
    long = maximum >= threshold
    if not long.any() or long.all():
        return None
    return float(np.where(long, high, low).mean())


def bearing_mean_view(x, codes, counts):
    """Project each fitting row toward its bearing's robustly scaled mean snapshot."""
    scale = np.maximum(np.median(np.abs(x), axis=0), 1e-12)
    values = np.arcsinh(x / scale)
    mean = np.array([values[codes == code].mean(axis=0) for code in range(len(counts))])
    return np.sinh(mean[codes]) * scale


def update_calibrated_duration(parent, x, y, groups, *, steps, rate, temperature, threshold, low, high,
                               spectral_consistency=0., spectral_threshold=.90, slope_weight=1.,
                               bearing_consistency=0.):
    """Continue Adam through the deployed minute forecast, with robust bearing weights."""
    state = parent.meta.get("school_b_transfer", {})
    cfg = parent.meta.get("school_b_conditional_shrink")
    if cfg is None or state.get("geometry_input") != "parent-normalized" or "rank_gate" not in state:
        raise ValueError("B calibrated update requires the inherited calibrated coverage model")
    _, codes, counts = np.unique(groups, return_inverse=True, return_counts=True)
    mean_weight = duration_mean_weight(y, codes, counts, threshold, low, high)
    if mean_weight is None:
        raise ValueError("B calibrated update requires both short and long fitting bearings")
    books = unpack_books(state["book_packed"], tuple(state["book_shape"]))
    if len(books) != len(counts):
        raise ValueError("B calibrated update requires the original fitting bearings")
    normalized = standardize(parent.normalized(x), *np.asarray(state["geometry"]))
    novelty = book_distance(normalized, books, codes, np.arange(len(books)))
    gate = state["rank_gate"]
    risk = np.maximum(empirical_rank(gate["novelty"], novelty),
                      empirical_rank(gate["disagreement"], physical_votes(parent, x).std(axis=0)))
    route = 1 / (1 + np.exp(np.clip((gate["threshold"] - risk) / gate["temperature"], -30, 30)))
    residual = np.maximum(0, state["prior"] + predict_histogram_boost(state["histogram"],
                                                                     dimensionless(x, parent.meta["paths"])))
    _, swap, amplitude = channel_layout(tuple(parent.meta["paths"]))
    views = (x, x[:, swap], *(gain_adjusted(x, amplitude, gains) for gains in GAIN_PAIRS))
    additional = ()
    if spectral_consistency and bearing_consistency:
        raise ValueError("B calibrated update requires one consistency view family")
    if spectral_consistency:
        low_gain, high_gain = (1.4, 1., .75), (.75, 1., 1.4)
        paths = parent.meta["paths"]
        additional = tuple(spectral_band_response(x, paths, a, b) for a, b in
                           ((low_gain, low_gain), (high_gain, high_gain), (low_gain, high_gain), (high_gain, low_gain)))
    if bearing_consistency:
        additional = (bearing_mean_view(x, codes, counts),)
    if additional:
        consistency_mass = np.sqrt(counts[codes]) / counts[codes]
        consistency_mass /= consistency_mass.sum()
    features = parent.normalized(np.concatenate((*views, *additional)))
    scale, mean = parent.meta["target_scale"], parent.meta["target_mean"]
    if slope_weight != 1.:
        group_lifetime = np.bincount(codes, weights=y) / counts
        lifetime = (y - group_lifetime[codes]) / scale
        lifetime_power = np.maximum(1e-12, np.bincount(codes, weights=lifetime**2))
    retained = cfg["retain"]
    gap = (1 - retained) * (cfg["high"] - cfg["low"])
    model = deepcopy(parent)
    model.meta["learning_rate"] = rate
    initial_loss = None
    for _ in range(steps):
        hidden, result = model.forward(features)
        raw = result[:, 0].reshape(len(views) + len(additional), len(x))
        signal = raw[:len(views)] * scale + mean
        physical = np.maximum(0, signal).mean(axis=0)
        predicted = physical + route * (residual - physical)
        logged = np.log1p(predicted)
        high_prior = 1 / (1 + np.exp(np.clip((cfg["boundary"] - logged) / cfg["width"], -700, 700)))
        minutes = np.exp(retained * logged + (1 - retained) * cfg["low"] + gap * high_prior)
        error = (minutes - 1 - y) / scale
        offset = (np.bincount(codes, weights=error) / counts)[codes]
        group_error = (error - offset)**2 + mean_weight * offset**2
        centered_error = error + (mean_weight - 1) * offset
        if slope_weight != 1.:
            coefficients = np.bincount(codes, weights=error * lifetime) / lifetime_power
            slope_error = lifetime * coefficients[codes]
            group_error = (error - offset - slope_error)**2 + slope_weight * slope_error**2 + mean_weight * offset**2
            centered_error += (slope_weight - 1) * slope_error
        loss = np.bincount(codes, weights=group_error) / counts
        if initial_loss is None:
            initial_loss = loss.copy()
        group_scale = np.sqrt(initial_loss + .02)
        change = (loss - initial_loss) / group_scale / temperature
        importance = np.exp(change - change.max()) / group_scale
        mass = importance[codes] / counts[codes]
        mass /= mass.sum()
        slope = minutes / (1 + predicted) * (retained + gap * high_prior * (1 - high_prior) / cfg["width"])
        derivative = (2 * mass / len(views)) * centered_error * (1 - route) * slope
        gradient = (signal > 0) * derivative
        if additional:
            difference = raw[len(views):] - raw[:len(views)].mean(axis=0)
            consistency = (2 * (spectral_consistency + bearing_consistency) / len(additional)) * consistency_mass * difference
            gradient -= consistency.sum(axis=0) / len(views)
            gradient = np.concatenate((gradient, consistency))
        gradient = gradient.reshape(-1, 1)
        group_adam_step(model, group_regression_gradients(model, features, hidden, gradient), rate)
    model.validate()
    rebase_group_gate(model, x, codes, counts)
    trace = {"step": model.meta["steps"], "epoch": model.meta["epochs"], "learning_rate": rate,
             "mean_error_weight": mean_weight, "group_temperature": temperature,
             "duration_threshold_minutes": threshold, "training_views": len(views) + len(additional)}
    if slope_weight != 1.:
        trace["lifetime_slope_weight"] = slope_weight
    if bearing_consistency:
        trace["bearing_consistency"] = bearing_consistency
    if spectral_consistency:
        fit_spectral_residual(model, x, y, codes, counts, normalized, books, spectral_threshold)
        trace.update(spectral_consistency=spectral_consistency, spectral_rank_threshold=spectral_threshold,
                     spectral_residual_trees=80)
        check_transfer(model)
    return model, [trace]


def fit_spectral_residual(model, x, y, codes, counts, normalized, books, threshold):
    paths = model.meta["paths"]
    columns = spectral_shape_columns(tuple(paths))
    novelty = book_distance(normalized[:, columns], books[:, :, columns], codes, np.arange(len(counts)))
    low, high = (1.45, 1., .75), (.75, 1., 1.45)
    crossed = (spectral_band_response(x, paths, low, high), spectral_band_response(x, paths, high, low))
    target = y - (np.bincount(codes, weights=y) / counts)[codes]
    mass = counts[codes].astype(float)**-.5
    histogram = fit_histogram_boost(dimensionless(np.concatenate((x, *crossed)), paths),
                                    np.tile(target, 3), np.tile(mass, 3),
                                    stages=80, leaves=6, min_leaf=90, rate=.05)
    model.meta["school_b_transfer"]["spectral_residual"] = {
        "threshold": threshold, "novelty": np.quantile(novelty, GATE_LEVELS).tolist(), "histogram": histogram,
    }


def fit_out_of_bearing_offset(parent, x, y, groups, seed):
    """Calibrate the inherited model using temporary bearing-exclusive fits."""
    state = parent.meta.get("school_b_transfer", {})
    if (parent.meta.get("school_b_conditional_shrink") is None or "rank_gate" not in state
            or state.get("geometry_input") != "parent-normalized"):
        raise ValueError("B offset calibration requires a calibrated coverage parent")
    _, codes, counts = np.unique(groups, return_inverse=True, return_counts=True)
    if len(counts) < 3 or duration_mean_weight(y, codes, counts, 1500., .02, .65) is None:
        raise ValueError("B offset calibration requires at least three fitting bearings with mixed lifetime support")
    offsets = []
    proxy_steps = 0
    for code in range(len(counts)):
        training = codes != code
        proxy_x, proxy_y = x[training], y[training]
        proxy_groups = codes[training]
        proxy = type(parent)("regression", parent.meta["paths"], [], proxy_x, proxy_y,
                             hidden=24, seed=seed, batch_size=128, learning_rate=.003)
        proxy.fit(proxy_x, proxy_y, 15)
        weights = 1. / counts[proxy_groups]
        proxy, _ = update_prediction(proxy, proxy_x, proxy_y, proxy_groups, weights, "transfer-parent-coverage")
        proxy, _ = update_group_shape(proxy, proxy_x, proxy_y, proxy_groups)
        proxy.meta["school_b_conditional_shrink"] = fit_conditional_shrink(
            proxy_y, proxy_groups, quantile=.25, width=.5, retain=.15,
        )
        error = y[~training] - predict_model(proxy, x[~training])
        offsets.append(float(error.mean() / max(1., float(proxy_y.mean()))))
        proxy_steps += proxy.meta["steps"]
    relative = float(np.mean(offsets)) / 1.25
    offset = float(y.mean() * np.clip(relative, -.35, .35))
    calibrated = deepcopy(parent)
    calibrated.meta["school_b_transfer"]["out_of_bearing_offset_minutes"] = offset
    check_transfer(calibrated)
    return calibrated, [{
        "step": calibrated.meta["steps"], "epoch": calibrated.meta["epochs"],
        "calibration_fit_bearings": len(counts), "calibration_fit_rows": len(y),
        "out_of_bearing_offset_minutes": offset, "proxy_optimizer_steps": proxy_steps,
    }]
