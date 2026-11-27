"""Official school B RUL continuation via contest_harness SDK."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from functools import lru_cache, partial
from inspect import getsource
import json
import math
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from time import process_time

from baselines import model as baseline_model
from baselines.data import flatten
from baselines.model import Model
import numpy as np

from school_b_adaptation import refine_readout, training_bearings, training_bearing_weights
from school_b_transfer import (B_SHORT_PROTECTION_KEY, B_TRANSFER_DOSE_KEY, MIN_TRANSFER_SNAPSHOTS, check_transfer, fit_conditional_shrink,
                               fit_short_protection, predict_model, store_transfer_anchor, update_group_shape,
                               update_prediction)
from school_cache import RoundCache, array_digest, check_candidate_upload, file_digest, fingerprint, projection_digest
from school_planner import history_evidence, planner_messages as structured_messages, rank_untried_actions
from school_runtime import metric_gain, request_plan, run_trials, scope_regressions, scope_rmse, write_candidate

PROFILE = "B"
TASK = "regression"
METRIC = "rmse"
DIRECTION = "lower"
TARGET_UNIT = "minutes"
TASKS = (TASK,)
PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]
CODE_FILES = ("school_b_submission.py", "school_b_adaptation.py", "school_runtime.py", "school_cache.py",
              "school_planner.py", "school_profile.py", "school_b_transfer.py", "school_b_histograms.py")
BASELINE_FILES = ("baselines/model.py", "baselines/data.py", "baselines/predict.py", "baselines/__init__.py")
# Tolerance accommodates vectorized versus row-wise floating-point scoring.
METRIC_EPSILON = 1e-6
READOUT_RECIPES = {"readout-balanced": 3.0, "readout-conservative": 150.0}
CENTERED_RECIPE = "continue-within-bearing"
CENTERED_SETTINGS = (40, .001)
CENTERED_SHRINK_RECIPE = "continue-centered-and-calibrate"
GAIN_RECIPE = "continue-relative-gain"
GAIN_SETTINGS = (32, .001, 1.4)
PHASE_RECIPE = "continue-balanced-phase"
PHASE_SETTINGS = (64, .001, 1.4, .96, 1, 1.)
SHRINK_RECIPE = "continue-phase-and-calibrate"
SHRINK_SETTINGS = (.25, .5, .15)
CALIBRATED_UPDATE = "calibrated-update-1"
GUARDED_TRANSFER = "transfer-coverage-quarter"
GUARDED_CENTER = "centered-calibration-quarter"
TRANSFER_RECIPES = {
    "transfer-parent-coverage": ("physical-average", 110, "parent-normalized-max-fitted-rank", 0.70, .025),
    "transfer-two-signal": ("physical-average", 110, "max-fitted-rank", 0.80, .02),
    "transfer-physical": ("physical-average", 110, 0.95, 1.0),
    "readout-smoothing": ("physical-average",),
}
RECIPES = {
    **TRANSFER_RECIPES,
    CENTERED_RECIPE: CENTERED_SETTINGS,
    CENTERED_SHRINK_RECIPE: (*CENTERED_SETTINGS, *SHRINK_SETTINGS),
    SHRINK_RECIPE: (*PHASE_SETTINGS, *SHRINK_SETTINGS),
    PHASE_RECIPE: PHASE_SETTINGS,
    GAIN_RECIPE: GAIN_SETTINGS,
    **{name: ("anchored_readout", penalty, .0001) for name, penalty in READOUT_RECIPES.items()},
    "official-5": (5, 0.003, 128),
    "quick-2": (2, 0.003, 128),
    "long-12": (12, 0.003, 128),
    "fine-8": (8, 0.0008, 128),
}
CITY_RECIPES = {**RECIPES, CALIBRATED_UPDATE: (1, .0004, 128),
                GUARDED_TRANSFER: (*RECIPES["transfer-parent-coverage"], .25),
                GUARDED_CENTER: (*RECIPES[CENTERED_SHRINK_RECIPE], .25)}
DESCRIPTIONS = {
    "transfer-parent-coverage": "fit a bearing residual and gate on inherited feature coverage plus sensor disagreement",
    CENTERED_RECIPE: "continue the accepted coverage model using 40 Adam steps on within-bearing RUL error",
    CENTERED_SHRINK_RECIPE: "continue for 40 steps and fit conditional log shrinkage from training labels",
    GAIN_RECIPE: "continue the accepted centered model for 32 Adam steps with reciprocal channel gains",
    PHASE_RECIPE: "continue the accepted centered model with equal bearing weights, late-phase emphasis and reciprocal gains",
    SHRINK_RECIPE: "continue the accepted centered model and fit conditional log shrinkage from training labels",
    "transfer-two-signal": "fit a physical residual and use training novelty plus sensor disagreement for transfer",
    "transfer-physical": "fit physical-average readout and a residual for vibration outside training coverage",
    "readout-smoothing": "fit a frozen-hidden readout and average physically consistent sensor transformations",
    "readout-balanced": "equal bearing loss, freeze hidden layer, anchored output fit plus one Adam step",
    "readout-conservative": "same equal bearing output fit with stronger anchoring to the parent",
    "official-5": "official continuation for 5 epochs at 0.003 with batch 128",
    "quick-2": "short official update, 2 epochs at 0.003 with batch 128",
    "long-12": "extended official update, 12 epochs at 0.003 with batch 128",
    "fine-8": "lower-rate update, 8 epochs at 0.0008 with batch 128",
    CALIBRATED_UPDATE: "one low-rate epoch preserving the fitted coverage calibration",
    GUARDED_TRANSFER: "fit the same coverage route with quarter-strength transfer from the anchor",
    GUARDED_CENTER: "center and calibrate with quarter-strength transfer from the anchor",
}
SYSTEM = (
    "Plan one continuation round for school B rolling-bearing remaining useful life regression. "
    "RMSE in minutes is primary, lower is better, and MAE is secondary. "
    "Signed error is prediction minus target: positive means overestimation. "
    "Historical gain is parent RMSE minus candidate RMSE: positive is improvement. "
    "Use the current parent history to avoid repeating measured failures. "
    "Prior-parent results guide choices but do not measure this parent. "
    "Return one compact JSON object with actions (0 to 3 distinct available IDs), hypothesis, expected, and failure. "
    "Limit each text field to 12 English words. Do not repeat numbers or explain your reasoning. "
    "Use an empty actions array when no fresh experiment is justified. "
    "Example format: "
    '{"actions":[],"hypothesis":"Tested actions degraded validation.","expected":"Keep parent RMSE.",'
    '"failure":"A fresh action lowers RMSE."}'
)
CITY_SYSTEM = ("Plan B bearing RUL continuation. Lower RMSE and protect RUL bands and cohorts. "
               "Positive gain is better. Avoid repeating rejected trials on the same parent. "
               "Return compact JSON with actions, hypothesis, expected, failure. "
               "Use at most two offered IDs and short text. Use [] when exhausted. "
               "Band rows are range, count, mean prediction minus truth, and RMSE.")
CITY_DESCRIPTIONS = {
    "transfer-parent-coverage": "fit coverage residual", "transfer-two-signal": "fit dual-signal residual",
    CENTERED_SHRINK_RECIPE: "center and calibrate", GUARDED_TRANSFER: "quarter-strength coverage",
    GUARDED_CENTER: "quarter-strength calibration", CALIBRATED_UPDATE: "one epoch, preserve calibration",
    "official-5": "official five epochs", "quick-2": "official two epochs", "long-12": "official twelve epochs",
    "fine-8": "low-rate eight epochs", "readout-balanced": "balance bearing output error",
    "readout-conservative": "tightly anchor output update",
}


def check_task(dataset, contract):
    expected = {"dataset_key": PROFILE, "task": TASK, "metric": METRIC, "metric_direction": DIRECTION,
                "target_unit": TARGET_UNIT}
    if dataset != PROFILE or any(contract.get(key, value) != value for key, value in expected.items()):
        raise ValueError("this submission requires school B regression in minutes with lower-is-better RMSE")


def check_checkpoint(model):
    meta = model.meta
    if (meta["task"] != TASK or meta["classes"] or len(meta["paths"]) != 34
            or meta["hidden"] != 24 or meta["parameter_count"] != 865):
        raise ValueError("unexpected B parent architecture")
    check_transfer(model)


def arrays(rows, model):
    features, targets, ids = [], [], {}
    for row in rows:
        identity = row["id"]
        if not isinstance(identity, str) or not identity or identity in ids:
            raise ValueError("public projection requires unique nonempty sample IDs")
        ids[identity] = None
        paths, values = flatten(row["inputs"])
        if paths != model.meta["paths"]:
            raise ValueError("public features do not match the B checkpoint")
        answer = row["answer"]
        if type(answer) not in (float, int) or not math.isfinite(answer) or answer < 0:
            raise ValueError("B target must be finite nonnegative minutes")
        features.append(values)
        targets.append(answer)
    if not ids:
        raise ValueError("empty B public projection")
    x, y = np.asarray(features, dtype=np.float64), np.asarray(targets, dtype=np.float64)
    return x, y, set(ids), projection_digest(x, y, ids, target_dtype="<f8")


def regression_metrics(expected, predicted):
    residual = predicted - expected
    return {"rmse": float(np.sqrt(np.mean(np.square(residual)))), "mae": float(np.mean(np.abs(residual)))}


def score_projection(model, x, y):
    predicted = predict_model(model, x)
    if not np.all(np.isfinite(predicted) & (predicted >= 0)):
        raise ValueError("B predictions must be finite nonnegative minutes")
    return regression_metrics(y, predicted), predicted


def score_city_projection(model, x, y, *, groups):
    score, predicted = score_projection(model, x, y)
    return {**score, "scope_rmse": scope_rmse(y, predicted, (360, 720), groups)}, predicted


def validation_bearings(rows):
    matches = [re.fullmatch(r"xjtu_(Bearing[1-3]_[1-5])_s[0-9]{6}", row["id"], re.ASCII) for row in rows]
    if all(matches):
        return np.asarray([match[1] for match in matches])
    return None


def error_by_rul_band(y, predicted):
    bands = (("0-59", y < 60), ("60-239", (y >= 60) & (y < 240)), ("240+", y >= 240))
    return [{"minutes": label, "count": int(np.count_nonzero(mask)),
             "mean_signed_error": round(float(np.mean(predicted[mask] - y[mask])), 4),
             **{key: round(value, 4) for key, value in regression_metrics(y[mask], predicted[mask]).items()}}
            for label, mask in bands if mask.any()]


def trainer(parent, x, y, recipe, seed, *, fit_weights=None, fit_groups=None, fit_projection=None, city_policy=False):
    if recipe in (GUARDED_TRANSFER, GUARDED_CENTER):
        source = "transfer-parent-coverage" if recipe == GUARDED_TRANSFER else CENTERED_SHRINK_RECIPE
        candidate, history = trainer(parent, x, y, source, seed, fit_weights=fit_weights, fit_groups=fit_groups,
                                     fit_projection=fit_projection, city_policy=True)
        candidate.meta[B_TRANSFER_DOSE_KEY] = CITY_RECIPES[recipe][-1]
        history[0]["transfer_dose"] = CITY_RECIPES[recipe][-1]
        return candidate, history
    if recipe in (CENTERED_RECIPE, CENTERED_SHRINK_RECIPE, GAIN_RECIPE, PHASE_RECIPE, SHRINK_RECIPE):
        if fit_groups is None or fit_projection is None or parent.meta.get("school_b_readout_projection") != fit_projection:
            raise ValueError("B group-centered continuation requires its original fitting projection")
        centered = parent.meta.get("school_b_centered_projection") == fit_projection
        weighting = {}
        if recipe in (CENTERED_RECIPE, CENTERED_SHRINK_RECIPE):
            if centered:
                raise ValueError("B group-centered continuation already used this fitting projection")
            steps, rate = CENTERED_SETTINGS
            relative_gain = 1.
            marker = "school_b_centered_projection"
        else:
            if not centered or parent.meta.get("school_b_gain_projection") == fit_projection:
                raise ValueError("B relative-gain continuation requires an unaugmented centered parent")
            if recipe in (PHASE_RECIPE, SHRINK_RECIPE):
                steps, rate, relative_gain, late_emphasis, late_power, group_power = PHASE_SETTINGS
                weighting = {"late_emphasis": late_emphasis, "late_power": late_power, "group_power": group_power}
            else:
                steps, rate, relative_gain = GAIN_SETTINGS
            marker = "school_b_gain_projection"
        candidate, history = update_group_shape(parent, x, y, fit_groups, steps=steps, rate=rate,
                                                relative_gain=relative_gain, **weighting)
        candidate.meta.pop("school_b_conditional_shrink", None)
        candidate.meta.pop(B_SHORT_PROTECTION_KEY, None)
        candidate.meta.pop(B_TRANSFER_DOSE_KEY, None)
        if recipe in (CENTERED_SHRINK_RECIPE, SHRINK_RECIPE):
            quantile, width, retain = SHRINK_SETTINGS
            calibration = fit_conditional_shrink(y, fit_groups, quantile=quantile, width=width, retain=retain)
            candidate.meta["school_b_conditional_shrink"] = calibration
            history[0]["conditional_shrink"] = calibration
            if city_policy:
                protection = fit_short_protection(y, np.asarray(fit_groups))
                if protection is not None:
                    store_transfer_anchor(candidate, parent)
                    candidate.meta[B_SHORT_PROTECTION_KEY] = protection
                    history[0]["short_life_anchor_boundary_minutes"] = protection
            check_checkpoint(candidate)
        candidate.meta[marker] = fit_projection
        return candidate, history
    if recipe in TRANSFER_RECIPES:
        if fit_groups is None or fit_weights is None or fit_projection is None:
            raise ValueError("vibration transfer requires verified B fitting groups")
        candidate, history = update_prediction(parent, x, y, fit_groups, fit_weights, recipe)
        candidate.meta.pop(B_SHORT_PROTECTION_KEY, None)
        candidate.meta.pop(B_TRANSFER_DOSE_KEY, None)
        if city_policy:
            store_transfer_anchor(candidate, parent)
        candidate.meta["school_b_readout_projection"] = fit_projection
        candidate.meta.pop("school_b_centered_projection", None)
        candidate.meta.pop("school_b_gain_projection", None)
        return candidate, history
    if recipe in READOUT_RECIPES:
        if fit_weights is None or fit_projection is None:
            raise ValueError("readout requires verified B bearing groups")
        candidate, history = refine_readout(parent, x, y, fit_weights, READOUT_RECIPES[recipe])
        candidate.meta.pop("school_b_conditional_shrink", None)
        candidate.meta.pop(B_SHORT_PROTECTION_KEY, None)
        candidate.meta.pop(B_TRANSFER_DOSE_KEY, None)
        candidate.meta["school_b_readout_projection"] = fit_projection
        return candidate, history
    epochs, learning_rate, batch_size = CITY_RECIPES[recipe]
    model = deepcopy(parent)
    model.meta.update(learning_rate=learning_rate, batch_size=batch_size, seed=seed)
    if recipe != CALIBRATED_UPDATE:
        model.meta.pop("school_b_readout_projection", None)
        model.meta.pop("school_b_conditional_shrink", None)
        model.meta.pop(B_SHORT_PROTECTION_KEY, None)
        model.meta.pop(B_TRANSFER_DOSE_KEY, None)
    steps_per_epoch = (len(x) + batch_size - 1) // batch_size
    losses = model.fit(x, y, epochs)
    history = [{"epoch": parent.meta["epochs"] + i, "step": parent.meta["steps"] + i * steps_per_epoch,
                "loss": loss} for i, loss in enumerate(losses, start=1)]
    return model, history


def planner_messages(context):
    return structured_messages(context, SYSTEM)


def fallback_plan(context):
    actions, tested = rank_untried_actions(context)
    return {
        "hypothesis": f"Compare remaining B continuations after measuring {len(tested)} on this parent.",
        "actions": actions,
        "expected": "Lower public validation RMSE in minutes, or keep the measured parent when exhausted.",
        "failure": "No candidate lowers RMSE relative to the parent.",
    }


def received_rmse(submission):
    summary = submission.get("summary")
    metrics = summary.get("metrics") if isinstance(summary, dict) else None
    value = metrics.get(METRIC) if isinstance(metrics, dict) else None
    if type(value) not in (float, int) or value < 0 or value > np.finfo(np.float64).max or not math.isfinite(value):
        raise ValueError("host did not provide finite nonnegative B RMSE")
    return float(value)


def select_best(trials):
    return min(trials, key=lambda trial: (trial[METRIC], trial["mae"], trial["training_rows"]), default=None)


def adoption_reason(trial, *, parent_hash, gain, accepted):
    if accepted:
        return "lower_rmse"
    if trial is None:
        return "no_restorable_candidate"
    if trial["cache"]["prediction_sha256"] == parent_hash:
        return "predictions_identical"
    if abs(gain) <= METRIC_EPSILON:
        return "rmse_tie_with_different_predictions"
    return "candidate_worse"


def open_state(rnd, path, identity):
    state = rnd.state_workspace(path)
    marker = state / "profile.json"
    if marker.is_file():
        if json.loads(marker.read_bytes()) != identity:
            raise ValueError("state belongs to another dataset profile")
    elif any(state.iterdir()):
        raise ValueError("cannot import unmarked round state into B")
    marker.write_text(json.dumps(identity), encoding="utf-8")
    return state


def supported_recipes(parent, fit_weights, projection):
    exclusions = {CENTERED_RECIPE, CENTERED_SHRINK_RECIPE, GAIN_RECIPE, PHASE_RECIPE, SHRINK_RECIPE}
    previous = parent.meta.get("school_b_readout_projection")
    if fit_weights is None or previous == projection:
        exclusions.update(TRANSFER_RECIPES)
        exclusions.update(READOUT_RECIPES)
    if (fit_weights is not None and previous == projection
            and parent.meta.get("school_b_transfer", {}).get("geometry_input") == "parent-normalized"):
        if parent.meta.get("school_b_centered_projection") != projection:
            exclusions.difference_update((CENTERED_RECIPE, CENTERED_SHRINK_RECIPE))
        elif parent.meta.get("school_b_gain_projection") != projection:
            exclusions.difference_update((GAIN_RECIPE, PHASE_RECIPE, SHRINK_RECIPE))
    return {name: config for name, config in RECIPES.items() if name not in exclusions}


def city_grouped_recipes(supported):
    """Offer routes that survived whole-bearing and whole-condition screening."""
    if CENTERED_SHRINK_RECIPE in supported:
        names = (CENTERED_SHRINK_RECIPE,)
    else:
        names = ("transfer-parent-coverage", "transfer-two-signal")
    return {name: supported[name] for name in names if name in supported}


def city_trials(supported, parent, *, grouped):
    options = city_grouped_recipes(supported) if grouped else supported
    if CENTERED_SHRINK_RECIPE in options:
        options = {GUARDED_CENTER: CITY_RECIPES[GUARDED_CENTER], **options}
    if "transfer-parent-coverage" in options:
        options = {GUARDED_TRANSFER: CITY_RECIPES[GUARDED_TRANSFER], **options}
    # After the calibrated round there is still a conservative continuation,
    # including when refreshed inputs no longer reveal reliable bearing IDs.
    if "school_b_conditional_shrink" in parent.meta:
        options = {CALIBRATED_UPDATE: CITY_RECIPES[CALIBRATED_UPDATE], **options}
    return options


def city_risk(parent, candidate):
    return scope_regressions(parent["scope_rmse"], candidate["scope_rmse"], ratio=1.05, absolute=5.)


def missing_historical_bands(previous, parent_hash, train_projection, current_scopes):
    """Protect previously validated life stages when only evaluation coverage contracts."""
    past_bands = {scope for record in previous
                  if record["training"] == train_projection and record.get("selected") == parent_hash
                  for scope in record.get("validation_bands", ())}
    return sorted(past_bands.difference(current_scopes))


def train(rnd, *, city_screen=False):
    started = process_time()
    check_task(rnd.dataset_key, rnd.task)
    history = rnd.history()
    parent_path = rnd.parent.path("checkpoint.npz")
    parent_hash = file_digest(parent_path)
    if parent_hash != rnd.parent.checkpoint_sha256:
        raise ValueError("wrong B parent checkpoint")
    parent = Model.load(parent_path)
    check_checkpoint(parent)
    train_rows = list(rnd.data.training_rows())
    x, y, train_ids, train_projection = arrays(train_rows, parent)
    fit_weights = training_bearing_weights(train_rows)
    fit_groups = training_bearings(train_rows)
    if city_screen and fit_groups is not None and min(Counter(fit_groups).values()) < MIN_TRANSFER_SNAPSHOTS:
        # A newly released projection can have too few snapshots in a bearing for the transfer feature book.
        fit_weights = fit_groups = None
    available_recipes = supported_recipes(parent, fit_weights, train_projection)
    screened = city_screen and fit_weights is not None
    if city_screen:
        available_recipes = city_trials(available_recipes, parent, grouped=screened)
    fit_candidate = partial(trainer, fit_weights=fit_weights, fit_groups=fit_groups, fit_projection=train_projection,
                            city_policy=city_screen)
    validation_rows = list(rnd.data.validation_rows())
    vx, vy, validation_ids, validation_projection = arrays(validation_rows, parent)
    if train_ids & validation_ids:
        raise ValueError("training and validation IDs overlap")
    validation_groups = validation_bearings(validation_rows) if city_screen else None
    evaluate = partial(score_city_projection, groups=validation_groups) if city_screen else score_projection
    parent_score, parent_prediction = evaluate(parent, vx, vy)
    parent_prediction_hash = array_digest(parent_prediction, "<f8")
    seed = getattr(rnd, "run_seed", None)
    if seed is None:
        seed = parent.meta["seed"]
    model_sources = Path(baseline_model.__file__).resolve().parent
    reference = {"numpy": np.__version__, "model": file_digest(model_sources / "model.py")}
    adapters = {name: file_digest(Path(__file__).with_name(name)) for name in
                ("school_b_adaptation.py", "school_b_transfer.py", "school_b_histograms.py")}
    trainer_id = fingerprint({"reference": reference, "source": getsource(trainer), "adapters": adapters})
    scorer_id = fingerprint({"reference": reference, "source": getsource(score_projection),
                             "metrics": getsource(regression_metrics), "adapters": adapters,
                             "scopes": ([getsource(score_city_projection), getsource(scope_rmse),
                                          validation_groups.tolist() if validation_groups is not None else None]
                                        if city_screen else None)})
    pack_id = fingerprint({"dataset": PROFILE, "task": TASK, "unit": TARGET_UNIT,
                           "label_origin": "remaining_time_from_lifecycle", "features": parent.meta["paths"]})
    projection_key = fingerprint({"validation": validation_projection, "scorer": scorer_id})
    workspace = Path(os.environ.get("CONTEST_WORK", "/workspace"))
    with TemporaryDirectory(prefix="school-b-", dir=workspace) as name:
        root = Path(name)
        state = open_state(rnd, root / "state", {"profile": PROFILE, "fingerprint": pack_id})
        cache = RoundCache(state, rnd.round_index)
        decision_path = state / "decision-evidence.json"
        previous = json.loads(decision_path.read_bytes()) if decision_path.is_file() else []
        current_scopes = parent_score.get("scope_rmse", {})
        missing_bands = (missing_historical_bands(previous, parent_hash, train_projection, current_scopes)
                         if city_screen else [])
        if missing_bands:
            available_recipes = {}
        evidence = history_evidence(previous, parent_hash=parent_hash, train_key=train_projection,
                                    evaluation_key=projection_key, seed=seed, compact=city_screen)
        tested = {row["action"] for row in evidence["current"]}
        next_actions = [name for name in available_recipes if name not in tested]
        max_tokens = min(168 if city_screen else 384, rnd.max_output_tokens)
        planner_parent = {key: round(parent_score[key], 6) for key in (METRIC, "mae")}
        bands = error_by_rul_band(vy, parent_prediction)
        context = {
            "profile": PROFILE, "round": rnd.round_index, "parent": parent_hash[7:19] if city_screen else parent_hash,
            "parent_steps": parent.meta["steps"], "task": TASK,
            "metric": {"primary": METRIC, "direction": DIRECTION, "unit": TARGET_UNIT, "secondary": "mae"},
            "parent_metrics": planner_parent,
            "error_by_rul_band": ([[row["minutes"], row["count"], round(row["mean_signed_error"], 1), round(row["rmse"], 1)]
                                   for row in bands] if city_screen else bands),
            "history": evidence, "available": {name: CITY_DESCRIPTIONS.get(name, DESCRIPTIONS[name]) if city_screen
                                               else DESCRIPTIONS[name] for name in next_actions},
            "preference": next_actions,
            "budget": {"actions": 2 if city_screen else 3, "rounds_remaining": max(0, rnd.frame.get("rounds_per_dataset", 10) - rnd.round_index + 1),
                       "wall_seconds": rnd.wall_seconds, "max_output_tokens": max_tokens},
        }
        build_messages = partial(structured_messages, system=CITY_SYSTEM) if city_screen else planner_messages
        receipt, plan, fallback = request_plan(rnd, context, build_messages, fallback_plan, max_tokens)
        requested = plan["actions"]
        reuse_only = not requested
        trials, artifacts, execution = run_trials(
            rnd, root, cache, requested or list(available_recipes), available_recipes, reuse_only=reuse_only, pack_id=pack_id,
            parent_hash=parent_hash, parent=parent, seed=seed, train_projection=train_projection,
            projection_key=projection_key, trainer_id=trainer_id, trainer=fit_candidate, score_projection=evaluate,
            x=x, y=y, vx=vx, vy=vy, load_checkpoint=Model.load, prediction_dtype="<f8",
        )
        rejected = {trial["recipe"]: city_risk(parent_score, trial) for trial in trials} if city_screen else {}
        best = select_best([trial for trial in trials if not rejected.get(trial["recipe"])])
        recipe = best["recipe"] if best else None
        raw_reply = receipt.text if isinstance(receipt.text, str) else ""
        config = {
            "profile": PROFILE, "recipe": recipe, "plan": plan, "trials": trials, "reuse_only": reuse_only,
            "planner_fallback": fallback, "planner_issue": "parse_error" if fallback else None,
            "raw_reply": raw_reply.encode("utf-8")[:4000].decode("utf-8", errors="ignore"),
            "parent_checkpoint_sha256": parent_hash, "parent_prediction_sha256": parent_prediction_hash,
            "execution": {**execution, "cpu_seconds": process_time() - started},
        }
        if screened:
            config["screening"] = "whole-bearing and whole-condition training projections"
        if city_screen:
            config.update(fitting_projection_sha256=train_projection, validation_projection_sha256=validation_projection,
                          validation_bearings_verified=validation_groups is not None,
                          historical_bands_missing=missing_bands,
                          stage_rejections={recipe: failures for recipe, failures in rejected.items() if failures})
        measured = best[METRIC] if best else parent_score[METRIC]
        evaluation = (cache.entries[best["cache"]["training_key"]]["scores"][projection_key] if best else {})
        prior_host_rmse = evaluation.get("host_rmse")
        already_rejected = (prior_host_rmse is not None
                            and metric_gain(parent_score[METRIC], prior_host_rmse, DIRECTION) <= METRIC_EPSILON)
        submission, candidate_bytes = None, 0
        measurement_source = "local"
        if already_rejected:
            # Same parent, candidate bytes, and validation/scorer identity.
            measured = prior_host_rmse
            measurement_source = "cached_host"
        elif best and (not reuse_only or metric_gain(parent_score[METRIC], measured, DIRECTION) > METRIC_EPSILON):
            checkpoint, epochs = artifacts[recipe]
            candidate_dir = root / recipe
            report = {"round": rnd.round_index, "profile": PROFILE, "seed": seed, "parent_checkpoint_sha256": parent_hash,
                      "checkpoint_sha256": best["cache"]["checkpoint_sha256"], "parent": parent_score,
                      "candidate": best, "epoch_history": epochs, "plan": plan}
            write_candidate(candidate_dir, checkpoint, report, source_dir=Path(__file__).resolve().parent,
                            baseline_dir=model_sources, code_files=CODE_FILES)
            candidate_bytes = check_candidate_upload(candidate_dir, rnd.limits)
            reused = best["cache"]["training_reused_from_round"]
            change = (f"Recovered {recipe} weights and Adam state from round {reused}." if reused is not None
                      else f"Continued the B parent weights, optimizer and scaler using {recipe}.")
            submission = rnd.submit_candidate(candidate_dir, hypothesis=plan["hypothesis"], change_summary=change,
                                              configuration=config, seed=seed, checkpoint_paths=["checkpoint.npz"],
                                              code_paths=[*CODE_FILES, *BASELINE_FILES], predict_argv=PREDICT_ARGV)
            try:
                measured = received_rmse(submission)
            except ValueError:
                measured = None
                measurement_source = "host_unavailable"
            else:
                evaluation["host_rmse"] = measured
                measurement_source = "host"
        gain = metric_gain(parent_score[METRIC], measured, DIRECTION) if measured is not None else None
        accepted = submission is not None and gain is not None and gain > METRIC_EPSILON
        why = ("historical_scope_unobserved" if missing_bands else "host_metric_unavailable" if gain is None else
               "stage_guard" if best is None and trials else
               adoption_reason(best, parent_hash=parent_prediction_hash, gain=gain, accepted=accepted))
        selected_artifact = submission["artifact_id"] if accepted else rnd.parent.artifact_id
        selected_hash = best["cache"]["checkpoint_sha256"] if accepted else parent_hash
        config["adoption"] = {"margin": METRIC_EPSILON, "gain": gain, "accepted": accepted, "reason": why,
                              "selected_rmse": measured if accepted else parent_score[METRIC],
                              "measurement_source": measurement_source}
        trace = {"context": context, "raw_reply": raw_reply, "parent": parent_score, "candidate_rmse": measured,
                 "selected_checkpoint_sha256": selected_hash, "selected_artifact_id": selected_artifact,
                 "configuration": config}
        (state / f"round-{rnd.round_index:02d}.json").write_text(json.dumps(trace, allow_nan=False), encoding="utf-8")
        previous.append({"parent": parent_hash, "training": train_projection, "validation": projection_key,
                         "seed": seed, "round": rnd.round_index,
                         **({"selected": selected_hash,
                             "validation_bands": sorted(key for key in current_scopes if key.startswith("rul:"))}
                            if city_screen else {}),
                         "outcomes": [{"action": trial["recipe"], "rmse": round(trial[METRIC], 6),
                                       "mae": round(trial["mae"], 6),
                                       "gain": round(metric_gain(parent_score[METRIC], trial[METRIC], DIRECTION), 6),
                                       **({"stage_safe": not bool(rejected[trial["recipe"]])} if city_screen else {}),
                                       "optimizer_steps": trial["final_steps"] - trial["initial_steps"],
                                       "first_round": trial["cache"]["score_first_round"]} for trial in trials]})
        decision_path.write_text(json.dumps(previous[-10:], allow_nan=False), encoding="utf-8")
        cache.persist(rnd.limits, candidate_bytes)
        rnd.save_state(state)
        candidate_description = f"{measured:.6f} minutes" if measured is not None else "unavailable from host"
        rnd.finish(selected_artifact_id=selected_artifact, hypothesis=plan["hypothesis"], configuration=config, seed=seed,
                   change_summary=(f"Actions {', '.join(requested) or 'reuse only'}, "
                                   f"trained {execution['trained_candidates']}, reused {execution['reused_candidates']}. "
                                   + ("Adopted the candidate." if accepted else "Kept the parent.")),
                   reason=(f"Parent RMSE {parent_score[METRIC]:.6f}, candidate RMSE {candidate_description}, {why}. "
                           f"{'Invalid planner text used the history fallback. ' if fallback else ''}"
                           f"Expected {plan['expected']} Failure {plan['failure']} Receipt {receipt.call_id}."),
                   history_ids=[entry["id"] for entry in history], model_call_ids=list(rnd.call_ids))


@lru_cache(maxsize=1)
def load_checkpoint(path):
    model = Model.load(path)
    check_checkpoint(model)
    return model


def predict(sample, model_dir):
    if sample.task != TASK or getattr(sample, "dataset_key", None) not in (None, PROFILE):
        raise ValueError("B requires a bearing RUL regression request")
    checkpoint = Path(os.environ.get("CONTEST_ARTIFACT_PATH", Path(model_dir) / "checkpoint.npz"))
    model = load_checkpoint(checkpoint)
    paths, values = flatten(sample.inputs)
    if paths != model.meta["paths"]:
        raise ValueError("prediction features do not match the B checkpoint")
    value = float(predict_model(model, np.asarray([values], dtype=np.float64))[0])
    if not math.isfinite(value) or value < 0:
        raise ValueError("B prediction must be finite nonnegative minutes")
    return {"value": value}
