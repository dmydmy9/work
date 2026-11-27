"""Continue the official C-MAPSS MLP through contest SDK rounds."""

from functools import lru_cache, partial
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import process_time

from baselines import model as baseline_model
from baselines.data import flatten
from baselines.model import Model
import numpy as np

from school_c_recipes import CAP_RECIPE, CITY_RECIPES, continue_parent, group_ids, metrics
from school_c_kernel import RUL_CAP_KEY, SETTINGS_PATH, compile_regime_kernel, predict_regime_kernel, validate_rul_caps
from school_cache import RoundCache, check_candidate_upload, file_digest, fingerprint, projection_digest
from school_planner import history_evidence, planner_messages, rank_untried_actions
from school_runtime import request_plan, run_trials, scope_regressions, scope_rmse, write_candidate


CODE_FILES = ("school_c_submission.py", "school_c_recipes.py", "school_c_kernel.py", "school_runtime.py", "school_cache.py",
              "school_planner.py", "school_profile.py")
BASELINE_FILES = ("baselines/model.py", "baselines/data.py", "baselines/predict.py", "baselines/__init__.py")
PREFERENCES = (CAP_RECIPE, "regime-kernel-5", "conservative-5", "official-5", "engine-balanced-5", "accelerated-5")
DESCRIPTIONS = {CAP_RECIPE: "fit regime residuals with a high RUL prediction cap estimated on training data",
                "regime-kernel-5": "continue with lower learning rate and refit nonlinear sensor residuals by regime",
                "conservative-5": "lower the official learning rate by four",
                "official-5": "continue at the official learning rate",
                "engine-balanced-5": "sample engines with equal probability at the official rate",
                "accelerated-5": "double the official learning rate"}
SYSTEM = ("Plan one C-MAPSS continuation experiment from the supplied checkpoint. Minimize RMSE in cycles. "
          "Protect RUL bands and overestimate tails. Positive historical gain means lower RMSE. "
          "Return JSON only with actions, hypothesis, expected and failure. Choose at most one offered ID. "
          "Use an empty actions array only when existing trials exhaust the useful candidates. "
          "Keep each statement under 12 words.")
MIN_GAIN_CYCLES = .025
C_PATHS = flatten({"sensors": [0] * 21, "settings": [0] * 3})[0]


def c_features(inputs, expected_paths):
    if expected_paths != C_PATHS:
        paths, values = flatten(inputs)
        if paths != expected_paths:
            raise ValueError("C features do not match the checkpoint")
        return values
    if not isinstance(inputs, dict) or inputs.keys() != {"sensors", "settings"}:
        raise ValueError("C inputs must contain sensors and settings")
    sensors, settings = inputs["sensors"], inputs["settings"]
    if not isinstance(sensors, list) or not isinstance(settings, list) or len(sensors) != 21 or len(settings) != 3:
        raise ValueError("C sensor and setting dimensions do not match the checkpoint")
    values = sensors + settings
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        raise ValueError("C features must be finite numbers")
    return values


def read_projection(rows, model):
    ids, features, targets = {}, [], []
    paths = model.meta["paths"]
    for row in rows:
        key, target = row["id"], row["answer"]
        if not isinstance(key, str) or not key or key in ids:
            raise ValueError("C sample IDs must be nonempty and unique")
        if type(target) not in (int, float) or not math.isfinite(target) or target < 0:
            raise ValueError("C target must be finite and nonnegative")
        values = c_features(row["inputs"], paths)
        ids[key] = None
        features.append(values)
        targets.append(target)
    if not ids:
        raise ValueError("empty C projection")
    x, y = np.asarray(features, dtype=np.float64), np.asarray(targets, dtype=np.float64)
    return x, y, tuple(ids), projection_digest(x, y, ids, target_dtype="<f8")


def scorer(model, x, y):
    prediction = predict_regime_kernel(model, x, compile_regime_kernel(model))
    if not np.isfinite(prediction).all():
        raise ValueError("C predictions must be finite")
    regimes = np.rint(x[:, model.meta["paths"].index(SETTINGS_PATH)]).astype(int)
    positive = np.maximum(prediction - y, 0.)
    result = metrics(y, prediction)
    result["scope_rmse"] = scope_rmse(y, prediction, (30, 90, 180), regimes)
    result["overestimate_q99"] = float(np.quantile(positive, .99))
    result["overestimate_nasa"] = float(np.expm1(positive / 10).sum())
    return result, prediction


def validation_risk(parent, candidate):
    failures = scope_regressions(parent["scope_rmse"], candidate["scope_rmse"], ratio=1.05, absolute=2.)
    for key, ratio, floor in (("nasa_score", 1.10, 1.), ("overestimate_q99", 1.05, 2.),
                              ("overestimate_nasa", 1.10, 1.)):
        delta = candidate[key] - parent[key]
        if delta > max(floor, (ratio - 1.) * parent[key]):
            failures[key] = delta
    return failures


def fallback(context):
    actions, _ = rank_untried_actions(context)
    return {"actions": actions, "hypothesis": "Measure the highest ranked untried continuation.",
            "expected": "Reduce validation RMSE in cycles.", "failure": "Keep the parent if RMSE does not improve."}


@lru_cache(maxsize=1)
def load_checkpoint(path):
    model = Model.load(path)
    meta = model.meta
    if (meta["task"], len(meta["paths"]), meta["hidden"], meta["parameter_count"]) != ("regression", 24, 32, 833):
        raise ValueError("unexpected C architecture")
    validate_rul_caps(model)
    return model


def train(rnd):
    started = process_time()
    expected = {"task": "regression", "metric": "rmse", "metric_direction": "lower", "target_unit": "cycles"}
    if rnd.dataset_key != "C" or any(rnd.task.get(k, v) != v for k, v in expected.items()):
        raise ValueError("C requires regression in cycles with lower-is-better RMSE")
    history = rnd.history()
    path = rnd.parent.path("checkpoint.npz")
    parent_hash = file_digest(path)
    if parent_hash != rnd.parent.checkpoint_sha256:
        raise ValueError("wrong C parent checkpoint")
    parent = load_checkpoint(path)
    seed = parent.meta["seed"]
    x, y, ids, fit_key = read_projection(rnd.data.training_rows(), parent)
    vx, vy, validation_ids, validation_key = read_projection(rnd.data.validation_rows(), parent)
    if set(ids).intersection(validation_ids):
        raise ValueError("C training and validation IDs overlap")
    groups = group_ids(ids, allow_opaque=True)
    validation_groups = group_ids(validation_ids, allow_opaque=True)
    if groups is not None and validation_groups is not None and set(groups).intersection(validation_groups):
        raise ValueError("C training and validation share an engine")
    parent_metrics, _ = scorer(parent, vx, vy)
    baseline = Path(baseline_model.__file__).resolve().parent
    source = Path(__file__).resolve().parent
    trainer_id = fingerprint([np.__version__, file_digest(baseline / "model.py"),
                              file_digest(source / "school_c_recipes.py"), file_digest(source / "school_c_kernel.py"),
                              file_digest(source / "school_runtime.py"), file_digest(__file__)])
    pack_id = fingerprint(["C", parent.meta["paths"]])
    evaluation_key = fingerprint([validation_key, trainer_id])

    with TemporaryDirectory(prefix="school-c-", dir=os.environ.get("CONTEST_WORK", "/workspace")) as name:
        root = Path(name)
        state = rnd.state_workspace(root / "state")
        decision_file = state / "c-decisions.json"
        events = json.loads(decision_file.read_bytes()) if decision_file.is_file() else []
        evidence = history_evidence(events, parent_hash=parent_hash, train_key=fit_key,
                                    evaluation_key=evaluation_key, seed=seed, compact=True)
        tested = {item["action"] for item in evidence["current"]}
        available = [action for action in PREFERENCES if action not in tested
                     and (groups is not None or not CITY_RECIPES[action][2])]
        # When the fit is identical the ordinary kernel arm would refit the same capped predictor.
        if RUL_CAP_KEY in parent.meta and parent.meta.get("c_fit_projection") == fit_key:
            available = [action for action in available if action != "regime-kernel-5"]
        max_tokens = min(160, rnd.max_output_tokens)
        context = {"profile": "C", "round": rnd.round_index, "parent": parent_hash[7:19],
                   "parent_metrics": {key: round(parent_metrics[key], 5) for key in ("rmse", "mae", "overestimate_q99")},
                   "history": evidence, "preference": available,
                   "available": {action: DESCRIPTIONS[action] for action in available},
                   "budget": {"actions": 1, "max_output_tokens": max_tokens, "wall_seconds": rnd.wall_seconds}}
        cache = RoundCache(state, rnd.round_index)

        def fit_candidate(old, data, targets, action, _seed):
            trained, losses = continue_parent(old, data, targets, groups, action)
            if action == CAP_RECIPE:
                trained.meta["c_fit_projection"] = fit_key
            return trained, losses

        _, plan, used_fallback = request_plan(rnd, context, partial(planner_messages, system=SYSTEM), fallback, max_tokens)
        trials, artifacts, execution = run_trials(
            rnd, root, cache, plan["actions"], CITY_RECIPES, reuse_only=False, pack_id=pack_id,
            parent_hash=parent_hash, parent=parent, seed=seed, train_projection=fit_key,
            projection_key=evaluation_key, trainer_id=trainer_id, trainer=fit_candidate, score_projection=scorer,
            x=x, y=y, vx=vx, vy=vy, load_checkpoint=load_checkpoint, prediction_dtype="<f8")
        candidate = trials[0] if trials else None
        guard_failures = validation_risk(parent_metrics, candidate) if candidate else {}
        config = {"profile": "C", "parent_checkpoint_sha256": parent_hash, "parent_metrics": parent_metrics,
                  "plan": plan, "context": context, "planner_fallback": used_fallback, "trials": trials,
                  "execution": execution, "fitting_engine_groups_verified": groups is not None,
                  "fitting_projection_sha256": fit_key, "validation_projection_sha256": validation_key,
                  "stage_rejections": guard_failures}
        selected, selected_hash, candidate_bytes = rnd.parent.artifact_id, parent_hash, 0
        measured = None
        if candidate and not guard_failures and parent_metrics["rmse"] - candidate["rmse"] > MIN_GAIN_CYCLES:
            recipe = candidate["recipe"]
            checkpoint, losses = artifacts[recipe]
            candidate_dir = root / "candidate"
            digest = candidate["cache"]["checkpoint_sha256"]
            report = {"round": rnd.round_index, "seed": seed, "profile": "C", "epoch_history": losses,
                      "parent_checkpoint_sha256": parent_hash, "checkpoint_sha256": digest,
                      "parent": parent_metrics, "candidate": candidate, "plan": plan}
            write_candidate(candidate_dir, checkpoint, report, source_dir=source, baseline_dir=baseline,
                            code_files=CODE_FILES)
            candidate_bytes = check_candidate_upload(candidate_dir, rnd.limits)
            receipt = rnd.submit_candidate(candidate_dir, hypothesis=plan["hypothesis"], configuration=config, seed=seed,
                                           change_summary=f"Continued the C parent and Adam state with {recipe}.",
                                           checkpoint_paths=["checkpoint.npz"], code_paths=[*CODE_FILES, *BASELINE_FILES],
                                           predict_argv=["python3", "/app/harness.py", "predict"])
            summary = receipt.get("summary")
            observed = summary.get("metrics") if isinstance(summary, dict) else None
            value = observed.get("rmse") if isinstance(observed, dict) else None
            if type(value) in (float, int) and math.isfinite(value) and value >= 0:
                measured = float(value)
                if parent_metrics["rmse"] - max(candidate["rmse"], measured) > MIN_GAIN_CYCLES:
                    selected, selected_hash = receipt["artifact_id"], digest
        selected_rmse = parent_metrics["rmse"] if selected_hash == parent_hash else measured
        reason = ("stage_guard" if guard_failures else "no_candidate" if not candidate else
                  "lower_rmse" if selected_hash != parent_hash else
                  "below_minimum_gain" if candidate["rmse"] >= parent_metrics["rmse"] - MIN_GAIN_CYCLES else
                  "host_metric_unavailable" if measured is None else "host_did_not_confirm_gain")
        config.update({"selected_checkpoint_sha256": selected_hash, "selected_rmse": selected_rmse,
                       "host_candidate_rmse": measured, "adoption_reason": reason, "minimum_gain_cycles": MIN_GAIN_CYCLES,
                       "cpu_seconds": process_time() - started})
        outcomes = [{"action": trial["recipe"], "gain": parent_metrics["rmse"] - trial["rmse"],
                     "rmse": trial["rmse"], "first_round": rnd.round_index,
                     "stage_safe": not bool(guard_failures)} for trial in trials]
        events.append({"parent": parent_hash, "training": fit_key, "validation": evaluation_key, "seed": seed,
                       "outcomes": outcomes, "configuration": {"plan": plan, "selected_rmse": selected_rmse}})
        decision_file.write_text(json.dumps(events[-10:], allow_nan=False), encoding="utf-8")
        cache.persist(rnd.limits, candidate_bytes)
        rnd.save_state(state)
        rnd.finish(selected_artifact_id=selected, hypothesis=plan["hypothesis"], configuration=config, seed=seed,
                   change_summary=f"C continuation actions {plan['actions']!r}.",
                   reason=f"Parent RMSE {parent_metrics['rmse']:.6f}, selected RMSE {selected_rmse:.6f} cycles. {reason}.",
                   history_ids=[record["id"] for record in history], model_call_ids=list(rnd.call_ids))


@lru_cache(maxsize=1)
def load_predictor(path):
    model = load_checkpoint(path)
    return model, compile_regime_kernel(model)


def predict(sample, model_dir):
    if sample.task != "regression" or sample.dataset_key not in (None, "C"):
        raise ValueError("C requires an engine RUL regression request")
    path = Path(os.environ.get("CONTEST_ARTIFACT_PATH", Path(model_dir) / "checkpoint.npz"))
    model, compiled = load_predictor(path)
    values = c_features(sample.inputs, model.meta["paths"])
    return {"value": float(predict_regime_kernel(model, np.asarray([values], dtype=np.float64), compiled)[0])}
