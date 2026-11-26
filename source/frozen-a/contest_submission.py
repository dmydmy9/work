"""School A entry for contest_harness 1.0.4 and the official NumPy checkpoint."""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from inspect import getsource
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from time import process_time

from baselines import model as baseline_model
from baselines.model import Model, checkpoint_digest
from baselines.data import flatten, json_value
import numpy as np
from school_cache import RoundCache, array_digest, check_candidate_upload, file_digest, fingerprint, projection_digest

TASKS = ("classification",)
PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]
RECIPES = {
    "stable": (4, 0.001, 64, False, 0.0),
    "balanced": (8, 0.0015, 64, True, 0.0),
    "explore": (20, 0.003, 64, True, 0.0),
    "polish": (12, 0.0006, 32, True, 0.0),
    "augment": (16, 0.0015, 64, True, 0.12),
}
SYSTEM = ("Rank public-validation experiments that continue the supplied A NumPy MLP and Adam checkpoint. "
          "Use class F1, class counts and measured history to rank up to three available recipes. "
          "stable=short conservative; balanced=correct class imbalance; explore=longer balanced update; "
          "polish=small balanced steps; augment=balanced update with sensor perturbation. "
          'Return only JSON {"order":["recipe-id"]}. Use only IDs from available.')


def arrays(rows, model, *, provenance=False):
    features, targets, seen = [], [], {}
    label_indices = {name: i for i, name in enumerate(model.meta["classes"])}
    for row in rows:
        identity = row["id"]
        if identity in seen:
            raise ValueError("duplicate sample ID in public projection")
        seen[identity] = None
        paths, values = flatten(row["inputs"])
        if paths != model.meta["paths"]:
            raise ValueError("public features do not match parent checkpoint")
        features.append(values)
        targets.append(label_indices[row["answer"]])
    if not features:
        raise ValueError("empty public projection")
    matrix, labels = np.asarray(features, dtype=np.float64), np.asarray(targets, dtype=np.int64)
    if provenance:
        return matrix, labels, set(seen), projection_digest(matrix, labels, seen)
    return matrix, labels, set(seen)


def score_projection(model, x, y):
    predicted = model.predict(x).argmax(axis=1)
    count = len(model.meta["classes"])
    matrix = np.bincount(count * y + predicted, minlength=count * count).reshape(count, count)
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    f1 = np.divide(2 * matrix.diagonal(), denominator, out=np.zeros(count), where=denominator != 0)
    return {"macro_f1": float(f1[denominator > 0].mean()),
            "accuracy": float(np.mean(predicted == y)), "class_f1": f1.tolist(),
            "confusion": matrix.tolist()}, predicted


def score(model, x, y):
    return score_projection(model, x, y)[0]


def trainer(parent, x, y, recipe_id, seed):
    epochs, learning_rate, batch_size, balanced, jitter = RECIPES[recipe_id]
    candidate = deepcopy(parent)
    candidate.meta.update(learning_rate=learning_rate, batch_size=batch_size)
    rng = np.random.default_rng(np.random.SeedSequence([seed, parent.meta["steps"]]))
    records = []
    counts = np.bincount(y, minlength=len(parent.meta["classes"]))
    if balanced:
        # Temper the empirical class prior, preserving the original epoch size.
        weights = 1 / np.sqrt(counts[y])
        weights /= weights.sum()
    for _ in range(epochs):
        if balanced:
            rows = rng.choice(len(y), size=len(y), replace=True, p=weights)
            inputs, targets = x[rows], y[rows]
        else:
            inputs, targets = x, y
        if jitter:
            sensor_gain = np.exp(rng.normal(0, jitter, size=(len(inputs), 16, 1)))
            inputs = (inputs.reshape(-1, 16, 8) * sensor_gain).reshape(-1, 128)
        loss = candidate.fit(inputs, targets, 1)[0]
        records.append({"epoch": candidate.meta["epochs"], "step": candidate.meta["steps"], "loss": loss})
    return candidate, records


def planner_messages(context):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(context, sort_keys=True, separators=(",", ":"), allow_nan=False)}]


def baseline_order(context):
    available = context["available"]
    priorities = {recipe: 0.0 for recipe in available}
    for event in context["history"]:
        if event["recipe"] in priorities:
            priorities[event["recipe"]] += event["gain"]
    defaults = ("balanced", "augment", "explore", "polish", "stable")
    return sorted(available, key=lambda recipe: (-priorities[recipe], defaults.index(recipe)))[:3]


def parse_order(text, context):
    if not isinstance(text, str):
        return None
    try:
        parsed = json_value(text.strip().removesuffix("<|im_end|>"))
    except ValueError:
        return None
    order = parsed.get("order") if isinstance(parsed, dict) and set(parsed) == {"order"} else None
    if (not isinstance(order, list) or not 1 <= len(order) <= 3
            or any(not isinstance(item, str) or item not in context["available"] for item in order)
            or len(set(order)) != len(order)):
        return None
    return order


def planner_context(round_index, parent, parent_score, train_targets, history):
    return {
        "profile": "A", "round": round_index, "parent_steps": parent.meta["steps"],
        "parent_macro_f1": round(parent_score["macro_f1"], 5),
        "train_counts": np.bincount(train_targets, minlength=6).tolist(),
        "class_f1": [round(value, 5) for value in parent_score["class_f1"]],
        "history": history[-8:], "available": list(RECIPES),
    }


def rank_trials(trials):
    return sorted(trials, key=lambda trial: (trial["macro_f1"], trial["accuracy"], -trial["training_rows"]),
                  reverse=True)


def platform_score(response, local):
    metrics = response.get("summary", {}).get("metrics", {})
    if not isinstance(metrics, dict):
        return local
    value = metrics.get("macro_f1")
    if type(value) in (int, float) and np.isfinite(value) and 0 <= value <= 1:
        return float(value)
    return local


def materialize_candidate(directory, checkpoint, report):
    directory.mkdir()
    shutil.copyfile(checkpoint, directory / "checkpoint.npz")
    (directory / "baselines").mkdir()
    for filename in ("model.py", "data.py", "predict.py", "__init__.py"):
        shutil.copyfile(Path(baseline_model.__file__).resolve().parent / filename,
                        directory / "baselines" / filename)
    for filename in ("contest_submission.py", "school_cache.py"):
        shutil.copyfile(Path(__file__).with_name(filename), directory / filename)
    (directory / "training.json").write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def train(rnd):
    started_cpu = process_time()
    if rnd.dataset_key != "A":
        raise ValueError("this submission supports profile A")
    history = rnd.history()
    parent_path = rnd.parent.path("checkpoint.npz")
    parent_hash = checkpoint_digest(parent_path)
    if parent_hash != rnd.parent.checkpoint_sha256:
        raise ValueError("wrong parent checkpoint")
    parent = Model.load(parent_path)
    if parent.meta["task"] != "classification" or parent.meta["parameter_count"] != 4326:
        raise ValueError("unexpected A parent architecture")
    x, y, train_ids, train_projection = arrays(rnd.data.training_rows(), parent, provenance=True)
    vx, vy, val_ids, validation_projection = arrays(rnd.data.validation_rows(), parent, provenance=True)
    if train_ids & val_ids:
        raise ValueError("training and validation IDs overlap")
    parent_score, parent_prediction = score_projection(parent, vx, vy)
    run_seed = getattr(rnd, "run_seed", None)
    seed = parent.meta["seed"] if run_seed is None else run_seed
    model_sources = Path(baseline_model.__file__).resolve().parent
    reference = {"numpy": np.__version__, "model": file_digest(model_sources / "model.py"),
                 "data": file_digest(model_sources / "data.py")}
    trainer_id = fingerprint({"reference": reference, "source": getsource(trainer)})
    scorer_id = fingerprint({"reference": reference, "source": getsource(score_projection)})
    pack_id = fingerprint({"dataset": rnd.dataset_key, "task": parent.meta["task"],
                           "classes": parent.meta["classes"], "features": parent.meta["paths"]})
    projection_key = fingerprint({"validation": validation_projection, "scorer": scorer_id})
    workspace = Path(os.environ.get("CONTEST_WORK", "/workspace"))
    with TemporaryDirectory(prefix="school-a-", dir=workspace) as name:
        root = Path(name)
        state = rnd.state_workspace(root / "state")
        cache = RoundCache(state, rnd.round_index)
        decisions = state / "planner-history.json"
        previous = json.loads(decisions.read_text(encoding="utf-8")) if decisions.is_file() else []
        context = planner_context(rnd.round_index, parent, parent_score, y, previous)
        receipt = rnd.model(planner_messages(context), max_tokens=min(192, rnd.max_output_tokens),
                            temperature=0, enable_thinking=False)
        order = parse_order(receipt.text, context)
        fallback = order is None
        order = order or baseline_order(context)
        # Give malformed output a complete bounded round, and single-recipe plans a control.
        if len(order) == 1 and order[0] != "stable":
            order.append("stable")
        trials, artifacts = [], {}
        counts = dict(trained_candidates=0, reused_candidates=0, scored_candidates=0, reused_scores=0,
                      training_rows=0, optimizer_steps=0)
        for recipe in order:
            key = fingerprint({"pack": pack_id, "parent": parent_hash, "recipe": recipe,
                               "parameters": RECIPES[recipe], "training": train_projection,
                               "trainer": trainer_id, "seed": seed})
            cached = cache.training(key) if getattr(rnd, "reuse_candidates", True) else None
            candidate = None
            if cached is None:
                candidate, epochs = trainer(parent, x, y, recipe, seed)
                checkpoint = root / (recipe + ".npz")
                candidate.save(checkpoint)
                entry = cache.add_training(key, checkpoint, epoch_history=epochs, training_rows=len(x) * len(epochs),
                                           initial_steps=parent.meta["steps"], final_steps=candidate.meta["steps"])
                counts["trained_candidates"] += 1
            else:
                checkpoint, entry = cached
                counts["reused_candidates"] += 1
            evaluation = entry["scores"].get(projection_key)
            score_from = evaluation["round"] if evaluation is not None else None
            if evaluation is None:
                measured, predicted = score_projection(candidate or Model.load(checkpoint), vx, vy)
                cache.add_score(entry, projection_key, measured, array_digest(predicted, "<i8"))
                evaluation = entry["scores"][projection_key]
                counts["scored_candidates"] += 1
            else:
                counts["reused_scores"] += 1
            fit_rows = 0 if cached else entry["training_rows"]
            optimizer_steps = 0 if cached else entry["final_steps"] - entry["initial_steps"]
            counts["training_rows"] += fit_rows
            counts["optimizer_steps"] += optimizer_steps
            trials.append({"recipe": recipe, **evaluation["metrics"], "training_rows": entry["training_rows"],
                           "initial_steps": entry["initial_steps"], "final_steps": entry["final_steps"],
                           "actual_training_rows": fit_rows, "actual_optimizer_steps": optimizer_steps,
                           "cache": {"training_key": key, "score_key": fingerprint([key, projection_key]),
                                     "checkpoint_sha256": entry["checkpoint_sha256"],
                                     "prediction_sha256": evaluation["prediction_sha256"],
                                     "training_reused_from_round": entry["created_round"] if cached else None,
                                     "score_reused_from_round": score_from}})
            artifacts[recipe] = checkpoint, entry["epoch_history"]
        best = rank_trials(trials)[0]
        recipe = best["recipe"]
        checkpoint, epochs = artifacts[recipe]
        candidate_dir = root / recipe
        report = {"round": rnd.round_index, "parent_checkpoint_sha256": parent_hash, "seed": seed,
                  "checkpoint_sha256": best["cache"]["checkpoint_sha256"],
                  "parent": parent_score, "candidate": best, "epoch_history": epochs}
        materialize_candidate(candidate_dir, checkpoint, report)
        candidate_bytes = check_candidate_upload(candidate_dir, rnd.limits)
        config = {"recipe": recipe, "trials": trials, "planner_fallback": fallback,
                  "parent_checkpoint_sha256": parent_hash,
                  "parent_prediction_sha256": array_digest(parent_prediction, "<i8"),
                  "execution": {**counts, "cpu_seconds": process_time() - started_cpu}}
        reused = best["cache"]["training_reused_from_round"]
        change = (f"复用第 {reused} 轮的 {recipe} 检查点。" if reused is not None
                  else f"沿用权重、Adam 状态和归一化，完成 {recipe} 梯度续训。")
        submission = rnd.submit_candidate(
            candidate_dir, hypothesis="按公开验证的宏平均 F1 选择基于父检查点的训练配方。",
            change_summary=change,
            configuration=config, seed=seed, checkpoint_paths=["checkpoint.npz"],
            code_paths=["contest_submission.py", "school_cache.py", "baselines/model.py", "baselines/data.py",
                        "baselines/predict.py"],
            predict_argv=PREDICT_ARGV)
        measured = platform_score(submission, best["macro_f1"])
        gain = measured - parent_score["macro_f1"]
        accept = gain > 1e-9
        trace = {"context": context, "trials": trials, "accepted": accept, "chosen": recipe,
                 "execution": config["execution"],
                 "parent_macro_f1": parent_score["macro_f1"], "platform_macro_f1": measured}
        (state / f"round-{rnd.round_index:02d}.json").write_text(json.dumps(trace, allow_nan=False), encoding="utf-8")
        previous.extend({"recipe": trial["recipe"], "gain": round(trial["macro_f1"] - parent_score["macro_f1"], 5)}
                        for trial in trials)
        decisions.write_text(json.dumps(previous[-8:], allow_nan=False), encoding="utf-8")
        cache.persist(rnd.limits, candidate_bytes)
        rnd.save_state(state)
        rnd.finish(
            selected_artifact_id=submission["artifact_id"] if accept else rnd.parent.artifact_id,
            hypothesis="根据历史记录和公开验证比较续训候选。",
            change_summary=(f"比较 {', '.join(order)} 配方，新训练 {counts['trained_candidates']} 个，"
                            f"复用 {counts['reused_candidates']} 个。" + ("采纳最佳候选。" if accept else "保留父模型。")),
            reason=f"父模型 Macro-F1 {parent_score['macro_f1']:.6f}，候选 {measured:.6f}。模型回执 {receipt.call_id}。",
            configuration=config, seed=seed, history_ids=[event["id"] for event in history],
            model_call_ids=list(rnd.call_ids))


@lru_cache(maxsize=1)
def load_checkpoint(path):
    return Model.load(path)


def predict(sample, model_dir):
    checkpoint = Path(os.environ.get("CONTEST_ARTIFACT_PATH", Path(model_dir) / "checkpoint.npz"))
    model = load_checkpoint(checkpoint)
    if sample.task != "classification":
        raise ValueError("A requires classification")
    paths, values = flatten(sample.inputs)
    if paths != model.meta["paths"]:
        raise ValueError("prediction features do not match the checkpoint")
    scores = model.predict(np.asarray([values], dtype=np.float64))[0]
    best = int(scores.argmax())
    if np.count_nonzero(scores == scores[best]) > 1:
        scores[scores == scores[best]] = np.nextafter(scores[best], 0)
        scores[best] = np.nextafter(scores[best], 1)
    return [{"label": label, "score": float(value)} for label, value in zip(model.meta["classes"], scores)]
