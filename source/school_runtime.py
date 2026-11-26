"""Shared SDK planning, candidate execution, and artifact handling."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from school_cache import array_digest, fingerprint
from school_planner import parse_plan


def request_plan(rnd, context, messages, fallback, max_tokens):
    receipt = rnd.model(messages(context), max_tokens=max_tokens, temperature=0, enable_thinking=False)
    if not getattr(receipt, "call_id", None) or receipt.call_id not in rnd.call_ids:
        raise RuntimeError("planner response has no current-round SDK receipt")
    parsed = parse_plan(receipt.text, context)
    return receipt, parsed if parsed is not None else fallback(context), parsed is None


def run_trials(rnd, root, cache, order, recipes, *, reuse_only, pack_id, parent_hash, parent, seed,
               train_projection, projection_key, trainer_id, trainer, score_projection, x, y, vx, vy,
               load_checkpoint, prediction_dtype):
    trials, artifacts = [], {}
    counts = dict(trained_candidates=0, reused_candidates=0, scored_candidates=0, reused_scores=0,
                  training_rows=0, optimizer_steps=0)
    for recipe in order:
        key = fingerprint({"pack": pack_id, "parent": parent_hash, "recipe": recipe,
                           "parameters": recipes[recipe], "training": train_projection,
                           "trainer": trainer_id, "seed": seed})
        cached = cache.training(key) if getattr(rnd, "reuse_candidates", True) else None
        if reuse_only and cached is None:
            continue
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
            measured, predicted = score_projection(candidate or load_checkpoint(checkpoint), vx, vy)
            cache.add_score(entry, projection_key, measured, array_digest(predicted, prediction_dtype))
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
                                 "score_first_round": evaluation["round"],
                                 "training_reused_from_round": entry["created_round"] if cached else None,
                                 "score_reused_from_round": score_from}})
        artifacts[recipe] = checkpoint, entry["epoch_history"]
    return trials, artifacts, counts


def metric_gain(parent, candidate, direction):
    if direction == "higher":
        return candidate - parent
    if direction == "lower":
        return parent - candidate
    raise ValueError(f"unsupported metric direction {direction}")


def write_candidate(directory, checkpoint, report, *, source_dir, baseline_dir, code_files):
    directory.mkdir()
    shutil.copyfile(checkpoint, directory / "checkpoint.npz")
    (directory / "baselines").mkdir()
    for filename in ("model.py", "data.py", "predict.py", "__init__.py"):
        shutil.copyfile(baseline_dir / filename, directory / "baselines" / filename)
    for filename in code_files:
        shutil.copyfile(source_dir / filename, directory / filename)
    (directory / "training.json").write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
