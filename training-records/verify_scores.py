"""Recompute the RC03 evidence scores from saved predictions and official labels."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from math import isclose, isfinite, sqrt
from pathlib import Path
from statistics import mean


CLASSES = ("Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene")


def read_json(path):
    return json.loads(path.read_bytes())


def rows(path):
    with path.open(encoding="utf-8") as stream:
        yield from map(json.loads, stream)


def select_labels(path, ids=None):
    answers = {row["id"]: row["answer"] for row in rows(path)}
    return answers if ids is None else {identity: answers[identity] for identity in ids}


def score(path, labels, classification=False):
    if classification:
        actual, guessed, correct = Counter(), Counter(), Counter()
        for (identity, answer), prediction in zip(labels.items(), rows(path), strict=True):
            assert prediction["id"] == identity and answer in CLASSES
            confidences = {item["label"]: item["score"] for item in prediction["outputs"]}
            assert set(confidences) == set(CLASSES) and all(0 <= value <= 1 for value in confidences.values())
            maximum = max(confidences.values())
            label, = (name for name, value in confidences.items() if value == maximum)
            actual[answer] += 1
            guessed[label] += 1
            correct[answer] += label == answer
        return mean(2 * correct[name] / (actual[name] + guessed[name]) if actual[name] + guessed[name] else 0
                    for name in CLASSES)
    errors = []
    for (identity, answer), prediction in zip(labels.items(), rows(path), strict=True):
        assert prediction.get("id", identity) == identity
        value = prediction["outputs"]["value"]
        assert isfinite(value) and value >= 0
        errors.append((value - answer)**2)
    return sqrt(mean(errors))


def reconcile(evidence, school):
    evidence_hashes = read_json(evidence / "EVIDENCE_SHA256.json")
    for name, expected in evidence_hashes.items():
        path = (evidence / name).resolve()
        assert path.is_relative_to(evidence) and sha256(path.read_bytes()).hexdigest() == expected, name
    registration = read_json(evidence / "verified.json")
    a_protocol = read_json(evidence / "history/a-protocol.json")["d2"]
    a_ids = a_protocol["development"]["ids"]
    b_ids = read_json(evidence / "history/b-split-protocol.json")["projections"]["development"]["ids"]
    labels = {"A": select_labels(school / "A/public/validation.jsonl", a_ids),
              "B": select_labels(school / "B/public/train.jsonl", b_ids),
              "C": select_labels(school / "C/public/validation.jsonl")}
    arms = {}
    for profile, folds in registration["packages"].items():
        for arm, reference in folds.items():
            directory = evidence / f"verify-{profile.lower()}-{arm}"
            parent = "sha256:" + sha256((school / profile / "checkpoint.npz").read_bytes()).hexdigest()
            baseline_path = directory / ("official-predict/stdout.jsonl" if profile == "B" else "official-predictions.jsonl")
            baseline = score(baseline_path, labels[profile], profile == "A")
            assert isclose(baseline, reference["official"], abs_tol=1e-9)
            current, fitted = baseline, []
            for index, row in enumerate(reference["rounds"], start=1):
                assert index == row["round"]
                if parent != row["checkpoint_sha256"]:
                    parent = row["checkpoint_sha256"]
                    name = "candidate-predict/stdout.jsonl" if profile == "B" else "candidate-predictions.jsonl"
                    current = score(directory / f"round-{index:02d}" / name, labels[profile], profile == "A")
                assert isclose(current, row["score"], abs_tol=1e-9), (profile, arm, index)
                fitted.append(current)
            assert len(fitted) == 10 and parent == reference["final_checkpoint_sha256"]
            assert isclose(mean(fitted), reference["ten_round_mean"], abs_tol=1e-9)
            assert isclose(fitted[-1], reference["final"], abs_tol=1e-9)
            arms[f"{profile}-{arm}"] = {"rows": len(labels[profile]), "official": baseline,
                                         "ten_round_mean": mean(fitted), "final": fitted[-1]}

    gate = read_json(evidence / "pre-gate/manifest.json")
    candidates = 0
    checked = {}
    for arm in gate["arms"].values():
        profile = arm["profile"]
        for trial in arm["candidates"]:
            name = trial["predictions"]
            path = evidence / "pre-gate" / name
            if name not in checked:
                assert sha256(path.read_bytes()).hexdigest() == trial["predictions_sha256"].removeprefix("sha256:")
                checked[name] = score(path, labels[profile], profile == "A")
            metric = "macro_f1" if profile == "A" else "rmse"
            assert isclose(checked[name], trial["metrics"][metric], abs_tol=1e-9)
            candidates += 1

    public_b = select_labels(school / "B/public/validation.jsonl")
    b_initial = score(evidence / "verify-b-public/official/stdout.jsonl", public_b)
    b_final = score(evidence / "verify-b-public/stdout.jsonl", public_b)
    assert len(public_b) == registration["B_public_review"]["rows"]
    assert isclose(b_final, registration["B_public_review"]["metrics"]["rmse"], abs_tol=1e-9)
    review_a = evidence / "verify-a-review"
    a_review_stats = read_json(review_a / "report.json")
    a_labels = select_labels(school / "A/public/validation.jsonl", a_protocol["review"]["ids"])
    assert len(a_labels) == a_review_stats["rows"] and set(a_labels).isdisjoint(a_ids)
    a_scores = {}
    for arm in ("official", "fixed", "dynamic"):
        predictions = review_a / f"{arm}-predictions.jsonl"
        assert sha256(predictions.read_bytes()).hexdigest() == a_review_stats["prediction_sha256"][arm]
        observed = score(predictions, a_labels, classification=True)
        assert isclose(observed, a_review_stats["metrics"][arm]["macro_f1"], abs_tol=1e-9)
        a_scores[arm] = observed
    return {"verified_files": len(evidence_hashes), "arms": arms, "A_review": {"rows": len(a_labels), "macro_f1": a_scores},
            "B_public": {"rows": len(public_b),
             "official_rmse": b_initial, "final_rmse": b_final, "relative_gain": 1 - b_final / b_initial},
            "pre_gate_trials": candidates, "pre_gate_unique_prediction_files": len(checked)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_directory", type=Path)
    parser.add_argument("official_packs", type=Path, help="directory containing official A, B and C folders")
    args = parser.parse_args()
    result = reconcile(args.evidence_directory.resolve(), args.official_packs.resolve())
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
