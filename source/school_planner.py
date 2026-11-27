"""Bounded experiment plans and parent-specific evidence for the school entry."""

import json
from school_profile import A


SYSTEM = (
    "Plan one round of continued training from the supplied parent checkpoint. "
    "Use the visible confusion, available action descriptions, and measured history. "
    "Actions already measured on this parent under the same projection will reuse their checkpoints. "
    "Prior-parent outcomes are guidance about recipes, not measurements on this parent. "
    "If fresh training cannot justify its cost, return an empty actions array to reuse existing results. "
    "Return only JSON with four fields: hypothesis (short text), actions (0 to 3 distinct available IDs), "
    "expected (class or metric and direction, short text), failure (measurable failure criterion, short text). "
    "Do not invent actions, IDs, model changes, or measurements."
)
DESCRIPTIONS = A.descriptions
PREFERENCE = A.preference
# Leave room for the measured metrics and receipt in SDK declarations (4,000 UTF-8 bytes).
PLAN_TEXT_BYTES = 3000


def planner_messages(context, system=SYSTEM):
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(context, sort_keys=True, separators=(",", ":"), allow_nan=False)}]


def parse_plan(text, context):
    if not isinstance(text, str):
        return None

    def unique(pairs):
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError("duplicate JSON key")
        return result

    try:
        plan = json.loads(text.strip().removesuffix("<|im_end|>"), object_pairs_hook=unique)
    except ValueError:
        return None
    if not isinstance(plan, dict) or set(plan) != {"hypothesis", "actions", "expected", "failure"}:
        return None
    statements = [plan[key] for key in ("hypothesis", "expected", "failure")]
    if any(not isinstance(statement, str) or not statement.strip() for statement in statements):
        return None
    try:
        if sum(len(statement.encode("utf-8")) for statement in statements) > PLAN_TEXT_BYTES:
            return None
    except UnicodeError:
        return None
    actions = plan["actions"]
    if (not isinstance(actions, list) or len(actions) > context["budget"]["actions"]
            or any(not isinstance(name, str) or name not in context["available"] for name in actions)
            or len(set(actions)) != len(actions)):
        return None
    return plan


def history_evidence(events, *, parent_hash, train_key, evaluation_key, seed, compact=False):
    current = {}
    groups = {}
    for event in reversed(events):
        same = (event["parent"] == parent_hash and event["training"] == train_key
                and event["validation"] == evaluation_key and event["seed"] == seed)
        if same:
            destination = current
        elif event["parent"] != parent_hash:
            destination = groups.setdefault(event["parent"], {})
        else:
            continue
        for outcome in event["outcomes"]:
            destination.setdefault(outcome["action"], outcome)
    def for_planner(records, limit):
        rows = list(records.values())[:limit]
        if not compact:
            return rows
        return [{"action": row["action"], "gain": round(row["gain"], 3),
                 "stage_safe": row.get("stage_safe", True), "first_round": row["first_round"]} for row in rows]

    return {
        "current": for_planner(current, 6 if compact else 12),
        "prior": [{"parent": identity[7:19] if compact else identity,
                   "outcomes": for_planner(group, 2 if compact else 4)}
                  for identity, group in list(groups.items())[:2]],
    }


def decision_context(profile, *, round_index, parent_hash, parent_steps, parent_score, train_counts,
                     available, preference, evidence, rounds, wall_seconds, max_tokens,
                     actions=3):
    class_index = {name: i for i, name in enumerate(profile.classes)}
    acetaldehyde = class_index["Acetaldehyde"]
    ethanol = class_index["Ethanol"]
    confusion = parent_score["confusion"]
    return {
        "profile": profile.key, "round": round_index, "parent": parent_hash, "parent_steps": parent_steps,
        "parent_macro_f1": round(parent_score[profile.metric], 5),
        "classes": list(profile.classes), "train_counts": train_counts,
        "class_f1": [round(value, 5) for value in parent_score["class_f1"]],
        "confusion": {"Acetaldehyde_to_Ethanol": confusion[acetaldehyde][ethanol],
                      "Acetaldehyde_support": sum(confusion[acetaldehyde])},
        "history": evidence, "available": available, "preference": list(preference),
        "budget": {"actions": actions, "rounds_remaining": max(0, rounds - round_index + 1),
                   "wall_seconds": wall_seconds, "max_output_tokens": max_tokens},
    }


def rank_untried_actions(context):
    history = context["history"]
    tested = {item["action"] for item in history["current"]}
    prior_gains = {name: 0.0 for name in context["available"]}
    for group in history["prior"]:
        for outcome in group["outcomes"]:
            if outcome["action"] in prior_gains and outcome.get("stage_safe", True):
                prior_gains[outcome["action"]] += outcome["gain"]
    preference = context["preference"]
    candidates = sorted((name for name in preference if name not in tested),
                        key=lambda name: (-prior_gains[name], preference.index(name)))
    return candidates[:context["budget"]["actions"]], tested


def fallback_plan(context):
    actions, tested = rank_untried_actions(context)
    history = context["history"]
    if not actions:
        return {"hypothesis": "Existing trials exhaust the current parent search budget.",
                "actions": [], "expected": "Maintain Macro-F1 with zero new training.",
                "failure": "A cached candidate exceeds the parent by the adoption margin."}
    rounds = sorted({outcome["first_round"] for outcome in history["current"]})
    hypothesis = (f"Trials from rounds {rounds} already measured {len(tested)} actions on this parent. "
                  "Measure remaining choices." if rounds else "Screen plausible recipes on the current parent.")
    return {"hypothesis": hypothesis, "actions": actions,
            "expected": "Improve public Macro-F1 without losing minority-class accuracy.",
            "failure": "No candidate clears the Macro-F1 adoption margin."}
