"""Route the common contest SDK entry to each official dataset."""

import sys

from a_frozen import contest_submission as pack_a
import school_b_submission as pack_b
import school_c_submission as pack_c


TASKS = ("classification", "regression")
HANDLERS = {"A": pack_a, "B": pack_b, "C": pack_c}
INPUT_FIELDS = {frozenset({"features"}): "B", frozenset({"settings", "sensors"}): "C"}


def train(rnd):
    try:
        if rnd.dataset_key == "B":
            pack_b.train(rnd, city_screen=True)
        else:
            HANDLERS[rnd.dataset_key].train(rnd)
    except Exception as error:
        print(f"Training failure in pack {rnd.dataset_key}, round {rnd.round_index}: {type(error).__name__}",
              file=sys.stderr)
        raise


def predict(sample, model_dir):
    profile = sample.dataset_key
    if profile is None:
        # The SDK permits legacy, context-free predictions. Each pack also validates
        # the selected checkpoint before it can return a value.
        profile = "A" if sample.task == "classification" else INPUT_FIELDS.get(frozenset(sample.inputs))
    if profile not in HANDLERS:
        raise ValueError("unknown prediction dataset")
    return HANDLERS[profile].predict(sample, model_dir)
