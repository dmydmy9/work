"""Official School A contract used by the entrypoint and local scorer."""

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np


@dataclass(frozen=True)
class SchoolProfile:
    key: str
    task: str
    metric: str
    metric_direction: str
    classes: tuple[str, ...]
    features: int
    parameters: int
    recipes: MappingProxyType
    descriptions: MappingProxyType
    preference: tuple[str, ...]

    def check_task(self, dataset, contract):
        if dataset != self.key or any(contract.get(name, value) != value for name, value in (
            ("task", self.task), ("metric", self.metric), ("metric_direction", self.metric_direction),
        )):
            raise ValueError(f"this submission requires the official profile {self.key} contract")

    def check_checkpoint(self, metadata):
        if (metadata["task"] != self.task or tuple(metadata["classes"]) != self.classes
                or len(metadata["paths"]) != self.features or metadata["parameter_count"] != self.parameters):
            raise ValueError(f"unexpected {self.key} parent architecture")

    def prediction(self, scores):
        best = int(scores.argmax())
        ties = scores == scores[best]
        if np.count_nonzero(ties) > 1:
            scores[ties] = np.nextafter(scores[best], 0)
            scores[best] = np.nextafter(scores[best], 1)
        return [{"label": label, "score": float(value)} for label, value in zip(self.classes, scores, strict=True)]


A = SchoolProfile(
    key="A", task="classification", metric="macro_f1", metric_direction="higher",
    classes=("Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"),
    features=128, parameters=4326,
    recipes=MappingProxyType({
        "stable": (4, 0.001, 64, False, 0.0),
        "balanced": (8, 0.0015, 64, True, 0.0),
        "explore": (20, 0.003, 64, True, 0.0),
        "polish": (12, 0.0006, 32, True, 0.0),
        "augment": (16, 0.0015, 64, True, 0.12),
    }),
    descriptions=MappingProxyType({
        "stable": "four short conservative epochs on original rows",
        "balanced": "eight epochs with tempered class resampling",
        "explore": "twenty longer epochs with tempered class resampling",
        "polish": "twelve low-rate epochs with smaller batches and class resampling",
        "augment": "sixteen epochs with class resampling and coherent sensor gains",
    }),
    preference=("balanced", "augment", "explore", "polish", "stable"),
)
