"""Model registry for every comparable person-node classifier."""

from typing import Mapping

from .compgcn import RelationalCompGCNClassifier
from .features import FeatureSpec
from .rgat import RelationalGATClassifier
from .rgcn import RelationalGCNClassifier

MODEL_REGISTRY = {
    "rgat": RelationalGATClassifier,
    "rgcn": RelationalGCNClassifier,
    "compgcn": RelationalCompGCNClassifier,
}


def build_model(name: str, **kwargs):
    try:
        return MODEL_REGISTRY[name.lower()](**kwargs)
    except KeyError as error:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model {name!r}. Available models: {available}") from error


__all__ = [
    "FeatureSpec",
    "RelationalGATClassifier",
    "RelationalGCNClassifier",
    "RelationalCompGCNClassifier",
    "MODEL_REGISTRY",
    "build_model",
]
