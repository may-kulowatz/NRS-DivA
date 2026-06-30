from .metrics_protocols import (
    RootMeanSquaredError,
    MetricEvaluator,
    AccuracyScore,
    LogLossScore,
    NdcgScore,
    AucScore,
    F1Score,
    MrrScore,
)

from ._beyond_accuracy import *
from ._classification import *
from ._ranking import *
from ._sklearn import *

# The model classes (LSTUR/NRMS/...) pull in TensorFlow, which is heavy and not
# needed for the metric/helper utilities above (e.g. _beyond_accuracy, utils).
# Import them lazily via PEP 562 so that merely importing this package does NOT
# drag in TensorFlow — it is loaded only when a model class is actually accessed.
# `from data.datasets.ebnerd.utils import NRMSModel` keeps working; the TF import
# just happens at that point instead of at package import.
_LAZY_MODELS = {
    "LSTURModel": ".lstur",
    "NRMSModel": ".nrms",
    "NPAModel": ".npa",
    "NAMLModel": ".naml",
}


def __getattr__(name):
    """Lazily import the TensorFlow-backed model classes on first access (PEP 562)."""
    module_path = _LAZY_MODELS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_path, __name__)
    return getattr(module, name)


def __dir__():
    return sorted([*globals(), *_LAZY_MODELS])