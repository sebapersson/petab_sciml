"""Constants for reserved keywords or useful lists."""

from __future__ import annotations

__all__ = [
    "METADATA",
    "DATA",
    "CONDITION_IDS",
    "INPUTS",
    "PARAMETERS",
    "ALL_CONDITION_IDS",
    "ARRAY",
    "SUPPORTED_LAYERS",
    "SUPPORTED_ACTIVATIONS",
    "SUPPORTED_TENSOR_OPS",
]


# --- Array data file (HDF5) -------------------------------------------------

#: Metadata group.
METADATA = "metadata"
#: Name of a dataset.
DATA = "data"
#: Condition IDs field.
CONDITION_IDS = "conditionIds"
#: Inputs group.
INPUTS = "inputs"
#: Parameters group.
PARAMETERS = "parameters"
#: Condition IDs field value when the array data applies to all conditions.
ALL_CONDITION_IDS = "0"


# --- Reserved keywords ------------------------------------------------------

#: Indicate that a variable value is an array.
ARRAY = "array"


# --- NN model YAML format ---------------------------------------------------

#: Supported layers.
SUPPORTED_LAYERS = [
    "Linear",
    "Bilinear",
    "Flatten",
    "Dropout",
    "Dropout1d",
    "Dropout2d",
    "Dropout3d",
    "AlphaDropout",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "MaxPool1d",
    "MaxPool2d",
    "MaxPool3d",
    "AvgPool1d",
    "AvgPool2d",
    "AvgPool3d",
    "LPPool1d",
    "LPPool2d",
    "LPPool3d",
    "AdaptiveMaxPool1d",
    "AdaptiveMaxPool2d",
    "AdaptiveMaxPool3d",
    "AdaptiveAvgPool1d",
    "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d",
]

#: Supported activation functions.
SUPPORTED_ACTIVATIONS = [
    "relu",
    "relu6",
    "hardtanh",
    "hardswish",
    "selu",
    "leaky_relu",
    "gelu",
    "tanhshrink",
    "softsign",
    "softplus",
    "tanh",
    "sigmoid",
    "hardsigmoid",
    "silu",
    "mish",
    "elu",
    "celu",
    "softmax",
    "log_softmax",
]

#: These are handled by ``torch`` rather than ``torch.nn`` in the pytorch
#: compatibility code.
SUPPORTED_TENSOR_OPS = frozenset({"flatten", "cat"})
