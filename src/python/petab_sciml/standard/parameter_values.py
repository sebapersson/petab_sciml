from __future__ import annotations

from pydantic import BaseModel, Field

from mkstd import Hdf5Standard
from mkstd.types.array import get_array_type


__all__ = [
    "SingleParameterValue",
    "SingleLayerParameterValues",
    "ParameterValues",
    "ParameterValuesStandard",
]


class SingleParameterValue(BaseModel):
    """The value for a single parameter in an MLModel layer.

    For example, the weight matrix of a linear layer is considered a single
    parameter.
    """

    framework_parameter_name: str
    """The name of the parameter in the framework you are working in.

    For example, in PyTorch, this could be "weight" or "bias" for a
    Linear module.
    """
    value: get_array_type()
    """The (tensor) value.

    For example, with a PyTorch linear module, this could be the full weight
    matrix.
    """


class SingleLayerParameterValues(BaseModel):
    """The values for the parameters in a single model layer."""

    layer_id: str
    """The layer ID."""
    values: list[SingleParameterValue]
    """The values for multiple layer parameters."""


class ParameterValues(BaseModel):
    """Parameter values for an ML model."""

    root: list[SingleLayerParameterValues]
    """The parameter values."""


ParameterValuesStandard = Hdf5Standard(model=ParameterValues)


if __name__ == "__main__":
    from pathlib import Path


    ParameterValuesStandard.save_schema(Path(__file__).resolve().parents[4] / "docs" / "src" / "assets" / "parameter_values_schema.json")
