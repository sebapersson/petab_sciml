from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from mkstd import Hdf5Standard
from mkstd.types.array import get_array_type


__all__ = [
    "Array",
    "Metadata",
    "Arrays",
    "ArrayStandard",
]


class Array(BaseModel):
    """An array.
    
    For example, input data, or layer weights.
    """

    conditionIds: list[str] | None = Field(default=None)
    """The dataset is used with these conditions.

    The default (`None`) indicates all conditions.
    """

    targetId: str | None = Field(default=None)
    """The Id for the target that will be assigned the tensor value.

    If this is not provided, then the target must be assigned via the
    NN YAML file.
    """

    targetValue: get_array_type() | str
    """The data."""

    # Post-constructor validation: ensure no duplicates in the condition IDs?


class Metadata(BaseModel):
    """Metadata for array(s)."""

    perm: Literal["row", "column"]
    """The order of the dimensions of arrays.

    i.e., row-major or column-major arrays.
    """


class Arrays(BaseModel):
    """Multiple arrays.

    For example, data for different inputs for different conditions,
    or values for different parameters of different layers.
    """

    metadata: Metadata
    """Additional metadata for the input data."""

    arrays: list[Array]
    """The array."""


ArraysStandard = Hdf5Standard(model=Arrays)


if __name__ == "__main__":
    from pathlib import Path


    ArraysStandard.save_schema(
        Path(__file__).resolve().parents[4]
        / "docs" / "src" / "assets" / "arrays_schema.json"
    )
