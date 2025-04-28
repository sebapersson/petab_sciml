from __future__ import annotations

from pydantic import BaseModel, Field

from mkstd import Hdf5Standard
from mkstd.types.array import get_array_type


__all__ = [
    "Data",
    "SingleInputData",
    "InputMetadata",
    "InputData",
    "InputDataStandard",
]


class Data(BaseModel):
    """A dataset."""

    condition_ids: list[str] | None = Field(default=None)
    """The dataset is used with these conditions.

    The default (`None`) indicates all conditions.
    """

    data: get_array_type() | str
    """The data.

    Either the data tensor directly, or the filename (e.g. an image file).
    """

    # Post-constructor validation: ensure no duplicates in the condition IDs?


class SingleInputData(BaseModel):
    """Datasets for an input."""

    input_id: str
    """The input ID."""
    datasets: list[Data]
    """The datasets for the input."""

    # Post-constructor validation: ensure no duplicates in the condition IDs,
    # across all datasets?


class InputMetadata(BaseModel):
    """Input array metadata."""

    perm: Literal["row", "column"]
    """The order of the dimensions of arrays.

    i.e., row-major or column-major arrays.
    """


class InputData(BaseModel):
    """Datasets for inputs."""

    metadata: InputMetadata
    """Additional metadata for the input data."""
    inputs: list[SingleInputData]
    """The datasets."""


InputDataStandard = Hdf5Standard(model=InputData)


if __name__ == "__main__":
    from pathlib import Path


    InputDataStandard.save_schema(Path(__file__).resolve().parents[4] / "docs" / "src" / "assets" / "input_data_schema.json")
