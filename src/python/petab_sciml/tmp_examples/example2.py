import torch
import torch.nn as nn
import torch.nn.functional as F
from petab_sciml_standard import Input, MLModel, PetabScimlStandard


class Net(nn.Module):
    """Single layer."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Define the computational graph."""
        x = self.conv1(input)
        x = F.relu(x)
        return x


# Create a pytorch module, convert it to PEtab SciML, then save it to disk.
net0 = Net()
mlmodel0 = MLModel.from_pytorch_module(
    module=net0, mlmodel_id="model1", inputs=[Input(input_id="input1")]
)
petab_sciml_models0 = PetabScimlStandard.model(models=[mlmodel0])
PetabScimlStandard.save_data(
    data=petab_sciml_models0, filename="data2/models0.yaml"
)

# Read the stored model from disk, reconstruct the pytorch module
loaded_petab_sciml_models = PetabScimlStandard.load_data("data2/models0.yaml")
net1 = loaded_petab_sciml_models.models[0].to_pytorch_module()

print(net1.code)  # noqa: T201

# Store the pytorch module to disk again and verify that the round-trip was successful
mlmodel1 = MLModel.from_pytorch_module(
    module=net1, mlmodel_id="model1", inputs=[Input(input_id="input1")]
)
petab_sciml_models1 = PetabScimlStandard.model(models=[mlmodel1])
PetabScimlStandard.save_data(
    data=petab_sciml_models1, filename="data2/models1.yaml"
)

with open("data2/models0.yaml") as f:
    data0 = f.read()
with open("data2/models1.yaml") as f:
    data1 = f.read()


if not data0 == data1:
    raise ValueError(
        "The round-trip of saving the pytorch modules to disk failed."
    )
