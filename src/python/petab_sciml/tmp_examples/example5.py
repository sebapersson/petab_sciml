import torch
import torch.nn as nn
from petab_sciml_standard import Input, MLModel, PetabScimlStandard


class Net(nn.Module):
    """Example network with LayerNorm and tuple argument."""

    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm((4, 10, 11, 12))
        self.layer1 = nn.Conv3d(4, 1, 5)
        self.flatten1 = nn.Flatten()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Execute the computational graph."""
        x = self.norm1(input)
        x = self.layer1(x)
        x = self.flatten1(x)
        return x


# Create a pytorch module, convert it to PEtab SciML, then save it to disk.
net0 = Net()

mlmodel0 = MLModel.from_pytorch_module(
    module=net0, mlmodel_id="model1", inputs=[Input(input_id="input1")]
)
petab_sciml_models0 = PetabScimlStandard.model(models=[mlmodel0])
PetabScimlStandard.save_data(
    data=petab_sciml_models0, filename="data5/models1.yaml"
)
