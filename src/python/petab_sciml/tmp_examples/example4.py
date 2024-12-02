import torch
import torch.nn as nn
from petab_sciml_standard import Input, MLModel, PetabScimlStandard


class Net1(nn.Module):
    """Example network with BatchNorm."""

    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.RNN(5, 10, 3)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Define the computational graph."""
        x = self.layer1(input)
        return x


# Create a pytorch module, convert it to PEtab SciML, then save it to disk.
net0 = Net1()
input_ = torch.ones(1, 5, 5)
net0.forward(input_)
mlmodel0 = MLModel.from_pytorch_module(
    module=net0, mlmodel_id="model1", inputs=[Input(input_id="input1")]
)
petab_sciml_models0 = PetabScimlStandard.model(models=[mlmodel0])
PetabScimlStandard.save_data(
    data=petab_sciml_models0, filename="data4/models1.yaml"
)
