import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from src.python.helper import make_yaml, test_nn

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(2, 5)
        self.layer2 = nn.Linear(5, 1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.layer1(input)
        x = F.leaky_relu(x)
        out = self.layer2(x)
        return out

# Create a pytorch module, convert it to PEtab SciML, then save it to disk.
dir_save = os.path.join(os.getcwd(), 'test_cases', 'net_import', "028")
net = Net()
make_yaml(net, dir_save)
test_nn(net, dir_save, ["layer1", "layer2"])
