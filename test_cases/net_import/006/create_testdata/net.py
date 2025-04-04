import torch
import torch.nn as nn
import os

from src.python.helper import make_yaml, test_nn

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.ConvTranspose1d(1, 2, 5)
        self.layer2 = nn.ConvTranspose1d(2, 1, 5)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.layer1(input)
        x = self.layer2(x)
        return x

# Create a pytorch module, convert it to PEtab SciML, then save it to disk.
dir_save = os.path.join(os.getcwd(), 'test_cases', 'net_import', "006")
net = Net()
make_yaml(net, dir_save)
input = torch.ones(1, 10)
out = net.forward(input)
test_nn(net, dir_save, ["layer1", "layer2"])
