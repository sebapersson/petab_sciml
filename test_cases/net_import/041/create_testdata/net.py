import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from src.python.helper import make_yaml, test_nn

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(2, 5)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = F.log_softmax(input, dim=0)
        return out

# Create a pytorch module, convert it to PEtab SciML, then save it to disk.
dir_save = os.path.join(os.getcwd(), 'test_cases', 'net_import', "041")
net = Net()
make_yaml(net, dir_save)
test_nn(net, dir_save, None)
