import torch
import torch.nn as nn
import os

from src.python.helper import make_yaml, test_nn

class Net(nn.Module):
    """Example network.
    Ref: https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
    """
    def __init__(self) -> None:
        super().__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.max_pool1 = nn.MaxPool2d((2, 2))
        self.fc1 = nn.Linear(64, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.flatten1 = nn.Flatten()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        c1 = self.conv1(input)
        s2 = self.max_pool1(c1)
        c3 = self.conv2(s2)
        s4 = self.max_pool1(c3)
        s4 = self.flatten1(s4)
        f5 = self.fc1(s4)
        f6 = self.fc2(f5)
        output = self.fc3(f6)
        return output

# Create a pytorch module, convert it to PEtab SciML, then save it to disk.
dir_save = os.path.join(os.getcwd(), 'test_cases', 'net_import', "018")
net = Net()
make_yaml(net, dir_save)
test_nn(net, dir_save, ["conv1", "conv2", "fc1", "fc2", "fc3"])
