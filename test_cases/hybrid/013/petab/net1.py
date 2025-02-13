import os
import torch.nn as nn
import torch.nn.functional as F
from src.python.helper import make_yaml


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(in_features=2, out_features=5)
        self.layer2 = nn.Linear(in_features=5, out_features=5)
        self.layer3 = nn.Linear(in_features=5, out_features=1)
    def forward(self, x):
        x = self.layer1(x)
        x = F.tanh(x)
        x = self.layer2(x)
        x = F.tanh(x)
        x = self.layer3(x)
        return x

dir_save = os.path.join(os.getcwd(), 'test_cases', "009", "petab")
net = Net()
make_yaml(net, dir_save, net_name="net1.yaml")