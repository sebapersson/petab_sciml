import torch.nn as nn
import torch.nn.functional as F
import os
from src.python.helper import make_yaml

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Conv2d(3, 1, (5, 5))
        self.layer2 = nn.Flatten()
        self.layer3 = nn.Linear(in_features=36, out_features=1)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.relu(x)
        return x

dir_save = os.path.join(os.getcwd(), 'test_cases', "015", "petab")
net = Net()
make_yaml(net, dir_save, net_name="net1.yaml")
