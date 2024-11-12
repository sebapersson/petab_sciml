import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import pandas as pd

sys.path.insert(1, os.path.join(os.getcwd(), 'mkstd', "examples", "petab_sciml"))
sys.path.insert(1, os.path.join(os.getcwd(), 'test_cases'))
from petab_sciml_standard import Input, MLModel, PetabScimlStandard
from helper import read_array, get_ps_layer

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
dir_save = os.path.join(os.getcwd(), 'test_cases', "net_018")
net = Net()
mlmodel = MLModel.from_pytorch_module(
    module=net, mlmodel_id="model1", inputs=[Input(input_id="input1")]
)
petab_sciml_mlmodel = PetabScimlStandard.model(models=[mlmodel])
PetabScimlStandard.save_data(
    data=petab_sciml_mlmodel, filename=os.path.join(dir_save, "net.yaml")
)

for i in range(1, 4):
    layer_names = ["conv1", "conv2", "fc1", "fc2", "fc3"]
    for layer_name in layer_names:
        df = pd.read_csv(os.path.join(dir_save, "net_ps_" + str(i) + ".tsv"), delimiter='\t')
        ps_weight = get_ps_layer(df, layer_name, "weight")
        ps_bias = get_ps_layer(df, layer_name, "bias")
        with torch.no_grad():
            layer = getattr(net, layer_name)
            layer.weight[:] = ps_weight
            layer.bias[:] = ps_bias

    df_input = pd.read_csv(os.path.join(dir_save, "net_input_" + str(i) + ".tsv"), delimiter='\t')
    df_output = pd.read_csv(os.path.join(dir_save, "net_output_" + str(i) + ".tsv"), delimiter='\t')
    input = read_array(df_input)
    output_ref = read_array(df_output)
    output = net.forward(input)
    torch.testing.assert_close(output_ref, output, atol=1e-3, rtol=0.0)
