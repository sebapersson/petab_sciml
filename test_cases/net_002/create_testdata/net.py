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
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(2, 5)
        self.layer2 = nn.Linear(2, 10)
        self.layer3 = nn.Bilinear(5, 10, 2)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x1 = self.layer1(input)
        x2 = self.layer2(input)
        out = self.layer3(x1, x2)
        return out

# Create a pytorch module, convert it to PEtab SciML, then save it to disk.
dir_save = os.path.join(os.getcwd(), 'test_cases', "net_002")
net = Net()
mlmodel = MLModel.from_pytorch_module(
    module=net, mlmodel_id="model1", inputs=[Input(input_id="input1")]
)
petab_sciml_mlmodel = PetabScimlStandard.model(models=[mlmodel])
PetabScimlStandard.save_data(
    data=petab_sciml_mlmodel, filename=os.path.join(dir_save, "net.yaml")
)

# Test consistency between Lux.jl and PyTorch (a double check for correct values)
for i in range(1, 4):
    layer_names = ["layer1", "layer2", "layer3"]
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
