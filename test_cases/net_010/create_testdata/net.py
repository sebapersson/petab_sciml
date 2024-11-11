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
        self.flatten1 = nn.Flatten()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.flatten1(input)
        return x

# Create a pytorch module, convert it to PEtab SciML, then save it to disk.
dir_save = os.path.join(os.getcwd(), 'test_cases', "net_010")
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
    df_input = pd.read_csv(os.path.join(dir_save, "net_input_" + str(i) + ".tsv"), delimiter='\t')
    df_output = pd.read_csv(os.path.join(dir_save, "net_output_" + str(i) + ".tsv"), delimiter='\t')
    input = read_array(df_input)
    output_ref = read_array(df_output)
    output = net.forward(input)
    torch.testing.assert_close(output_ref, output, atol=1e-3, rtol=0.0)
