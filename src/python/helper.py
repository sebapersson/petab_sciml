import torch
import sys
import os
import h5py

sys.path.insert(1, os.path.join(os.getcwd(), 'mkstd', "examples", "petab_sciml"))
from petab_sciml_standard import Input, MLModel, PetabScimlStandard

def make_yaml(net, dir_save, net_name="net.yaml"):
    mlmodel = MLModel.from_pytorch_module(
    module=net, mlmodel_id="model1", inputs=[Input(input_id="input1")])
    petab_sciml_mlmodel = PetabScimlStandard.model(models=[mlmodel])
    PetabScimlStandard.save_data(
        data=petab_sciml_mlmodel, filename=os.path.join(dir_save, net_name)
    )

def test_nn(net, dir_save, layer_names, dropout=False, atol=1e-3):
    for i in range(1, 4):
        if not layer_names is None:
            for layer_name in layer_names:
                layer = getattr(net, layer_name)
                ps_h5 = h5py.File(os.path.join(dir_save, "net_ps_" + str(i) + ".hdf5"), "r")
                ps_weight = ps_h5[layer_name]["weight"][:]
                with torch.no_grad():
                    layer.weight[:] = torch.from_numpy(ps_weight)
                if hasattr(layer, "bias") and (not layer.bias is None):
                    ps_bias = ps_h5[layer_name]["bias"][:]
                    with torch.no_grad():
                        layer.bias[:] = torch.from_numpy(ps_bias)

        input_h5 = h5py.File(os.path.join(dir_save, "net_input_" + str(i) + ".hdf5"))
        output_h5 = h5py.File(os.path.join(dir_save, "net_output_" + str(i) + ".hdf5"))
        input = torch.from_numpy(input_h5["input"][:])
        output_ref = torch.from_numpy(output_h5["output"][:])
        if dropout == False:    
            output = net.forward(input)
        else:
            output = torch.zeros_like(output_ref)
            for i in range(50000):
                output += net.forward(input)
            output /= 50000
        torch.testing.assert_close(output_ref, output, atol=atol, rtol=0.0)
