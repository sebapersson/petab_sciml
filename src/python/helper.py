import torch
import sys
import pandas as pd
import os

sys.path.insert(1, os.path.join(os.getcwd(), 'mkstd', "examples", "petab_sciml"))
from petab_sciml_standard import Input, MLModel, PetabScimlStandard

def make_yaml(net, dir_save):
    mlmodel = MLModel.from_pytorch_module(
    module=net, mlmodel_id="model1", inputs=[Input(input_id="input1")])
    petab_sciml_mlmodel = PetabScimlStandard.model(models=[mlmodel])
    PetabScimlStandard.save_data(
        data=petab_sciml_mlmodel, filename=os.path.join(dir_save, "net.yaml")
    )

def test_nn(net, dir_save, layer_names, dropout=False, atol=1e-3):
    for i in range(1, 4):
        if not layer_names is None:
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
        if dropout == False:    
            output = net.forward(input)
        else:
            output = torch.zeros_like(output_ref)
            for i in range(40000):
                output += net.forward(input)
            output /= 40000
        torch.testing.assert_close(output_ref, output, atol=atol, rtol=0.0)


def extract_numbers(series):
    out = [[int(part) for part in s.split('_') if part] for s in series]
    return out

def get_dim(ix):
    max_values = [max(column) + 1 for column in zip(*ix)]
    return max_values

def get_ps_layer(df, layer_name, ps_name):
    i_layer = df.loc[:, "parameterId"].str.startswith("net_" + layer_name + "_" + ps_name)
    df_layer = df.loc[i_layer, :]
    df_layer.reset_index(drop=True, inplace=True)
    ix = df_layer.loc[:, "parameterId"]
    ix = extract_numbers(ix.str.replace("net_" + layer_name + "_" + ps_name, ""))
    dims = get_dim(ix)
    out = torch.ones(*dims)
    for i in range(df_layer.shape[0]):
        _ix = ix[i]
        out[*_ix] = df_layer.loc[i, "value"]

    return out

def read_array(df):
    ix = df.loc[:, "ix"].astype("string")
    ix = ix.apply(lambda x: [int(i) for i in x.split(';')]).tolist()
    dims = get_dim(ix)
    out = torch.ones(*dims)
    for i in range(df.shape[0]):
        _ix = ix[i]
        out[*_ix] = df.loc[i, "value"]
    return out
