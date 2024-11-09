import torch

def extract_numbers(series):
    out = [[int(part) for part in s.split('_') if part] for s in series]
    return out

def get_dim(ix):
    max_values = [max(column) for column in zip(*ix)]
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
        _ix = [x - 1 for x in ix[i]]
        out[*_ix] = df_layer.loc[i, "value"]

    return out

def read_array(df):
    ix = df.loc[:, "ix"].astype("string")
    ix = ix.apply(lambda x: [int(i) for i in x.split(';')]).tolist()
    dims = get_dim(ix)
    out = torch.ones(*dims)
    for i in range(df.shape[0]):
        _ix = [x - 1 for x in ix[i]]
        out[*_ix] = df.loc[i, "value"]
    return out
