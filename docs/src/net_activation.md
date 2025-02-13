# [Supported Layers and Activation Functions](@id layers_activation)

The PEtab SciML neural network model YAML format supports numerous standard neural network layers and activation functions. Layer names and associated keyword arguments follow the PyTorch naming scheme. PyTorch is used because it is currently the most popular machine learning framework, and its comprehensive documentation makes it easy to look up details for any specific layer or activation function.

If support is lacking for a layer or activation function you would like to see, please file an issue on [GitHub](https://github.com/sebapersson/petab_sciml/issues).

## Supported Neural Network Layers

The table below lists the supported and tested neural network layers along with links to their respective PyTorch documentation. Additionally, the table indicates which tools support each layer.

| layer                                                                                                                     | PEtab.jl | AMICI |
|---------------------------------------------------------------------------------------------------------------------------|:--------:|-------|
| [Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear)                                  | x        |       |
| [Bilinear](https://pytorch.org/docs/stable/generated/torch.nn.Bilinear.html#torch.nn.Bilinear)                            | x        |       |
| [Flatten](https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html#torch.nn.Flatten)                               | x        |       |
| [Dropout](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html#torch.nn.Dropout)                               | x        |       |
| [Dropout1d](https://pytorch.org/docs/stable/generated/torch.nn.Dropout1d.html#torch.nn.Dropout1d)                         | x        |       |
| [Dropout2d](https://pytorch.org/docs/stable/generated/torch.nn.Dropout2d.html#torch.nn.Dropout2d)                         | x        |       |
| [Dropout3d](https://pytorch.org/docs/stable/generated/torch.nn.Dropout3d.html#torch.nn.Dropout3d)                         | x        |       |
| [AlphaDropout](https://pytorch.org/docs/stable/generated/torch.nn.AlphaDropout.html#torch.nn.AlphaDropout)                | x        |       |
| [Conv1d](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html#torch.nn.Conv1d)                                  | x        |       |
| [Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d)                                  | x        |       |
| [Conv3d](https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html#torch.nn.Conv3d)                                  | x        |       |
| [ConvTranspose1d](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html#torch.nn.ConvTranspose1d)       | x        |       |
| [ConvTranspose2d](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html#torch.nn.ConvTranspose2d)       | x        |       |
| [ConvTranspose3d](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose3d.html#torch.nn.ConvTranspose3d)       | x        |       |
| [MaxPool1d](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html#torch.nn.MaxPool1d)                         | x        |       |
| [MaxPool2d](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d)                         | x        |       |
| [MaxPool3d](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool3d.html#torch.nn.MaxPool3d)                         | x        |       |
| [AvgPool1d](https://pytorch.org/docs/stable/generated/torch.nn.AvgPool1d.html#torch.nn.AvgPool1d)                         | x        |       |
| [AvgPool2d](https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html#torch.nn.AvgPool2d)                         | x        |       |
| [AvgPool3d](https://pytorch.org/docs/stable/generated/torch.nn.AvgPool3d.html#torch.nn.AvgPool3d)                         | x        |       |
| [LPPool1](https://pytorch.org/docs/stable/generated/torch.nn.LPPool1d.html#torch.nn.LPPool1d)                             | x        |       |
| [LPPool2](https://pytorch.org/docs/stable/generated/torch.nn.LPPool2d.html#torch.nn.LPPool2d)                             | x        |       |
| [LPPool3](https://pytorch.org/docs/stable/generated/torch.nn.LPPool3d.html#torch.nn.LPPool3d)                             | x        |       |
| [AdaptiveMaxPool1d](https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveMaxPool1d.html#torch.nn.AdaptiveMaxPool1d) | x        |       |
| [AdaptiveMaxPool2d](https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveMaxPool2d.html#torch.nn.AdaptiveMaxPool2d) | x        |       |
| [AdaptiveMaxPool3d](https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveMaxPool3d.html#torch.nn.AdaptiveMaxPool3d) | x        |       |
| [AdaptiveAvgPool1d](https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool1d.html#torch.nn.AdaptiveAvgPool1d) | x        |       |
| [AdaptiveAvgPool2d](https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html#torch.nn.AdaptiveAvgPool2d) | x        |       |
| [AdaptiveAvgPool3d](https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool3d.html#torch.nn.AdaptiveAvgPool3d) | x        |       |

## Supported Activation Function

The table below lists the supported and tested activation functions along with links to their respective PyTorch documentation. Additionally, the table indicates which tools support each layer.

| Function                                                                                                                      | PEtab.jl | AMICI |
|-------------------------------------------------------------------------------------------------------------------------------|:--------:|-------|
| [relu](https://pytorch.org/docs/stable/generated/torch.nn.functional.relu.html#torch.nn.functional.relu)                      |     x    |       |
| [relu6](https://pytorch.org/docs/stable/generated/torch.nn.functional.relu6.html#torch.nn.functional.relu6)                   |     x    |       |
| [hardtanh](https://pytorch.org/docs/stable/generated/torch.nn.functional.hardtanh.html#torch.nn.functional.hardtanh)          |     x    |       |
| [hardswish](https://pytorch.org/docs/stable/generated/torch.nn.functional.hardswish.html#torch.nn.functional.hardswish)       |     x    |       |
| [selu](https://pytorch.org/docs/stable/generated/torch.nn.functional.selu.html#torch.nn.functional.selu)                      |     x    |       |
| [leaky_relu](https://pytorch.org/docs/stable/generated/torch.nn.functional.leaky_relu.html#torch.nn.functional.leaky_relu)    |     x    |       |
| [gelu](https://pytorch.org/docs/stable/generated/torch.nn.functional.gelu.html#torch.nn.functional.gelu)                      |     x    |       |
| [tanhshrink](https://pytorch.org/docs/stable/generated/torch.nn.functional.tanhshrink.html#torch.nn.functional.tanhshrink)    |     x    |       |
| [softsign](https://pytorch.org/docs/stable/generated/torch.nn.functional.softsign.html#torch.nn.functional.softsign)          |     x    |       |
| [softplus](https://pytorch.org/docs/stable/generated/torch.nn.functional.softplus.html#torch.nn.functional.softplus)          |     x    |       |
| [tanh](https://pytorch.org/docs/stable/generated/torch.nn.functional.tanh.html#torch.nn.functional.tanh)                      |     x    |       |
| [sigmoid](https://pytorch.org/docs/stable/generated/torch.nn.functional.sigmoid.html#torch.nn.functional.sigmoid)             |     x    |       |
| [hardsigmoid](https://pytorch.org/docs/stable/generated/torch.nn.functional.hardsigmoid.html#torch.nn.functional.hardsigmoid) |     x    |       |
| [mish](https://pytorch.org/docs/stable/generated/torch.nn.functional.mish.html#torch.nn.functional.mish)                      |     x    |       |
| [elu](https://pytorch.org/docs/stable/generated/torch.nn.functional.elu.html#torch.nn.functional.elu)                         |     x    |       |
| [celu](https://pytorch.org/docs/stable/generated/torch.nn.functional.celu.html#torch.nn.functional.celu)                      |     x    |       |
| [softmax](https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html#torch.nn.functional.softmax)             |     x    |       |
| [log_softmax](https://pytorch.org/docs/stable/generated/torch.nn.functional.log_softmax.html#torch.nn.functional.log_softmax) |     x    |       |
