from __future__ import annotations
from typing import Any

import ast
import inspect
import re

import torch.fx
import torch.nn as nn
from pydantic import BaseModel, Field

from mkstd import YamlStandard


__all__ = ["Input", "Layer", "Node", "MLModel", "MLModels", "Output", "PetabScimlStandard"]


class Input(BaseModel):
    """Specify (transformations of) the input layer."""

    input_id: str
    transform: dict | None = Field(
        default=None
    )  # TODO class of supported transforms
    shape: tuple[int]


class Output(BaseModel):
    """Specify the output layer."""

    output_id: str
    shape: tuple[int]


class Layer(BaseModel):
    """Specify layers."""

    layer_id: str
    layer_type: str
    # FIXME currently handled as kwargs
    args: dict | None = Field(
        default=None
    )  # TODO class of layer-specific supported args


class Node(BaseModel):
    """A node of the computational graph.

    e.g. a node in the forward call of a PyTorch model.
    Ref: https://pytorch.org/docs/stable/fx.html#torch.fx.Node
    """

    name: str
    op: str
    target: str
    args: list | None = Field(default=None)
    kwargs: dict | None = Field(default=None)


# Some customization needs to be manually extracted. These are things
# that might only be present in `Module.__repr__` output conditionally,
# usually because pytorch avoids outputting default values in its `__repr__`.
# We will explicitly save them so tools don't need to find out what pytorch
# defaults are.
extra_repr = {
    "Conv2d": {
        # Missing from `Conv2d.__init__`: `output_padding`
        # "output_padding": lambda m: m.output_padding,
        "__all__": ["padding", "dilation", "groups", "bias", "padding_mode"],
        "getters": {
            "bias": lambda m: m.bias is not None,
        },
    },
    "RNN": {
        "__all__": [
            "input_size",
            "hidden_size",
            "proj_size",
            "num_layers",
            "bias",
            "batch_first",
            "dropout",
            "bidirectional",
        ],
        "getters": {},
    },
}

extra_repr = {
    module_id: {
        attr: module_def["getters"].get(
            attr, lambda m, attr=attr: getattr(m, attr)
        )
        for attr in module_def["__all__"]
    }
    for module_id, module_def in extra_repr.items()
}


def extract_module_args(module: nn.Module) -> dict:
    """Get the arguments used to create the module.

    N.B.: currently, all arguments must be Python literals compatible
    with `ast.literal_eval`, and cannot contain the character `=`.

    Args:
        module:
            The model.

    Returns:
        The arguments, as keyword arguments.
    """
    # Stage 1: get all arguments in the intersection of `__constants__` and
    # `__init__`
    init_arg_names = set(inspect.signature(module.__init__).parameters)
    constant_init_args = {
        arg: getattr(module, arg)
        for arg in module.__constants__
        if arg in init_arg_names
    }

    # Stage 2: add all arguments suggested in the `__repr__
    ## Get names of module positional arguments
    arg_names = []
    for arg in inspect.signature(module.__init__).parameters.values():
        if arg.name == "self":
            continue
        if arg.default != inspect.Parameter.empty:
            continue
        if arg.kind not in [
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ]:
            continue
        arg_names.append(arg.name)

    ## Extract all arguments
    class_name, all_args_str = re.match(
        r"(\w+)\((.*)\)", repr(module).strip()
    ).groups()

    ## All positional arguments exist
    args = {}
    args_str_list = []
    for arg_str in all_args_str.split(","):
        if "=" in arg_str:
            break
        args_str_list.append(arg_str)
    if args_str_list:
        args = dict(
            zip(arg_names, ast.literal_eval(",".join(args_str_list) + ","))
        )

    kwargs = {
        kw: ast.literal_eval(arg_str.strip())
        for kw, arg_str in re.findall(
            r"(\w*)=([^=]*)(?=,\s*\w+=|$)", all_args_str
        )
    }

    # See `extra_repr`
    extra = {
        kw: getter(module)
        for kw, getter in extra_repr.get(
            get_module_layer_type(module), {}
        ).items()
    }

    return constant_init_args | extra | args | kwargs


def get_module_layer_type(module: nn.Module) -> str:
    """Get the layer type of a module.

    Args:
        module:
            The module.

    Returns:
        The module type.
    """
    return type(module).__name__


def node_to_str(args: list[Node], pytorch_nodes: list[Node]) -> list[str]:
    """Convert nodes in layer/method inputs to node names.

    Args:
        args:
            The nodes.
        pytorch_nodes:
            All PyTorch nodes, from an FX symbolic trace.

    Returns:
        The converted nodes.
    """
    return [str(arg) if arg in pytorch_nodes else arg for arg in args]


def str_to_node(all_args: tuple[Any], state: dict[str, Node], index: int | None = None) -> tuple[Any]:
    """Convert node names in layer inputs to nodes.

    For example, ``torch.stack`` stacks all tensors in its first argument.
    The first argument is therefore a nested iterable. In this iterable, each tensor
    will be some node in the preceding computational graph. On disk, these nodes are
    their names. This substitutes the names for their nodes.

    Args:
        all_args:
            All layer/method inputs/arguments, if ``index`` is specified. Otherwise,
            the nodes.
        state:
            The nodes in the computational graph preceding these args.
            Keys are node names, values are the nodes.
        index:
            The index of ``all_args`` that will be converted.

    Returns:
        The arguments, with conversions only at ``index`` if specified.
    """
    if index is None:
        return tuple([state[arg] if arg in state else arg for arg in all_args])
    return tuple(
        list(all_args[:index])
        + [tuple([state[arg] if arg in state else arg for arg in all_args[index]])]
        + list(all_args[index+1:])
    )


class MLModel(BaseModel):
    """An easy-to-use format to specify simple deep ML models.

    There is a function to export this to a PyTorch module, or to YAML.
    """

    mlmodel_id: str

    inputs: list[Input]
    outputs: list[Output]

    layers: list[Layer]
    """The components of the model (e.g., layers of a neural network)."""

    forward: list[Node]

    @staticmethod
    def from_pytorch_module(
        module: nn.Module, mlmodel_id: str, inputs: list[Input], outputs: list[Output],
    ) -> MLModel:
        """Create a PEtab SciML ML model from a pytorch module."""
        layers = []
        layer_ids = []
        for layer_id, layer_module in module.named_modules():
            if not layer_id:
                # first entry is all modules combined
                continue
            layer = Layer(
                layer_id=layer_id,
                layer_type=get_module_layer_type(layer_module),
                args=extract_module_args(module=layer_module),
            )
            layers.append(layer)
            layer_ids.append(layer_id)

        nodes = []
        pytorch_nodes = list(torch.fx.symbolic_trace(module).graph.nodes)
        for pytorch_node in pytorch_nodes:
            op = pytorch_node.op
            target = pytorch_node.target
            args = node_to_str(args=pytorch_node.args, pytorch_nodes=pytorch_nodes)
            if op == "call_function":
                target = pytorch_node.target.__name__
                if target == "stack":
                    args[0] = node_to_str(args=args[0], pytorch_nodes=pytorch_nodes)
            if op == "output" and isinstance(args[0], tuple):
                # handle multiple outputs
                args[0] = node_to_str(args=args[0], pytorch_nodes=pytorch_nodes)

            node = Node(
                name=pytorch_node.name,
                op=op,
                target=target,
                args=args,
                kwargs=pytorch_node.kwargs,
            )
            nodes.append(node)

        mlmodel = MLModel(
            mlmodel_id=mlmodel_id, inputs=inputs, outputs=outputs, layers=layers, forward=nodes
        )
        return mlmodel

    def to_pytorch_module(self) -> nn.Module:
        """Create a pytorch module from a PEtab SciML ML model."""
        self2 = self

        class _PytorchModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                for layer in self2.layers:
                    setattr(
                        self,
                        layer.layer_id,
                        getattr(nn, layer.layer_type)(**layer.args),
                    )

        graph = torch.fx.Graph()
        state = {}
        for node in self.forward:
            args = []
            if node.args:
                for node_arg in node.args:
                    arg = node_arg
                    try:
                        if arg in state:
                            arg = state[arg]
                    except TypeError:
                        pass
                    args.append(arg)
            args = tuple(args)
            kwargs = {}
            if node.kwargs:
                kwargs = {k: state.get(v, v) for k, v in node.kwargs.items()}
            match node.op:
                case "placeholder":
                    state[node.name] = graph.placeholder(node.target)
                case "call_function":
                    call_function_module = nn.functional
                    if node.target in ["flatten", "stack"]:
                        call_function_module = torch
                    function = getattr(call_function_module, node.target)
                    if node.target in ["stack"]:
                        args = str_to_node(all_args=args, state=state, index=0)
                    state[node.name] = graph.call_function(
                        function, args, kwargs
                    )
                case "call_method":
                    state[node.name] = graph.call_method(
                        node.target, args, kwargs
                    )
                case "call_module":
                    state[node.name] = graph.call_module(
                        node.target, args, kwargs
                    )
                case "output":
                    if isinstance(args[0], list):
                        args = str_to_node(all_args=args[0], state=state)
                    else:
                        args = args[0]
                    graph.output(args)
                case _:
                    raise ValueError(f"Unhandled op: {node.op}")

        return torch.fx.GraphModule(_PytorchModule(), graph)


class MLModels(BaseModel):
    """Specify all ML models of your hybrid model."""

    models: list[MLModel]


PetabScimlStandard = YamlStandard(model=MLModels)

if __name__ == "__main__":
    PetabScimlStandard.save_schema("standard/schema.yaml")
