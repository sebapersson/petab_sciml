In the first example (`example.py`) we:
1. define a data format standard for ML models and save the schema to `standard/schema.yaml`. See `petab_sciml_standard.py`.
2. create an ML model in pytorch
3. convert that pytorch model into a PEtab SciML ML model and store it to disk (see `data/models0.yaml`)
4. read the model from disk, reconstruct the pytorch model, then convert that reconstructed pytorch model back into PEtab SciML, and store it to disk once more (see `data/models1.yaml`)

In total, this means we do:
```
pytorch model
-> petab sciml model
-> petab sciml yaml
-> petab sciml model
-> pytorch model
-> petab sciml model
-> petab sciml yaml
```
and then verify that the two YAML files match.


# TODO
- [ ] check that the original pytorch forward call provides that same output as the reconstructed pytorch forward call, for some different inputs.
- [ ] the following will have language-specific quirks that are currently not specified by pytorch as some attribute
  - python and julia differ in their flatten commands (one is row-major, the other column-major). For consistency, we only support tensors up to dimension 5, and their order is explicitly Width Height Depth Channel Batch. Larger tensors could be supported, but not with ops like flatten?
  - TODO: get input dimensions from first layer's input dimensions, and then annotate them with W,H,D,C,N up to the number of dimensions they have
