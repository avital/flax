# Train only a Few Layers (HOWTO)

Goal: Create a model from which only the last (few) layers are trained.

Implementation: This uses a base model (CNN), which outputs both the logits and
a representation_layer. This is used by a refinement model (RefineCNN). Only the
latter is trained (ie used in the optimizer). The following lines are
(hopefully) illustrative:
```
  base_logits, base_representation = base_model(batch['image'])
  logits = refine_model(base_representation)
  ...
  optimizer = create_optimizer(refine_model, … )
```

@insert_branch_diff howto-train-subset-layers