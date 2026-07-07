# Regression TTA

Research code for test-time adaptation for regression, centered on PSC.

The old implementation is archived under `bin/` and ignored by git. New code uses
Hydra with a single entrypoint:

```bash
python main.py method=psc data=utkface_corrupt model=resnet50
python main.py method=source data=svhn_mnist model=resnet26
```

Current layout:

- `data/`: source/target dataset-pair builders.
- `src/`: PSC and shared experiment runner code.
- `baselines/`: baseline methods.
- `utils/`: small shared utilities.
- `configs/`: Hydra configuration groups.

CWSA is intentionally not part of the refactored method set.
