# Classification
This is our humble approach to add all the classification models in one repo

## Train a model
```
python tools/train.py --cfg <path to config file>
```

## Test a model
```
python tools/test.py --cfg <path to config file>
```

## List of models added:
- VGG
- ResNet

## Adding a model
- create a new python file in lib/core/models
- Code your model
- add from lib.utils.registry import register_models
- decorate using register_models.register("your model")
