class Registry:
    def __init__(self):
        self._entrypoints = {}

    def register(self, name):
        def register_cls(cls):
            self._entrypoints[name] = cls
            return cls

        return register_cls

    def entrypoints(self, name):
        return self._entrypoints[name]

    def is_available(self, name):
        return name in self._entrypoints


register_models = Registry()
register_optimizers = Registry()
register_losses = Registry()
