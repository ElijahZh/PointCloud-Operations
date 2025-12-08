from addict import Dict
# import inspect

class Registry:
    def __init__(self, name):
        self._name = name
        self._registry = Dict()

    def register(self, register_name=None):
        if register_name:
            assert isinstance(register_name, str)

        def decorator(item):
            # nonlocal name
            name = register_name or item.__name__
            if name in self._registry:
                raise KeyError(f"item `{name}` already in the Registry(`{self._name}`) ")
            self._registry[name] = item
            return item

        return decorator

    def __len__(self):
        return len(self._registry)

    def __contains__(self, name):
        return self._registry[name] is not None

    def __getitem__(self, item):
        return self._registry[item]

    def get(self, name):
        return self._registry[name]

    def list_items(self):
        # tmp = cls.registries[category]
        return list(self._registry.keys())

    def __repr__(self):
        res = f"Registry Name: {self._name}\n"
        res += f"Contains {len(self.list_items())} items\n"
        res += f"{self.list_items()}"
        return res

METHODS = Registry("methods")

@METHODS.register(register_name="first method")
def func(a):
    return a

@METHODS.register(register_name="second method")
def func_b(b):
    return b

if __name__ == '__main__':
    print(METHODS.list_items())
    print(METHODS.get("method"))
