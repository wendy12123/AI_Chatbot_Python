import importlib
import pkgutil


__registry__ = {}
def register(name):
    def decorator(func):
        __registry__[name] = func
        return func
    return decorator
# Example usage:
#@register("example_function")
#def example_function():
#    print("This is an example function.")

def require(name):
    if name not in __registry__:
        raise ValueError(f"RuntimeFlowPlugins Error: Function '{name}' is not registered.")
    return __registry__[name]

def list_registered():
    return list(__registry__.keys())


def _autoload_plugins():
    # Import every module in this package so @register decorators run.
    package_name = __name__
    for module_info in pkgutil.iter_modules(__path__):
        module_name = module_info.name
        if module_name.startswith("_"):
            continue
        importlib.import_module(f"{package_name}.{module_name}")


_autoload_plugins()

