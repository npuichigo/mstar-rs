"""Model registry. Imports are lazy — pi05 pulls in the full mstar package
for its nn modules, vjepa2 pulls transformers."""


def __getattr__(name: str):
    if name == "VJEPA2":
        from .vjepa2 import VJEPA2

        return VJEPA2
    if name == "PI05":
        from .pi05 import PI05

        return PI05
    raise AttributeError(name)


__all__ = ["VJEPA2", "PI05"]
