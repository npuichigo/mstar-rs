"""Model registry. Imports are lazy — pi05 pulls in the full mstar package
for its nn modules, vjepa2 pulls transformers."""


def __getattr__(name: str):
    if name == "VJEPA2":
        from .vjepa2 import VJEPA2

        return VJEPA2
    if name in ("PI05", "Pi05Policy", "Pi05Engine"):
        from . import pi05

        return getattr(pi05, name)
    if name in ("Orpheus", "OrpheusPolicy", "OrpheusEngine"):
        from . import orpheus

        return getattr(orpheus, name)
    if name in ("Qwen3OmniPolicy", "Qwen3OmniThinkerPolicy",
                "Qwen3OmniThinkerEngine", "Qwen3OmniAudioEngine",
                "Qwen3OmniAudioPolicy"):
        from . import qwen3_omni

        return getattr(qwen3_omni, name)
    raise AttributeError(name)


__all__ = [
    "VJEPA2", "PI05", "Pi05Policy", "Pi05Engine",
    "Orpheus", "OrpheusPolicy", "OrpheusEngine",
    "Qwen3OmniPolicy", "Qwen3OmniThinkerPolicy", "Qwen3OmniThinkerEngine",
    "Qwen3OmniAudioEngine", "Qwen3OmniAudioPolicy",
]
