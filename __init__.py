"""ComfyUI-Krea2-WeightedConditioning.

The folder name is not a valid Python identifier (hyphens), so ComfyUI loads
this file by path. Pytest may import it the same way; fall back to an absolute
import so unit tests still collect.
"""

try:
    from .krea2_weighted.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:
    from krea2_weighted.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
