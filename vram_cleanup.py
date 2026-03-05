import comfy.model_management
from comfy.comfy_types.node_typing import IO


class FreeVRAM:
    """Unloads all cached models and flushes the GPU memory cache.

    Insert this node between two others to force VRAM cleanup at a
    specific point in a workflow.  The input value is passed through
    unchanged so existing connections are not disrupted.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any_input": (IO.ANY, {}),
            },
        }

    RETURN_TYPES = (IO.ANY,)
    RETURN_NAMES = ("any_output",)
    OUTPUT_TOOLTIPS = ("Passthrough of the input value",)
    FUNCTION = "free_vram"
    CATEGORY = "JG Utils/Utils"
    DESCRIPTION = (
        "Unloads all cached models from VRAM and empties the GPU memory cache. "
        "Connect between nodes to force cleanup at a specific point in the workflow."
    )

    def free_vram(self, any_input):
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        return (any_input,)


NODE_CLASS_MAPPINGS = {
    "FreeVRAM": FreeVRAM,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FreeVRAM": "Free VRAM",
}
