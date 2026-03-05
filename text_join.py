class JoinText:
    """Joins a list of text strings into a single string with a delimiter."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "delimiter": ("STRING", {"default": " "}),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_TOOLTIPS = ("Joined text",)
    FUNCTION = "join_text"
    CATEGORY = "JG Utils/Text"
    DESCRIPTION = (
        "Joins a list of text strings into a single string using a delimiter. "
        "Pairs well with CoRal Transcribe (Batch) for combining transcription segments."
    )

    def join_text(self, text, delimiter=None):
        if delimiter is None:
            sep = " "
        else:
            sep = delimiter[0]  # INPUT_IS_LIST makes all inputs lists
        return (sep.join(text),)


NODE_CLASS_MAPPINGS = {
    "JoinText": JoinText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JoinText": "Join Text",
}
