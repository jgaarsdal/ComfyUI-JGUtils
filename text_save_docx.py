import glob
import os
import re

from docx import Document


class SaveTextDocx:
    """Saves text to a .docx file.

    Accepts a single string or a list of strings.  When multiple strings
    are provided each one becomes a separate paragraph in the document.
    Files are saved with an auto-incrementing counter so that existing
    files are never overwritten.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "folder_path": ("STRING", {"default": ""}),
                "filename_prefix": ("STRING", {"default": "document"}),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    OUTPUT_TOOLTIPS = ("Absolute path of the saved .docx file",)
    FUNCTION = "save_docx"
    OUTPUT_NODE = True
    CATEGORY = "JG Utils/Text"
    DESCRIPTION = (
        "Saves text to a .docx file. Each input string becomes a paragraph. "
        "Files are named with an auto-incrementing counter to avoid overwrites."
    )

    def save_docx(self, text, folder_path, filename_prefix):
        # INPUT_IS_LIST makes every input a list; unwrap the scalars.
        folder = folder_path[0]
        prefix = filename_prefix[0]

        if not folder:
            raise ValueError("folder_path must not be empty.")

        os.makedirs(folder, exist_ok=True)

        # Find the next available counter by scanning existing files.
        pattern = os.path.join(folder, f"{glob.escape(prefix)}_[0-9][0-9][0-9][0-9][0-9].docx")
        existing = glob.glob(pattern)
        counter = 0
        if existing:
            numbers = []
            regex = re.compile(re.escape(prefix) + r"_(\d{5})\.docx$")
            for path in existing:
                m = regex.search(os.path.basename(path))
                if m:
                    numbers.append(int(m.group(1)))
            if numbers:
                counter = max(numbers) + 1

        filename = f"{prefix}_{counter:05d}.docx"
        file_path = os.path.join(folder, filename)

        doc = Document()
        for paragraph in text:
            doc.add_paragraph(paragraph)
        doc.save(file_path)

        return (file_path,)


NODE_CLASS_MAPPINGS = {
    "SaveTextDocx": SaveTextDocx,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveTextDocx": "Save Text to DOCX",
}
