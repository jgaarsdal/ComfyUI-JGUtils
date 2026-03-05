import torchaudio


class ResampleAudio:
    """Resamples audio to a target sample rate."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "sample_rate": (
                    "INT",
                    {
                        "default": 44100,
                        "min": 1,
                        "max": 192000,
                        "step": 1,
                        "tooltip": "Target sample rate in Hz (e.g. 16000 for ASR models, 44100 for CD quality)",
                    },
                ),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    OUTPUT_TOOLTIPS = ("Resampled audio",)
    FUNCTION = "resample"
    CATEGORY = "JG Utils/Audio"
    DESCRIPTION = (
        "Resamples audio to a target sample rate using torchaudio. "
        "Useful for preparing audio for ASR models that expect 16 kHz input."
    )

    def resample(self, audio, sample_rate):
        waveform = audio["waveform"]
        orig_rate = audio["sample_rate"]

        if orig_rate != sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_rate, sample_rate
            )

        return ({"waveform": waveform, "sample_rate": sample_rate},)


NODE_CLASS_MAPPINGS = {
    "ResampleAudio": ResampleAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ResampleAudio": "Resample Audio",
}
