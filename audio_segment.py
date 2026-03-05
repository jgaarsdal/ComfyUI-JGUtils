import math


class AudioSegment:
    """Segments audio into equal-length chunks with overlap if needed."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "segment_duration": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 0.1,
                        "max": 29.0,
                        "step": 0.1,
                        "tooltip": "Duration of each segment in seconds (max 29 s for Whisper compatibility)",
                    },
                ),
            },
        }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("segments", "length")
    OUTPUT_IS_LIST = (True, False)
    OUTPUT_TOOLTIPS = ("List of fixed-length audio segments", "Number of segments")
    FUNCTION = "segment"
    CATEGORY = "JG Utils/Audio"
    DESCRIPTION = (
        "Splits audio into fixed-length segments. If the audio duration is not "
        "evenly divisible by the segment duration, segments overlap equally so "
        "that every segment has exactly the requested length."
    )

    def segment(self, audio, segment_duration):
        waveform = audio["waveform"]  # (batch, channels, samples)
        sample_rate = audio["sample_rate"]  # int

        total_samples = waveform.shape[-1]
        segment_samples = round(segment_duration * sample_rate)

        # Clamp: if the audio is shorter than or equal to one segment, return as-is
        if total_samples <= segment_samples:
            return ([audio], 1)

        # How many segments do we need?
        num_segments = math.ceil(total_samples / segment_samples)

        # Calculate the stride (step) between segment start positions.
        # When total_samples is evenly divisible, stride == segment_samples (no overlap).
        # Otherwise the stride is smaller, producing equal overlap so every segment
        # is exactly the requested length.
        stride = (total_samples - segment_samples) / (num_segments - 1)

        segments = []
        for i in range(num_segments):
            start = round(i * stride)
            end = start + segment_samples
            # Safety clamp in case of rounding at the tail end
            end = min(end, total_samples)
            start = end - segment_samples

            # Clone so overlapping segments get independent memory
            chunk = waveform[..., start:end].clone()
            segments.append(
                {
                    "waveform": chunk,
                    "sample_rate": sample_rate,
                }
            )

        return (segments, num_segments)


NODE_CLASS_MAPPINGS = {
    "AudioSegment": AudioSegment,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioSegment": "Audio Segment",
}
