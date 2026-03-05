import os

import torch
import comfy.model_management
from pyannote.audio import Pipeline as PyannotePipeline

DIARIZATION_MODEL_TYPE = "DIARIZATION_MODEL"

_DEFAULT_MODEL_ID = "pyannote/speaker-diarization-community-1"


class SpeakerDiarizationModelLoader:
    """Loads a pyannote speaker diarization pipeline from HuggingFace."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (["auto", "cpu", "cuda"],),
            },
            "optional": {
                "hf_token": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "HuggingFace auth token for the gated pyannote model. "
                            "If empty, falls back to the HF_TOKEN environment variable. "
                            "You must accept the model license at "
                            "https://hf.co/pyannote/speaker-diarization-community-1"
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = (DIARIZATION_MODEL_TYPE,)
    RETURN_NAMES = ("model",)
    OUTPUT_TOOLTIPS = ("Loaded speaker diarization pipeline",)
    FUNCTION = "load_model"
    CATEGORY = "JG Utils/ASR"
    DESCRIPTION = (
        "Loads a pyannote speaker diarization pipeline from HuggingFace. "
        "Requires a HuggingFace auth token and accepting the model license."
    )

    def load_model(self, device, hf_token=""):
        token = hf_token.strip() if hf_token else ""
        if not token:
            token = os.environ.get("HF_TOKEN", "")
        if not token:
            raise ValueError(
                "No HuggingFace token provided. Set the hf_token input or "
                "the HF_TOKEN environment variable. You must also accept the "
                "model license at https://hf.co/pyannote/speaker-diarization-community-1"
            )

        if device == "auto":
            dev = comfy.model_management.get_torch_device()
        else:
            dev = torch.device(device)

        pipeline = PyannotePipeline.from_pretrained(
            _DEFAULT_MODEL_ID,
            token=token,
        )
        pipeline.to(dev)

        return ({"pipeline": pipeline},)


class SpeakerDiarize:
    """Diarizes audio into per-speaker-turn segments."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "model": (DIARIZATION_MODEL_TYPE,),
            },
            "optional": {
                "num_speakers": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 100,
                        "step": 1,
                        "tooltip": "Exact number of speakers. 0 = auto-detect.",
                    },
                ),
                "min_speakers": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "tooltip": "Minimum number of speakers (ignored when num_speakers > 0)",
                    },
                ),
                "max_speakers": (
                    "INT",
                    {
                        "default": 10,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "tooltip": "Maximum number of speakers (ignored when num_speakers > 0)",
                    },
                ),
                "max_segment_length": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 3600.0,
                        "step": 0.1,
                        "tooltip": (
                            "Maximum segment duration in seconds. Speaker turns longer "
                            "than this are split into consecutive sub-segments with the "
                            "same speaker label. 0 = no splitting."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING", "INT")
    RETURN_NAMES = ("segments", "speakers", "length")
    OUTPUT_IS_LIST = (True, True, False)
    OUTPUT_TOOLTIPS = (
        "Audio segments, one per speaker turn in chronological order",
        "Speaker label for each segment (e.g. SPEAKER_00)",
        "Number of segments",
    )
    FUNCTION = "diarize"
    CATEGORY = "JG Utils/ASR"
    DESCRIPTION = (
        "Splits audio into per-speaker-turn segments using pyannote speaker "
        "diarization. Each output segment contains one speaker's turn. "
        "Pairs well with CoRal Transcribe (Batch) for speaker-attributed transcription."
    )

    def diarize(self, audio, model, num_speakers=0, min_speakers=1,
                max_speakers=10, max_segment_length=0.0):
        pipeline = model["pipeline"]
        waveform = audio["waveform"]  # (batch, channels, samples)
        sample_rate = audio["sample_rate"]

        # pyannote expects (channels, samples) — squeeze the batch dim
        wav = waveform.squeeze(0)

        # Build kwargs for the pipeline call
        kwargs = {}
        if num_speakers > 0:
            kwargs["num_speakers"] = num_speakers
        else:
            kwargs["min_speakers"] = min_speakers
            kwargs["max_speakers"] = max_speakers

        # Run diarization
        output = pipeline(
            {"waveform": wav, "sample_rate": sample_rate},
            **kwargs,
        )

        # Extract speaker turns from the annotation
        diarization = output.speaker_diarization
        max_samples = (round(max_segment_length * sample_rate)
                       if max_segment_length > 0 else 0)

        segments = []
        speakers = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            start_sample = round(segment.start * sample_rate)
            end_sample = round(segment.end * sample_rate)
            # Clamp to waveform bounds
            start_sample = max(0, start_sample)
            end_sample = min(waveform.shape[-1], end_sample)
            if end_sample <= start_sample:
                continue

            # Split long segments into consecutive sub-segments
            if max_samples > 0 and (end_sample - start_sample) > max_samples:
                pos = start_sample
                while pos < end_sample:
                    sub_end = min(pos + max_samples, end_sample)
                    chunk = waveform[..., pos:sub_end].clone()
                    segments.append({"waveform": chunk, "sample_rate": sample_rate})
                    speakers.append(speaker)
                    pos = sub_end
            else:
                chunk = waveform[..., start_sample:end_sample].clone()
                segments.append({"waveform": chunk, "sample_rate": sample_rate})
                speakers.append(speaker)

        # Handle edge case: no speech detected
        if not segments:
            segments.append(audio)
            speakers.append("SPEAKER_00")

        return (segments, speakers, len(segments))


NODE_CLASS_MAPPINGS = {
    "SpeakerDiarizationModelLoader": SpeakerDiarizationModelLoader,
    "SpeakerDiarize": SpeakerDiarize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SpeakerDiarizationModelLoader": "Speaker Diarization Model Loader",
    "SpeakerDiarize": "Speaker Diarize",
}
