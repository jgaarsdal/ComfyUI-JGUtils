import torch
import torchaudio
from transformers import pipeline as hf_pipeline
import comfy.model_management

CORAL_ASR_MODEL_TYPE = "CORAL_ASR_MODEL"

_ASR_SAMPLE_RATE = 16000

MODEL_OPTIONS = [
    "roest-v3-whisper-1.5b",
    "roest-v3-wav2vec2-315m",
]


def _prepare_audio(audio):
    """Convert a ComfyUI AUDIO dict to a mono float32 numpy array at 16 kHz.

    ComfyUI audio is typically 44.1 kHz but ASR models expect 16 kHz.
    The HuggingFace ASR pipeline has a resampling bug (passes orig_freq
    twice to torchaudio.functional.resample), so we resample here.
    """
    waveform = audio["waveform"]  # (batch, channels, samples)
    sample_rate = audio["sample_rate"]
    # Remove batch dim, average channels to mono
    wav = waveform.squeeze(0).mean(dim=0, keepdim=True).float()
    # Resample to 16 kHz if needed
    if sample_rate != _ASR_SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sample_rate, _ASR_SAMPLE_RATE)
    wav = wav.squeeze(0).cpu().numpy()
    return wav, _ASR_SAMPLE_RATE


def _transcribe(model, audio, prompt=""):
    """Run ASR on a single ComfyUI AUDIO dict using a loaded CoRal model."""
    pipe = model["pipeline"]
    is_whisper = model["is_whisper"]

    wav, sample_rate = _prepare_audio(audio)

    kwargs = {}
    if is_whisper and prompt:
        prompt_ids = pipe.tokenizer.get_prompt_ids(prompt, return_tensors="pt")
        prompt_ids = prompt_ids.to(pipe.model.device)
        kwargs["generate_kwargs"] = {"prompt_ids": prompt_ids}

    result = pipe({"raw": wav, "sampling_rate": sample_rate}, **kwargs)
    return result["text"]


class CoRalASRModelLoader:
    """Loads a CoRal Danish ASR model from HuggingFace."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (MODEL_OPTIONS,),
                "device": (["auto", "cpu", "cuda"],),
            },
        }

    RETURN_TYPES = (CORAL_ASR_MODEL_TYPE,)
    RETURN_NAMES = ("model",)
    OUTPUT_TOOLTIPS = ("Loaded CoRal ASR pipeline",)
    FUNCTION = "load_model"
    CATEGORY = "JG Utils/ASR"
    DESCRIPTION = (
        "Loads a CoRal-project Danish ASR model from HuggingFace. "
        "Supports Whisper (1.5B) and Wav2Vec2 (315M) architectures."
    )

    def load_model(self, model_name, device):
        if device == "auto":
            dev = comfy.model_management.get_torch_device()
        else:
            dev = torch.device(device)

        # Pipeline handles device placement most reliably with int (CUDA
        # index) or string "cpu", rather than a torch.device object.
        if dev.type == "cuda":
            pipe_device = dev.index if dev.index is not None else 0
        else:
            pipe_device = dev.type

        model_id = f"CoRal-project/{model_name}"
        is_whisper = "whisper" in model_name

        pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=pipe_device,
        )

        return (
            {
                "pipeline": pipe,
                "is_whisper": is_whisper,
            },
        )


class CoRalTranscribe:
    """Transcribes a single audio clip using a CoRal Danish ASR model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "model": (CORAL_ASR_MODEL_TYPE,),
            },
            "optional": {
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional text prompt to guide Whisper transcription. Ignored for Wav2Vec2 models.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_TOOLTIPS = ("Transcribed Danish text",)
    FUNCTION = "transcribe"
    CATEGORY = "JG Utils/ASR"
    DESCRIPTION = "Transcribes a single audio clip to Danish text using a CoRal ASR model."

    def transcribe(self, audio, model, prompt=""):
        text = _transcribe(model, audio, prompt=prompt)
        return (text,)


class CoRalTranscribeBatch:
    """Batch transcribes a list of audio clips using a CoRal Danish ASR model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "model": (CORAL_ASR_MODEL_TYPE,),
            },
            "optional": {
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional text prompt to guide Whisper transcription. Ignored for Wav2Vec2 models.",
                    },
                ),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("texts",)
    OUTPUT_IS_LIST = (True,)
    OUTPUT_TOOLTIPS = ("List of transcribed Danish texts",)
    FUNCTION = "transcribe"
    CATEGORY = "JG Utils/ASR"
    DESCRIPTION = (
        "Batch transcribes a list of audio clips to Danish text using a CoRal ASR model. "
        "Pairs well with the Audio Segment node."
    )

    def transcribe(self, audio, model, prompt=None):
        mdl = model[0]  # INPUT_IS_LIST makes all inputs lists; model is the same for all
        p = prompt[0] if prompt is not None else ""
        texts = [_transcribe(mdl, a, prompt=p) for a in audio]
        return (texts,)


NODE_CLASS_MAPPINGS = {
    "CoRalASRModelLoader": CoRalASRModelLoader,
    "CoRalTranscribe": CoRalTranscribe,
    "CoRalTranscribeBatch": CoRalTranscribeBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CoRalASRModelLoader": "CoRal ASR Model Loader",
    "CoRalTranscribe": "CoRal Transcribe",
    "CoRalTranscribeBatch": "CoRal Transcribe (Batch)",
}
