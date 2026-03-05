import torch
from transformers import pipeline as hf_pipeline
import comfy.model_management

CORAL_ASR_MODEL_TYPE = "CORAL_ASR_MODEL"

MODEL_OPTIONS = [
    "roest-v3-whisper-1.5b",
    "roest-v3-wav2vec2-315m",
]


def _prepare_audio(audio):
    """Convert a ComfyUI AUDIO dict to a mono float32 numpy array + sample rate."""
    waveform = audio["waveform"]  # (batch, channels, samples)
    sample_rate = audio["sample_rate"]
    # Remove batch dim, average channels to mono, ensure float32 on CPU
    wav = waveform.squeeze(0).mean(dim=0).cpu().float().numpy()
    return wav, sample_rate


def _transcribe(model, audio, prompt=""):
    """Run ASR on a single ComfyUI AUDIO dict using a loaded CoRal model."""
    pipe = model["pipeline"]
    chunk_length_s = model["chunk_length_s"]
    is_whisper = model["is_whisper"]

    wav, sample_rate = _prepare_audio(audio)

    kwargs = {"chunk_length_s": chunk_length_s}
    if is_whisper:
        gen_kwargs = {"language": "da", "task": "transcribe"}
        if prompt:
            gen_kwargs["prompt_ids"] = pipe.tokenizer.get_prompt_ids(
                prompt, return_tensors="pt"
            )
        kwargs["generate_kwargs"] = gen_kwargs

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
                "chunk_length_s": (
                    "FLOAT",
                    {
                        "default": 30.0,
                        "min": 1.0,
                        "max": 300.0,
                        "step": 1.0,
                        "tooltip": "Audio chunk length in seconds for processing long audio",
                    },
                ),
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

    def load_model(self, model_name, device, chunk_length_s):
        if device == "auto":
            dev = comfy.model_management.get_torch_device()
        else:
            dev = torch.device(device)

        model_id = f"CoRal-project/{model_name}"
        is_whisper = "whisper" in model_name

        pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=dev,
        )

        return (
            {
                "pipeline": pipe,
                "chunk_length_s": chunk_length_s,
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
