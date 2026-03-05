# ComfyUI-JGUtils

A collection of custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) focused on audio processing and Danish speech recognition.

## Nodes

### Audio Segment

**Category:** `JG Utils/Audio`

Splits audio into fixed-length segments. When the total duration is not evenly divisible by the segment duration, segments overlap equally so that every segment has exactly the requested length.

| Input | Type | Description |
|---|---|---|
| audio | AUDIO | Input audio |
| segment_duration | FLOAT | Duration of each segment in seconds (default: 5.0) |

| Output | Type | Description |
|---|---|---|
| segments | AUDIO[] | List of fixed-length audio segments |
| length | INT | Number of segments |

---

### Resample Audio

**Category:** `JG Utils/Audio`

Resamples audio to a target sample rate using torchaudio. Useful for preparing audio for ASR models that expect 16 kHz input, or for converting between sample rates in general.

| Input | Type | Description |
|---|---|---|
| audio | AUDIO | Input audio |
| sample_rate | INT | Target sample rate in Hz (default: 44100) |

| Output | Type | Description |
|---|---|---|
| audio | AUDIO | Resampled audio |

---

### CoRal ASR Model Loader

**Category:** `JG Utils/ASR`

Loads a [CoRal-project](https://huggingface.co/CoRal-project) Danish ASR model from HuggingFace. Two model architectures are supported:

- **roest-v3-whisper-1.5b** -- Whisper-based, 1.5B parameters. Supports optional text prompts.
- **roest-v3-wav2vec2-315m** -- Wav2Vec2-based, 315M parameters. Faster and lighter.

| Input | Type | Description |
|---|---|---|
| model_name | COMBO | Model to load |
| device | COMBO | Device to run on (auto, cpu, cuda) |

| Output | Type | Description |
|---|---|---|
| model | CORAL_ASR_MODEL | Loaded ASR pipeline |

> **Note:** The Whisper model has a 30-second context window. For longer audio, use the **Audio Segment** node to split the audio into segments (up to 30 s each) before transcribing with **CoRal Transcribe (Batch)**. Audio is automatically resampled to 16 kHz as required by the ASR models. The Wav2Vec2 model handles variable-length audio natively but may still benefit from segmentation for very long files.

---

### CoRal Transcribe

**Category:** `JG Utils/ASR`

Transcribes a single audio clip to Danish text using a loaded CoRal ASR model.

| Input | Type | Description |
|---|---|---|
| audio | AUDIO | Audio to transcribe |
| model | CORAL_ASR_MODEL | Loaded CoRal ASR model |
| prompt | STRING (optional) | Text prompt to guide Whisper transcription. Ignored for Wav2Vec2. |

| Output | Type | Description |
|---|---|---|
| text | STRING | Transcribed Danish text |

---

### CoRal Transcribe (Batch)

**Category:** `JG Utils/ASR`

Batch transcribes a list of audio clips to Danish text. Pairs well with the **Audio Segment** node for transcribing long audio files split into chunks.

| Input | Type | Description |
|---|---|---|
| audio | AUDIO[] | List of audio clips to transcribe |
| model | CORAL_ASR_MODEL | Loaded CoRal ASR model |
| prompt | STRING (optional) | Text prompt to guide Whisper transcription. Ignored for Wav2Vec2. |

| Output | Type | Description |
|---|---|---|
| texts | STRING[] | List of transcribed Danish texts |

## Installation

Clone this repository into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jgaarsdal/ComfyUI-JGUtils.git
```

Then restart ComfyUI. Dependencies (`transformers`) will be installed automatically if you use [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager), or you can install them manually:

```bash
pip install -r requirements.txt
```

## License

[MIT](LICENSE)
