<h2 align="center">VoxCPM2: Tokenizer-Free TTS for Multilingual Speech Generation, Creative Voice Design, and True-to-Life Cloning</h2>

<p align="center">
  <b>English</b> | <a href="./README_zh.md">中文</a>
</p>

<p align="center">
  <a href="https://github.com/OpenBMB/VoxCPM/"><img src="https://img.shields.io/badge/Project%20Page-GitHub-blue" alt="Project Page"></a>
  <a href="https://huggingface.co/spaces/OpenBMB/VoxCPM-Demo"><img src="https://img.shields.io/badge/Live%20Playground-Demo-orange" alt="Live Playground"></a>
  <a href="https://voxcpm.readthedocs.io/en/latest/"><img src="https://img.shields.io/badge/Docs-ReadTheDocs-8CA1AF" alt="Documentation"></a>
  <a href="https://huggingface.co/openbmb/VoxCPM2"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-VoxCPM2-yellow" alt="Hugging Face"></a>
  <a href="https://modelscope.cn/models/OpenBMB/VoxCPM2"><img src="https://img.shields.io/badge/ModelScope-VoxCPM2-purple" alt="ModelScope"></a>
  <a href="https://openbmb.github.io/voxcpm2-demopage/"><img src="https://img.shields.io/badge/DemoPage-Audio Samples-red" alt="DemoPage"></a>
  <a href="https://arxiv.org/abs/2606.06928"><img src="https://img.shields.io/badge/arXiv-VoxCPM2%20Technical%20Report-red" alt="VoxCPM2 Technical Report"></a>
</p>

<div align="center">
  <img src="assets/voxcpm_logo.png" alt="VoxCPM Logo" width="35%">
  <br><br>
  <a href="https://trendshift.io/repositories/17704" target="_blank"><img src="https://trendshift.io/api/badge/repositories/17704" alt="OpenBMB%2FVoxCPM | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>

<br>

<p align="center">
  👋 Join our community for discussion and support!
  <br>
  <a href="./assets/feishu-group.png" style="display:inline-block;vertical-align:middle; margin-left: 10px;">
    <img src="./assets/feishu-logo.png" width="16" height="16" style="vertical-align:middle;"> Feishu
  </a>
  &nbsp;|&nbsp;
  <a href="https://discord.gg/KZUx7tVNwz" style="display:inline-block;vertical-align:middle;">
    <img src="./assets/discord-logo.png" width="16" height="16" style="vertical-align:middle;"> Discord
  </a>
  &nbsp;|&nbsp;
  <a href="https://modelbest.feishu.cn/wiki/UtWxwcERfiRIpIkBOjuc3h9tn1D?fromScene=spaceOverview" style="display:inline-block;vertical-align:middle;">
    📚 MiniCPM Wiki
  </a>
</p>

VoxCPM is a **tokenizer-free** Text-to-Speech system that directly generates continuous speech representations via an end-to-end **diffusion autoregressive architecture**, bypassing discrete tokenization to achieve highly natural and expressive synthesis.

**VoxCPM2** is the latest major release — a **2B** parameter model trained on **over 2 million hours** of multilingual speech data, now supporting **30 languages**, **Voice Design**, **Controllable Voice Cloning**, and **48kHz** studio-quality audio output. Built on a [MiniCPM-4](https://github.com/OpenBMB/MiniCPM) backbone.

### ✨ Highlights

- 🌍 **30-Language Multilingual** — Input text in any of the 30 supported languages and synthesize directly, no language tag needed
- 🎨 **Voice Design** — Create a brand-new voice from a natural-language description alone (gender, age, tone, emotion, pace …), no reference audio required
- 🎛️ **Controllable Cloning** — Clone any voice from a short reference clip, with optional style guidance to steer emotion, pace, and expression while preserving the original timbre
- 🎙️ **Ultimate Cloning** — Reproduce every vocal nuance: provide both reference audio and its transcript, and the model continues seamlessly from the reference, faithfully preserving every vocal detail — timbre, rhythm, emotion, and style (same as VoxCPM1.5)
- 🔊 **48kHz High-Quality Audio** — Accepts 16kHz reference audio and directly outputs 48kHz studio-quality audio via AudioVAE V2's asymmetric encode/decode design, with built-in super-resolution — no external upsampler needed
- 🧠 **Context-Aware Synthesis** — Automatically infers appropriate prosody and expressiveness from text content
- ⚡ **Real-Time Streaming** — RTF as low as ~0.3 on NVIDIA RTX 4090, and ~0.13 accelerated by [Nano-vLLM](https://github.com/a710128/nanovllm-voxcpm) or [vLLM-Omni](https://github.com/vllm-project/vllm-omni) — official vLLM omni-modal serving for VoxCPM2 with PagedAttention and an OpenAI-compatible API
- 📜 **Fully Open-Source & Commercial-Ready** — Weights and code released under the [Apache-2.0](LICENSE) license, free for commercial use

**🌍 Supported Languages (30)**  
Arabic, Burmese, Chinese, Danish, Dutch, English, Finnish, French, German, Greek, Hebrew, Hindi, Indonesian, Italian, Japanese, Khmer, Korean, Lao, Malay, Norwegian, Polish, Portuguese, Russian, Spanish, Swahili, Swedish, Tagalog, Thai, Turkish, Vietnamese

Chinese Dialect: 四川话, 粤语, 吴语, 东北话, 河南话, 陕西话, 山东话, 天津话, 闽南话

### News

- **[2026.04]** 🔥 We release **VoxCPM2** — 2B, 30 languages, Voice Design & Controllable Voice Cloning, 48kHz audio output! [Weights](https://huggingface.co/openbmb/VoxCPM2) | [Docs](https://voxcpm.readthedocs.io/en/latest/) | [Playground](https://huggingface.co/spaces/OpenBMB/VoxCPM-Demo) | [Technical Report](https://arxiv.org/abs/2606.06928)
- **[2025.12]** 🎉 Open-source **VoxCPM1.5** [weights](https://huggingface.co/openbmb/VoxCPM1.5) with SFT & LoRA fine-tuning. (**🏆 #1 GitHub Trending**)
- **[2025.09]** 🔥 Release VoxCPM [Technical Report](https://arxiv.org/abs/2509.24650).
- **[2025.09]** 🎉 Open-source **VoxCPM-0.5B** [weights](https://huggingface.co/openbmb/VoxCPM-0.5B) (**🏆 #1 HuggingFace Trending**)

---

## Contents

- [Quick Start](#-quick-start)
  - [Installation](#installation)
  - [Python API](#python-api)
  - [CLI Usage](#cli-usage)
  - [Web Demo](#web-demo)
  - [Production Deployment](#-production-deployment-nano-vllm)
  - [On-Device Inference (llama.cpp-omni)](#-on-device-inference-llamacpp-omni)
- [Models & Versions](#-models--versions)
- [Performance](#-performance)
- [Fine-tuning](#%EF%B8%8F-fine-tuning)
- [Documentation](#-documentation)
- [Ecosystem & Community](#-ecosystem--community)
- [Risks and Limitations](#%EF%B8%8F-risks-and-limitations)
- [Citation](#-citation)

---

## 🚀 Quick Start

### Installation

```sh
pip install voxcpm
```

> **Requirements:** Python ≥ 3.10 (<3.13), PyTorch ≥ 2.5.0, CUDA ≥ 12.0. See [Quick Start Docs](https://voxcpm.readthedocs.io/en/latest/quickstart.html) for details.

### Python API

#### 🗣️ Text-to-Speech

```python
from voxcpm import VoxCPM
import soundfile as sf

model = VoxCPM.from_pretrained(
  "openbmb/VoxCPM2",
  load_denoiser=False,
)

wav = model.generate(
    text="VoxCPM2 is the current recommended release for realistic multilingual speech synthesis.",
    cfg_value=2.0,
    inference_timesteps=10,
    seed=42,
)
sf.write("demo.wav", wav, model.tts_model.sample_rate)
print("saved: demo.wav")
```

If you prefer downloading from ModelScope first, you can use:

```bash
pip install modelscope
```

```python
from modelscope import snapshot_download
snapshot_download("OpenBMB/VoxCPM2", local_dir='./pretrained_models/VoxCPM2') # specify the local directory to save the model

from voxcpm import VoxCPM
import soundfile as sf
model = VoxCPM.from_pretrained("./pretrained_models/VoxCPM2", load_denoiser=False)

wav = model.generate(
    text="VoxCPM2 is the current recommended release for realistic multilingual speech synthesis.",
    cfg_value=2.0,
    inference_timesteps=10,
    seed=42,
)
sf.write("demo.wav", wav, model.tts_model.sample_rate)
```

#### 🎨 Voice Design

Create a voice from a natural-language description — no reference audio needed. **Format:** put the description in parentheses at the start of `text`(e.g. `"(your voice description)The text to synthesize."`):

```python
wav = model.generate(
    text="(A young woman, gentle and sweet voice)Hello, welcome to VoxCPM2!",
    cfg_value=2.0,
    inference_timesteps=10,
    seed=42,
)
sf.write("voice_design.wav", wav, model.tts_model.sample_rate)
```

#### 🎛️ Controllable Voice Cloning

Upload a reference audio. The model clones the timbre, and you can still use control instructions to adjust speed, emotion, or style.

```python
wav = model.generate(
    text="This is a cloned voice generated by VoxCPM2.",
    reference_wav_path="path/to/voice.wav",
)
sf.write("clone.wav", wav, model.tts_model.sample_rate)

wav = model.generate(
    text="(slightly faster, cheerful tone)This is a cloned voice with style control.",
    reference_wav_path="path/to/voice.wav",
    cfg_value=2.0,
    inference_timesteps=10,
    seed=42,
)
sf.write("controllable_clone.wav", wav, model.tts_model.sample_rate)
```

#### 🎙️ Ultimate Cloning

Provide both the reference audio and its exact transcript for audio-continuation-based cloning with every vocal nuance reproduced. For maximum cloning similarity, pass the same reference clip to both `reference_wav_path` and `prompt_wav_path` as shown below:

```python
wav = model.generate(
    text="This is an ultimate cloning demonstration using VoxCPM2.",
    prompt_wav_path="path/to/voice.wav",
    prompt_text="The transcript of the reference audio.",
    reference_wav_path="path/to/voice.wav", # optional, for better simliarity 
)
sf.write("hifi_clone.wav", wav, model.tts_model.sample_rate)
```

**🔄 Streaming API**

```python
import numpy as np

chunks = []
for chunk in model.generate_streaming(
    text="Streaming text to speech is easy with VoxCPM!",
):
    chunks.append(chunk)
wav = np.concatenate(chunks)
sf.write("streaming.wav", wav, model.tts_model.sample_rate)
```



### CLI Usage

```bash
# Voice design (no reference audio needed)
voxcpm design \
  --text "VoxCPM2 brings studio-quality multilingual speech synthesis." \
  --output out.wav

# Controllable voice cloning with style control
voxcpm design \
  --text "VoxCPM2 brings studio-quality multilingual speech synthesis." \
  --control "Young female voice, warm and gentle, slightly smiling" \
  --seed 42 \
  --output out.wav

# Voice cloning (reference audio)
voxcpm clone \
  --text "This is a voice cloning demo." \
  --reference-audio path/to/voice.wav \
  --output out.wav

# Ultimate cloning (prompt audio + transcript)
voxcpm clone \
  --text "This is a voice cloning demo." \
  --prompt-audio path/to/voice.wav \
  --prompt-text "reference transcript" \
  --reference-audio path/to/voice.wav \ # optional, for better simliarity
  --output out.wav

# Batch processing
voxcpm batch --input examples/input.txt --output-dir outs

# Optional post-generation timestamps with stable-ts
pip install "voxcpm[timestamps]"
voxcpm design \
  --text "VoxCPM2 brings studio-quality multilingual speech synthesis." \
  --output out.wav \
  --timestamps \
  --timestamp-level word \
  --timestamp-language en

# Character timestamps are best-effort and are derived from word alignment
voxcpm design \
  --text "欢迎使用 VoxCPM2。" \
  --output out.wav \
  --timestamps \
  --timestamp-level char \
  --timestamp-language zh

# Help
voxcpm --help
```

### Video Dubbing (multi-speaker)

Re-voice a conversation video line by line from its SRT subtitles. The original audio is muted, so the final
video contains only the generated voices:

- **Characters** — tag speakers in the SRT as `[Alice] Hello` or `Alice: Hello`. Untagged lines are grouped by *voice*
  (CAM++ speaker embeddings) into *Speaker 1*, *Speaker 2*, … (`--num-speakers` fixes the count), and tagged lines whose
  voice does not match their character are kept out of that character's reference.
- **Retried until consistent** — every generated line is checked (CAM++) against its speaker's voice and regenerated
  with a new seed until it is consistent: close enough to the speaker's generated first line *and* to the average of
  their lines accepted so far (`min_consistency`, 0.55), more like them than like anyone else, and at the speaker's
  pitch. Up to `--max-attempts` tries (default 8; a speaker's first line, which all their other lines copy, gets at
  least 12). Lines that still fail are marked ⚠ in the report and listed in a warning.
- **Accurate, consistent tone** — gender (pitch), emotion (SenseVoice: happy, sad, angry, …) and prosody are detected per line.
  Prosody is described relative to each character's *own* baseline, every character keeps one reference voice and one fixed
  seed, and a designed voice is reused as the reference for that character's later lines.
- **Clean output** — reference clips are denoised once (ZipEnhancer); every generated line is high-passed, spectrally
  gated and loudness-levelled, breaths/mumbles after the last word are cut, edges are faded, and the line is fitted into
  its time slot (silence trimmed, sped up with ffmpeg `atempo` by at most `--max-speedup`, or regenerated at a faster
  pace; trimmed only if it would overlap the next line).
- **One consistent voice per speaker (default)** — every speaker (*Speaker 1*, *Speaker 2*, … or your tags) gets a
  fixed voice prompt scanned from their own original lines (gender, pitch register, delivery, pace and loudness
  compared with the rest of the cast, usual mood; e.g. `Adult female voice, high, bright pitch, steady, calm delivery`).
  Their first line (long enough, at an adult pitch) is copied from the original video (`clone_first`), and every later
  line of that speaker clones that *generated* first line with the same prompt and seed, so line 2, 3, … sound like
  line 1. The report shows this as `clone→#<first line>`. Short lines (slot ≤ 1.6 s, e.g. a one-word reply) get only
  their emotion as a style hint — with the full voice prompt VoxCPM tends to babble on for seconds after one word — and
  every line's generation is capped at about twice its slot; a try that runs on or is nearly silent is retried. Short
  clips are voice-checked too (looped to a checkable length) and clips under 1 s are never pitch-shifted. Only the emotion and that line's difference from the
  speaker's usual delivery change per line; pitch follows the original line relative to the speaker's own usual pitch
  (within -2/+3 semitones) and is corrected by at most 2 semitones. Edit the prompts in the Characters table or in
  `--speakers`; `{"Speaker 1": {"reference": "my_voice.wav"}}` uses your own recording for every line instead.
- **Two voices only (`--two-voices`)** — voice the whole video with one adult male and one adult female voice. Each
  speaker goes to the voice of their gender; each voice copies the first adult line of that gender (`clone_first`; set
  `"voice_mode": "design"` to design it from a description instead), and each line is *styled* after its original
  speaker — pitch register, delivery, pace and loudness (e.g. `deep pitch, fast-paced, soft-spoken`) — so speakers
  sharing a voice still speak differently (`--no-speaker-style` turns this off). With `design`, the designed voice is
  first checked against the original speakers (pitch, and CAM++ voice match) and re-designed with new seeds (up to 5
  tries) until it fits; the closest reading is kept and pitch-corrected, with a warning naming the clip, when none fits.
- Voice modes: `clone_first`, `clone_speaker` (one cloned voice per character), `clone_line` (clone each line from its own original audio),
  `design` (voice from a description such as `young woman, bright and cheerful`).
- Without `--srt`, the video is auto-transcribed (`pip install "voxcpm[timestamps]"`). Requires `ffmpeg` on PATH.
- **VoxCPM2 only** — dubbing refuses to run with VoxCPM 1.x models (it needs VoxCPM2's reference cloning with style
  control, 48 kHz output and 30-language support).

```bash
voxcpm dub --video scene.mp4 --srt scene.srt --output scene_dub.mp4

# Per-character overrides
cat > speakers.json <<'JSON'
{"Alice": {"gender": "female", "voice_mode": "design", "description": "young woman, bright and cheerful"},
 "Bob":   {"voice_mode": "clone_speaker"}}
JSON
voxcpm dub --video scene.mp4 --srt scene.srt --speakers speakers.json --output scene_dub.mp4
```

Outputs: `scene_dub.mp4` (with soft subtitles), `scene_dub_dub.wav`, `scene_dub.srt` and `scene_dub_speakers.srt`.
Optional: `--background instrumental` keeps the original music & effects by removing its vocals with Hybrid Demucs
(weights downloaded once from `download.pytorch.org`); `--background duck|keep` keeps the original audio.
`--no-emotion` / `--no-clean` turn those steps off. Video dubbing is CLI-only; the web demo has the script-only
**📝 SRT → Speech** tab below.

### SRT → Speech (script only, no video)

Voice a whole subtitle script in bulk when there is no video, or when the original audio is too poor to clone from.
Tag each line with its speaker and, optionally, gender, age and emotion (any order; gender and age once per speaker):

```text
1
00:00:00,200 --> 00:00:02,800
[Dara|male|kid|sad] ម៉ាក់ កូននឹកម៉ាក់ណាស់

2
00:00:03,800 --> 00:00:05,400
[Srey|female|adult] រៀបការអស់រយៈពេល ៧ ឆ្នាំ

3
00:00:05,600 --> 00:00:07,000
[Dara] ម៉ាក់?
```

- Gender `male`/`female` (or `boy`/`girl`, which also mean kid), age `adult`/`kid`, emotion `happy`, `sad`, `angry`,
  `fearful`, `surprised`, … or any word. Untagged lines are read by *Narrator*. Missing emotions are guessed from
  punctuation (`?!`, `!`, `…`) and English keywords (`--no-guess-emotion` turns this off).
- Each speaker's voice is made once as a **calm, neutral reading of their first line** (joined with their next lines
  until it is ~4 s long), retried until its pitch is in the range for their gender and age (e.g. 200–380 Hz for a boy).
  **Every line of that speaker — the first one too — clones that voice**, adds its own emotion as a style hint and is
  retried until it is consistent with it (CAM++ voice check, up to `--max-attempts`). Being neutral, the voice does not
  carry one line's emotion (say, a happy first line) into all the others.
- **Clear male / female** — a man's voice is made between 90 and 140 Hz and a woman's between 185 and 255 Hz (well
  away from the ~165 Hz point where they meet), and every line must stay on its side (a man's under 160 Hz, a woman's
  over 175 Hz, a kid's over 200 Hz) or it is redone; lines that still are not are listed in a warning. Each line also
  stays within 2 semitones of its speaker's voice and at least 0.6 close to it (CAM++).
- **Silences removed** — when a line is finalized the silence before and after the speech is cut and every pause
  inside it longer than `--max-pause` (0.2 s) is shortened to it, so words never run together but no dead air is left
  (`--keep-silence` turns this off).
- In the combined track lines keep their natural pace (sped up by at most `--max-speedup`, 1.1); a line that is still
  longer than its slot pushes the following lines a little later instead of being squeezed.
- Outputs: `<output>.wav` (all lines on the SRT timing), `<output>_lines/` (every line as its own file, at its natural
  length; `--pad` pads a line shorter than its subtitle slot with trailing silence to the slot's length),
  `<output>_tagged.srt` (every line tagged with speaker, gender, age and emotion — edit it and run again).

```bash
voxcpm srt --srt script.srt --output script.wav            # bulk generate
voxcpm srt --srt script.srt --output script.wav --tag-only # only write script_tagged.srt to check emotions
voxcpm srt --srt script.srt --output script.wav --voices voices.json
# redo only failed lines of an earlier run (same voices, new seeds); the command is printed after a run
voxcpm srt --srt script.srt --output script.wav --workdir work --regenerate 5,31
# voices.json: {"Dara": {"gender": "male", "age": "kid", "description": "cheeky little boy", "reference": "dara.wav"}}
```

The web demo has the same workflow in the **📝 SRT → Speech** tab (where you can also add the **original video**:
**Analyze** then reads each line's tone — louder, softer, livelier, faster, … than that speaker usually is — from
the original speech at its SRT time; optionally also a rough emotion (SenseVoice) for lines with none in their tag): listen to and download every line (or all of
them as one `.zip`), **🔁 Regenerate** the lines that need a look (they are pre-selected), or give a speaker a
**🎭 New voice** (their voice is designed again and all of their lines are regenerated).
Projects are saved as you go in `~/.voxcpm/srt_projects/` (set `VOXCPM_PROJECTS_DIR` to change it): closing or
reloading the tab does not stop a generation, and reopening the page brings back the last project — its tables, every
line already generated, and the progress of a run still going. Lines left unfinished (⏳) can be finished with
**Regenerate**.

When the lines are generated, **🎬 Next: place the vocals on your video** opens the **🎬 Video Editor** tab (needs
`ffmpeg`). Drop in the video: every line becomes a clip on its speaker's track at its subtitle time, and the speech in
the video's own soundtrack is found and shown on a **Video speech** track (optionally after separating the voices from
the music), so **🎯 Align all to video speech** can line every clip up with the speech it replaces. Like in a video
editor you can play it all back over the video, drag clips (with snapping) left / right in time and up / down to
another row (clips that end up overlapping after aligning are put on separate rows, or **⇥ Remove overlaps** moves them
apart), trim their edges, **✂ cut** a clip at the playhead (or Alt+click) and move or delete the pieces, **🔗 join** the
pieces back, speed a clip up or slow it down (0.5×–2×, pitch kept; **Fit to subtitle** makes it fill its subtitle
time), nudge clips, keep every voice at the same loudness (**🎚 Level voices**, on by default: each clip is measured as it
plays — after trims, cuts, speed changes and regeneration — and brought to the same speech level, evened out from the inside too —
loud words turned down, quiet ones up, 3:1 — with a limiter so nothing clips; a clip's own volume is a change on top), change their volume, switch the **original video sound** on / off (🔊 in the toolbar, **M** on the Video speech track, or O), show the lines as captions on the preview (**CC**, off by default, never in the export), work in **⛶ Full screen** (F; drag the bar under the video to give the timeline more room), mute a clip or a whole speaker, undo / redo, and edit a line's text,
emotion or speaker and
**🔁 Regenerate** it (or give a speaker a **🎭 New voice**) without losing its place on the timeline. **⬇ Export
video** writes the video with the new voices over the original soundtrack: by default the original is muted wherever a
new voice replaces it (the whole stretch of original speech it covers) and kept everywhere else. Subtitles burned into the
picture are found and **removed** (filled in from around them, or blurred; **✏️ Area** moves the box), which
re-encodes the picture at its original size, codec and frame rate; with them kept, the picture is copied as it is. **New subtitles** are made from the
voice lines (one per line, on its voice's time) and drawn where the original ones were (or near the bottom): edit
them on the **💬 Subtitles** track above the video speech — double-click or **+** / N to add one, drag it or its
edges to change its time (subtitles that overlap in time go on separate rows — drag up/down or Alt+↑/↓; a
higher row is drawn right above the lower ones on the video), click it to edit its text or split it, Delete to remove it, **↻ Make again from the
voices** to start over. The editor draws them (so Khmer and other complex scripts look right) and the export
burns exactly that into the picture; it can also be lowered
under the new voices, kept, have **its voices removed** (music and sound effects kept — separated as soon as you
choose it, so you hear it while editing), or muted, plus the soundtrack and the new voices alone as `.wav`. The edit is saved with the project.

### Web Demo

```bash
python app.py --port 8808  # then open in browser: http://localhost:8808
```

Use `--device` to choose the runtime device:

```bash
python app.py --device auto
```

Supported values are `auto`, `cpu`, `mps`, `cuda`, and `cuda:N`. On Apple Silicon Macs, `auto` uses MPS when available.

### 🚢 Production Deployment (Nano-vLLM)

For high-throughput serving, use **[Nano-vLLM-VoxCPM](https://github.com/a710128/nanovllm-voxcpm)** — a dedicated inference engine built on Nano-vLLM with concurrent request support and an async API.

```bash
pip install nano-vllm-voxcpm
```

```python
from nanovllm_voxcpm import VoxCPM
import numpy as np, soundfile as sf

server = VoxCPM.from_pretrained(model="/path/to/VoxCPM", devices=[0])
chunks = list(server.generate(target_text="Hello from VoxCPM!"))
sf.write("out.wav", np.concatenate(chunks), 48000)
server.stop()
```

> **RTF as low as ~0.13 on NVIDIA RTX 4090** (vs ~0.3 with the standard PyTorch implementation), with support for batched concurrent requests and a FastAPI HTTP server. See the [Nano-vLLM-VoxCPM repo](https://github.com/a710128/nanovllm-voxcpm) for deployment details.

### 🏭 Production Serving (vLLM-Omni)

For production multi-tenant deployments, use **[vLLM-Omni](https://github.com/vllm-project/vllm-omni)** — the official vLLM project's omni-modal extension with native **VoxCPM2** support. PagedAttention KV cache, continuous batching, and a drop-in **OpenAI-compatible** `/v1/audio/speech` endpoint.

```bash
# Install from source (latest main — vllm-omni is rapidly evolving)
uv pip install vllm==0.19.0 --torch-backend=auto
git clone https://github.com/vllm-project/vllm-omni.git && cd vllm-omni
uv pip install -e .
```

See the [vLLM-Omni installation guide](https://vllm-omni.readthedocs.io/en/latest/getting_started/installation/) for other platforms (ROCm, XPU, MUSA, NPU) and Docker images.

```bash
# Launch an OpenAI-compatible TTS server (--omni enables omni-modal serving)
vllm serve openbmb/VoxCPM2 --omni --port 8000

# Call it from any OpenAI client
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"openbmb/VoxCPM2","input":"Hello from VoxCPM2 on vLLM-Omni!","voice":"default"}' \
  --output out.wav
```

> Built on the upstream vLLM scheduler, with batched concurrent requests, streaming chunk delivery, and multi-GPU deployment out of the box. See the [VoxCPM2 example](https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/voxcpm2) for full deployment recipes.

### 📱 On-Device Inference (llama.cpp-omni)

For on-device / edge deployment without Python, use **[llama.cpp-omni](https://github.com/tc-mb/llama.cpp-omni)** — a high-performance C++ inference engine built on llama.cpp, with native VoxCPM2 GGUF support on **CPU / Metal / CUDA / Vulkan**.

**1. Download GGUF weights** from [HuggingFace](https://huggingface.co/DennisHuang648/VoxCPM2-GGUF) | [ModelScope](https://modelscope.cn/models/DennisHuang/VoxCPM2-GGUF) — you need one **BaseLM** (F16 or Q8_0) + the **Acoustic** file. Q8_0 halves the download with negligible quality loss.

**2. Build**

```bash
git clone https://github.com/tc-mb/llama.cpp-omni.git && cd llama.cpp-omni
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target voxcpm2-cli -j
```

> CMake auto-detects Metal (macOS) or CUDA (Linux with NVIDIA GPU).

**3. Run**

```bash
# Basic TTS
./build/bin/voxcpm2-cli \
    -t "Hello, this is VoxCPM2 running through llama.cpp-omni." \
    -o output.wav VoxCPM2-BaseLM-Q8_0.gguf VoxCPM2-Acoustic-F16.gguf

# Voice cloning (reference audio)
./build/bin/voxcpm2-cli \
    -t "Cloned voice." -r speaker.wav -o clone.wav \
    VoxCPM2-BaseLM-Q8_0.gguf VoxCPM2-Acoustic-F16.gguf

# Ultimate cloning (reference audio + transcript)
./build/bin/voxcpm2-cli \
    -t "Target text." --prompt-wav speaker.wav --prompt-text "transcript of speaker.wav" \
    -o clone.wav VoxCPM2-BaseLM-Q8_0.gguf VoxCPM2-Acoustic-F16.gguf
```

> **RTF ~1.76 (Q8_0) on Apple M4 Pro / Metal.** Key flags: `--cfg` (guidance scale), `--timesteps` (CFM steps), `--seed`, `--temperature`, `--stream`. See the [llama.cpp-omni repo](https://github.com/tc-mb/llama.cpp-omni) and [GGUF weights page](https://huggingface.co/DennisHuang648/VoxCPM2-GGUF) for full details.

> **Full parameter reference, multi-scenario examples, and voice cloning tips →** [Quick Start Guide](https://voxcpm.readthedocs.io/en/latest/quickstart.html) | [Usage Guide](https://voxcpm.readthedocs.io/en/latest/usage_guide.html) | [Cookbook](https://voxcpm.readthedocs.io/en/latest/cookbook.html)

---

## 📦 Models & Versions


|                                 | **VoxCPM2**                                                                                          | **VoxCPM1.5**                                                                                            | **VoxCPM-0.5B**                                                                                              |
| ------------------------------- | ---------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **Status**                      | 🟢 Latest                                                                                            | Stable                                                                                                   | Legacy                                                                                                       |
| **Backbone Parameters**         | 2B                                                                                                   | 0.6B                                                                                                     | 0.5B                                                                                                         |
| **Audio Sample Rate**           | 48kHz                                                                                                | 44.1kHz                                                                                                  | 16kHz                                                                                                        |
| **LM Token Rate**               | 6.25Hz                                                                                               | 6.25Hz                                                                                                   | 12.5Hz                                                                                                       |
| **Languages**                   | 30                                                                                                   | 2 (zh, en)                                                                                               | 2 (zh, en)                                                                                                   |
| **Cloning Mode**                | Isolated Reference & Continuation                                                                    | Continuation only                                                                                        | Continuation only                                                                                            |
| **Voice Design**                | ✅                                                                                                    | —                                                                                                        | —                                                                                                            |
| **Controllable Voice Cloning**  | ✅                                                                                                    | —                                                                                                        | —                                                                                                            |
| **SFT / LoRA**                  | ✅                                                                                                    | ✅                                                                                                        | ✅                                                                                                            |
| **RTF (RTX 4090)**              | ~0.30                                                                                                | ~0.15                                                                                                    | ~0.17                                                                                                        |
| **RTF in Nano-VLLM (RTX 4090)** | ~0.13                                                                                                | ~0.08                                                                                                    | ~0.10                                                                                                        |
| **VRAM**                        | ~8 GB                                                                                                | ~6 GB                                                                                                    | ~5 GB                                                                                                        |
| **Weights**                     | [🤗 HF](https://huggingface.co/openbmb/VoxCPM2) / [MS](https://modelscope.cn/models/OpenBMB/VoxCPM2) | [🤗 HF](https://huggingface.co/openbmb/VoxCPM1.5) / [MS](https://modelscope.cn/models/OpenBMB/VoxCPM1.5) | [🤗 HF](https://huggingface.co/openbmb/VoxCPM-0.5B) / [MS](https://modelscope.cn/models/OpenBMB/VoxCPM-0.5B) |
| **Technical Report**            | [arXiv](https://arxiv.org/abs/2606.06928)                                                            | —                                                                                                        | [arXiv](https://arxiv.org/abs/2509.24650) [ICLR 2026](https://openreview.net/forum?id=h5KLpGoqzC)            |
| **Demo Page**                   | [Audio Samples](https://openbmb.github.io/voxcpm2-demopage)                                          | —                                                                                                        | [Audio Samples](https://openbmb.github.io/VoxCPM-demopage)                                                   |


VoxCPM2 is built on a **tokenizer-free, diffusion autoregressive** paradigm. The model operates entirely in the latent space of **AudioVAE V2**, following a four-stage pipeline: **LocEnc → TSLM → RALM → LocDiT**, enabling rich expressiveness and 48kHz native audio output.

<div align="center">
  <img src="assets/voxcpm2_model.png" alt="VoxCPM2 Model Architecture" width="90%">
</div>

> For full architectural details, VoxCPM2-specific upgrades, and a model comparison table, see the [Architecture Design](https://voxcpm.readthedocs.io/en/latest/models/architecture.html).

---

## 📊 Performance

VoxCPM2 achieves state-of-the-art or comparable results on public zero-shot and controllable TTS benchmarks. 

### Seed-TTS-eval

**Seed-TTS-eval WER(⬇)&SIM(⬆) Results (click to expand)**


| Model             | Parameters | Open-Source | test-EN |        | test-ZH |        | test-Hard |        |
| ----------------- | ---------- | ----------- | ------- | ------ | ------- | ------ | --------- | ------ |
|                   |            |             | WER/%⬇  | SIM/%⬆ | CER/%⬇  | SIM/%⬆ | CER/%⬇    | SIM/%⬆ |
| MegaTTS3          | 0.5B       | ❌           | 2.79    | 77.1   | 1.52    | 79.0   | -         | -      |
| DiTAR             | 0.6B       | ❌           | 1.69    | 73.5   | 1.02    | 75.3   | -         | -      |
| CosyVoice3        | 0.5B       | ❌           | 2.02    | 71.8   | 1.16    | 78.0   | 6.08      | 75.8   |
| CosyVoice3        | 1.5B       | ❌           | 2.22    | 72.0   | 1.12    | 78.1   | 5.83      | 75.8   |
| Seed-TTS          | -          | ❌           | 2.25    | 76.2   | 1.12    | 79.6   | 7.59      | 77.6   |
| MiniMax-Speech    | -          | ❌           | 1.65    | 69.2   | 0.83    | 78.3   | -         | -      |
| F5-TTS            | 0.3B       | ✅           | 2.00    | 67.0   | 1.53    | 76.0   | 8.67      | 71.3   |
| MaskGCT           | 1B         | ✅           | 2.62    | 71.7   | 2.27    | 77.4   | -         | -      |
| CosyVoice         | 0.3B       | ✅           | 4.29    | 60.9   | 3.63    | 72.3   | 11.75     | 70.9   |
| CosyVoice2        | 0.5B       | ✅           | 3.09    | 65.9   | 1.38    | 75.7   | 6.83      | 72.4   |
| SparkTTS          | 0.5B       | ✅           | 3.14    | 57.3   | 1.54    | 66.0   | -         | -      |
| FireRedTTS        | 0.5B       | ✅           | 3.82    | 46.0   | 1.51    | 63.5   | 17.45     | 62.1   |
| FireRedTTS-2      | 1.5B       | ✅           | 1.95    | 66.5   | 1.14    | 73.6   | -         | -      |
| Qwen2.5-Omni      | 7B         | ✅           | 2.72    | 63.2   | 1.70    | 75.2   | 7.97      | 74.7   |
| Qwen3-Omni        | 30B-A3B    | ✅           | 1.39    | -      | 1.07    | -      | -         | -      |
| OpenAudio-s1-mini | 0.5B       | ✅           | 1.94    | 55.0   | 1.18    | 68.5   | 23.37     | 64.3   |
| IndexTTS2         | 1.5B       | ✅           | 2.23    | 70.6   | 1.03    | 76.5   | 7.12      | 75.5   |
| VibeVoice         | 1.5B       | ✅           | 3.04    | 68.9   | 1.16    | 74.4   | -         | -      |
| HiggsAudio-v2     | 3B         | ✅           | 2.44    | 67.7   | 1.50    | 74.0   | 55.07     | 65.6   |
| VoxCPM-0.5B       | 0.6B       | ✅           | 1.85    | 72.9   | 0.93    | 77.2   | 8.87      | 73.0   |
| VoxCPM1.5         | 0.8B       | ✅           | 2.12    | 71.4   | 1.18    | 77.0   | 7.74      | 73.1   |
| MOSS-TTS          |            | ✅           | 1.85    | 73.4   | 1.20    | 78.8   | -         | -      |
| Qwen3-TTS         | 1.7B       | ✅           | 1.23    | 71.7   | 1.22    | 77.0   | 6.76      | 74.8   |
| FishAudio S2      | 4B         | ✅           | 0.99    | -      | 0.54    | -      | 5.99      | -      |
| LongCat-Audio-DiT | 3.5B       | ✅           | 1.50    | 78.6   | 1.09    | 81.8   | 6.04      | 79.7   |
| **VoxCPM2**       | 2B         | ✅           | 1.84    | 75.3   | 0.97    | 79.5   | 8.13      | 75.3   |




### CV3-eval

**CV3-eval Multilingual WER/CER(⬇) Results (click to expand)**


| Model           | zh   | en   | hard-zh | hard-en | ja   | ko   | de   | es   | fr   | it   | ru   |
| --------------- | ---- | ---- | ------- | ------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| CosyVoice2      | 4.08 | 6.32 | 12.58   | 11.96   | 9.13 | 19.7 | -    | -    | -    | -    | -    |
| CosyVoice3-1.5B | 3.91 | 4.99 | 9.77    | 10.55   | 7.57 | 5.69 | 6.43 | 4.47 | 11.8 | 10.5 | 6.64 |
| Fish Audio S2   | 2.65 | 2.43 | 9.10    | 4.40    | 3.96 | 2.76 | 2.22 | 2.00 | 6.26 | 2.04 | 2.78 |
| **VoxCPM2**     | 3.65 | 5.00 | 8.55    | 8.48    | 5.96 | 5.69 | 4.77 | 3.80 | 9.85 | 4.25 | 5.21 |




### MiniMax-Multilingual-Test

**Minimax-MLS-test WER(⬇) Results (click to expand)**


| Language   | Minimax   | ElevenLabs | Qwen3-TTS | FishAudio S2 | **VoxCPM2** |
| ---------- | --------- | ---------- | --------- | ------------ | ----------- |
| Arabic     | **1.665** | 1.666      | –         | 3.500        | 13.046      |
| Cantonese  | 34.111    | 51.513     | –         | **30.670**   | 38.584      |
| Chinese    | 2.252     | 16.026     | 0.928     | **0.730**    | 1.136       |
| Czech      | 3.875     | **2.108**  | –         | 2.840        | 24.132      |
| Dutch      | 1.143     | **0.803**  | –         | 0.990        | 0.913       |
| English    | 2.164     | 2.339      | **0.934** | 1.620        | 2.289       |
| Finnish    | 4.666     | 2.964      | –         | 3.330        | **2.632**   |
| French     | 4.099     | 5.216      | **2.858** | 3.050        | 4.534       |
| German     | 1.906     | 0.572      | 1.235     | **0.550**    | 0.679       |
| Greek      | 2.016     | **0.991**  | –         | 5.740        | 2.844       |
| Hindi      | 6.962     | **5.827**  | –         | 14.640       | 19.699      |
| Indonesian | 1.237     | **1.059**  | –         | 1.460        | 1.084       |
| Italian    | 1.543     | 1.743      | **0.948** | 1.270        | 1.563       |
| Japanese   | 3.519     | 10.646     | 3.823     | **2.760**    | 4.628       |
| Korean     | 1.747     | 1.865      | 1.755     | **1.180**    | 1.962       |
| Polish     | 1.415     | **0.766**  | –         | 1.260        | 1.141       |
| Portuguese | 1.877     | 1.331      | 1.526     | **1.140**    | 1.938       |
| Romanian   | 2.878     | **1.347**  | –         | 10.740       | 21.577      |
| Russian    | 4.281     | 3.878      | 3.212     | **2.400**    | 3.634       |
| Spanish    | 1.029     | 1.084      | 1.126     | **0.910**    | 1.438       |
| Thai       | 2.701     | 73.936     | –         | 4.230        | 2.961       |
| Turkish    | 1.52      | 0.699      | –         | 0.870        | 0.817       |
| Ukrainian  | 1.082     | **0.997**  | –         | 2.300        | 6.316       |
| Vietnamese | **0.88**  | 73.415     | –         | 7.410        | 3.307       |




**Minimax-MLS-test SIM(⬆) Results (click to expand)**


| Language   | Minimax  | ElevenLabs | Qwen3-TTS | FishAudio S2 | **VoxCPM2** |
| ---------- | -------- | ---------- | --------- | ------------ | ----------- |
| Arabic     | 73.6     | 70.6       | –         | 75.0         | **79.1**    |
| Cantonese  | 77.8     | 67.0       | –         | 80.5         | **83.5**    |
| Chinese    | 78.0     | 67.7       | 79.9      | 81.6         | **82.5**    |
| Czech      | 79.6     | 68.5       | –         | **79.8**     | 78.3        |
| Dutch      | 73.8     | 68.0       | –         | 73.0         | **80.8**    |
| English    | 75.6     | 61.3       | 77.5      | 79.7         | **85.4**    |
| Finnish    | 83.5     | 75.9       | –         | 81.9         | **89.0**    |
| French     | 62.8     | 53.5       | 62.8      | 69.8         | **73.5**    |
| German     | 73.3     | 61.4       | 77.5      | 76.7         | **80.3**    |
| Greek      | 82.6     | 73.3       | –         | 79.5         | **86.0**    |
| Hindi      | 81.8     | 73.0       | –         | 82.1         | **85.6**    |
| Indonesian | 72.9     | 66.0       | –         | 76.3         | **80.0**    |
| Italian    | 69.9     | 57.9       | 81.7      | 74.7         | **78.0**    |
| Japanese   | 77.6     | 73.8       | 78.8      | 79.6         | **82.8**    |
| Korean     | 77.6     | 70.0       | 79.9      | 81.7         | **83.3**    |
| Polish     | 80.2     | 72.9       | –         | 81.9         | **88.4**    |
| Portuguese | 80.5     | 71.1       | 81.7      | 78.1         | **83.7**    |
| Romanian   | **80.9** | 69.9       | –         | 73.3         | 79.7        |
| Russian    | 76.1     | 67.6       | 79.2      | 79.0         | **81.1**    |
| Spanish    | 76.2     | 61.5       | 81.4      | 77.6         | **83.1**    |
| Thai       | 80.0     | 58.8       | –         | 78.6         | **84.0**    |
| Turkish    | 77.9     | 59.6       | –         | 83.5         | **87.1**    |
| Ukrainian  | 73.0     | 64.7       | –         | 74.7         | **79.8**    |
| Vietnamese | 74.3     | 36.9       | –         | 74.0         | **80.6**    |




### Internal 30-Language ASR Benchmark

We additionally run an internal multilingual intelligibility benchmark with **30 languages × 500 samples**. ASR transcription is evaluated via **Gemini 3.1 Flash Lite API**.

**Internal 30-Language ASR Benchmark (click to expand)**


| Language               | Metric | VoxCPM2   | Fish S2-Pro |
| ---------------------- | ------ | --------- | ----------- |
| ar (Arabic)            | CER    | 1.23%     | 0.30%       |
| da (Danish)            | WER    | 2.70%     | 3.52%       |
| de (German)            | WER    | 0.96%     | 0.64%       |
| el (Greek)             | WER    | 3.17%     | 4.61%       |
| en (English)           | WER    | 0.42%     | 1.03%       |
| es (Spanish)           | WER    | 1.33%     | 0.64%       |
| fi (Finnish)           | WER    | 2.24%     | 2.80%       |
| fr (French)            | WER    | 2.16%     | 2.34%       |
| he (Hebrew)            | CER    | 2.98%     | 15.27%      |
| hi (Hindi)             | CER    | 0.79%     | 0.91%       |
| id (Indonesian)        | WER    | 1.36%     | 1.68%       |
| it (Italian)           | WER    | 1.65%     | 1.08%       |
| ja (Japanese)          | CER    | 2.40%     | 1.82%       |
| km (Khmer)             | CER    | 2.05%     | 75.15%      |
| ko (Korean)            | CER    | 0.95%     | 0.29%       |
| lo (Lao)               | CER    | 1.90%     | 87.40%      |
| ms (Malay)             | WER    | 1.75%     | 1.41%       |
| my (Burmese)           | CER    | 1.42%     | 85.27%      |
| nl (Dutch)             | WER    | 1.25%     | 1.68%       |
| no (Norwegian)         | WER    | 2.49%     | 3.76%       |
| pl (Polish)            | WER    | 1.90%     | 1.65%       |
| pt (Portuguese)        | WER    | 1.48%     | 1.49%       |
| ru (Russian)           | WER    | 0.90%     | 0.86%       |
| sv (Swedish)           | WER    | 2.22%     | 2.63%       |
| sw (Swahili)           | CER    | 1.07%     | 2.02%       |
| th (Thai)              | CER    | 0.94%     | 1.92%       |
| tl (Tagalog)           | WER    | 2.63%     | 4.00%       |
| tr (Turkish)           | WER    | 1.65%     | 1.65%       |
| vi (Vietnamese)        | WER    | 1.56%     | 5.56%       |
| zh (Chinese)           | CER    | 0.92%     | 1.02%       |
| Average (30 languages) |        | **1.68%** | -           |




### InstructTTSEval

**Instruction-Guided Voice Design Results (click to expand)**


| Model                  | InstructTTSEval-ZH |          |          | InstructTTSEval-EN |          |          |
| ---------------------- | ------------------ | -------- | -------- | ------------------ | -------- | -------- |
|                        | APS⬆               | DSD⬆     | RP⬆      | APS⬆               | DSD⬆     | RP⬆      |
| Hume                   | –                  | –        | –        | 83.0               | 75.3     | 54.3     |
| VoxInstruct            | 47.5               | 52.3     | 42.6     | 54.9               | 57.0     | 39.3     |
| Parler-tts-mini        | –                  | –        | –        | 63.4               | 48.7     | 28.6     |
| Parler-tts-large       | –                  | –        | –        | 60.0               | 45.9     | 31.2     |
| PromptTTS              | –                  | –        | –        | 64.3               | 47.2     | 31.4     |
| PromptStyle            | –                  | –        | –        | 57.4               | 46.4     | 30.9     |
| VoiceSculptor          | 75.7               | 64.7     | 61.5     | –                  | –        | –        |
| Mimo-Audio-7B-Instruct | 75.7               | 74.3     | 61.5     | 80.6               | 77.6     | 59.5     |
| Qwen3TTS-12Hz-1.7B-VD  | **85.2**           | **81.1** | **65.1** | 82.9               | 82.4     | 68.4     |
| **VoxCPM2**            | **85.2**           | 71.5     | 60.8     | **84.2**           | **83.2** | **71.4** |




---

## ⚙️ Fine-tuning

VoxCPM supports both **full fine-tuning (SFT)** and **LoRA fine-tuning**. With as little as **5–10 minutes** of audio, you can adapt to a specific speaker, language, or domain.

```bash
# LoRA fine-tuning (parameter-efficient, recommended)
python scripts/train_voxcpm_finetune.py \
    --config_path conf/voxcpm_v2/voxcpm_finetune_lora.yaml

# Full fine-tuning
python scripts/train_voxcpm_finetune.py \
    --config_path conf/voxcpm_v2/voxcpm_finetune_all.yaml

# WebUI for training & inference
python lora_ft_webui.py   # then open http://localhost:7860
```

> **Full guide →** [Fine-tuning Guide](https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html) (data preparation, configuration, training, LoRA hot-swapping, FAQ)

---

## 📚 Documentation

Full documentation: **[voxcpm.readthedocs.io](https://voxcpm.readthedocs.io/en/latest/)**


| Topic                      | Link                                                                                  |
| -------------------------- | ------------------------------------------------------------------------------------- |
| Quick Start & Installation | [Quick Start](https://voxcpm.readthedocs.io/en/latest/quickstart.html)                |
| Usage Guide & Cookbook     | [User Guide](https://voxcpm.readthedocs.io/en/latest/usage_guide.html)                |
| VoxCPM Series              | [Models](https://voxcpm.readthedocs.io/en/latest/models/version_history.html)         |
| Fine-tuning (SFT & LoRA)   | [Fine-tuning Guide](https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html) |
| FAQ & Troubleshooting      | [FAQ](https://voxcpm.readthedocs.io/en/latest/faq.html)                               |


---

## 🌟 Ecosystem & Community


| Project                                                                     | Description                                                                                      |
| --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **[Nano-vLLM](https://github.com/a710128/nanovllm-voxcpm)**                 | High-throughput and Fast GPU serving                                                             |
| **[vLLM-Omni](https://github.com/vllm-project/vllm-omni)**                  | Official vLLM omni-modal serving for VoxCPM2 — PagedAttention, OpenAI-compatible API             |
| **[llama.cpp-omni](https://github.com/tc-mb/llama.cpp-omni)**               | Full-duplex omni inference engine — VoxCPM2 GGUF on CPU / Metal / CUDA / Vulkan                   |
| **[VoxCPM.cpp](https://github.com/bluryar/VoxCPM.cpp)**                     | GGML/GGUF: CPU, CUDA, Vulkan inference                                                           |
| **[audio.cpp](https://github.com/0xShug0/audio.cpp)**                       | ggml-based unified C++ inference framework — CPU/CUDA/Vulkan/Metal, CLI & server, no Python      |
| **[VoxCPM-ONNX](https://github.com/bluryar/VoxCPM-ONNX)**                   | ONNX export for CPU inference                                                                    |
| **[VoxCPMANE](https://github.com/0seba/VoxCPMANE)**                         | Apple Neural Engine backend                                                                      |
| **[voxcpm_rs](https://github.com/madushan1000/voxcpm_rs)**                  | Rust re-implementation                                                                           |
| **[ComfyUI-VoxCPM](https://github.com/wildminder/ComfyUI-VoxCPM)**          | ComfyUI node-based workflows                                                                     |
| **[ComfyUI_RH_VoxCPM](https://github.com/HM-RunningHub/ComfyUI_RH_VoxCPM)** | Feature-complete ComfyUI workflow for VoxCPM 2 with multi-speaker generation, LoRA, and auto-ASR |
| **[ComfyUI-VoxCPMTTS](https://github.com/1038lab/ComfyUI-VoxCPMTTS)**       | ComfyUI TTS extension                                                                            |
| **[TTS WebUI](https://github.com/rsxdalv/tts_webui_extension.vox_cpm)**     | Browser-based TTS extension                                                                      |
| **[Video Dubbing Translator](https://github.com/kadirb4rut/video-dubbing-translator)** | Local-first video dubbing: vocal separation → WhisperX ASR → translation → VoxCPM2 voice cloning → lip-sync |


> See the full [Ecosystem](https://voxcpm.readthedocs.io/en/latest/) in the docs. Community projects are not officially maintained by OpenBMB. Built something cool? [Open an issue or PR](https://github.com/OpenBMB/VoxCPM/issues) to add it!

---

## ⚠️ Risks and Limitations

- **Potential for Misuse:** VoxCPM's voice cloning can generate highly realistic synthetic speech. It is **strictly forbidden** to use VoxCPM for impersonation, fraud, or disinformation. We strongly recommend clearly marking any AI-generated content.
- **Controllable Generation Stability:** Voice Design and Controllable Voice Cloning results can vary between runs — you may try to generate 1~3 times to obtain the desired voice or style. We are actively working on improving controllability consistency.
- **Language Coverage:** VoxCPM2 officially supports 30 languages. For languages not on the list, you are welcome to test directly or try fine-tuning on your own data. We plan to expand language coverage in future releases.
- **Usage:** This model is released under the Apache-2.0 license. For production deployments, we recommend conducting thorough testing and safety evaluation tailored to your use case.

---

## 📖 Citation

If you find VoxCPM helpful, please consider citing our work and starring ⭐ the repository!

```bib
@article{zhou2026voxcpm2,
  title   = {VoxCPM2 Technical Report},
  author  = {Zhou, Yixuan  and Zeng, Guoyang and Liu, Xin and Li, Xiang and Yu, Renjie and Gui, Jiancheng and Wu, Jiaheng and Wang, Ziyang and Shen, Xudong and Ye, Runchuan  and Zhang, Zhisheng and Zhou, Jiuyang and Bai, Bingsong and Sun, Weiyue and Deng, Mengyuan and Shi, Qundong and Wu, Zhiyong and Liu, Zhiyuan},
  journal = {arXiv preprint arXiv:2606.06928},
  year    = {2026},
}

@article{zhou2025voxcpm,
  title = {Voxcpm: Tokenizer-free TTS for context-aware speech generation and true-to-life voice cloning},
  author = {Zhou, Yixuan and Zeng, Guoyang and Liu, Xin and Li, Xiang and Yu, Renjie and Wang, Ziyang and Ye, Runchuan and Sun, Weiyue and Gui, Jiancheng and Li, Kehan and Wu, Zhiyong and Liu, Zhiyuan},
  journal = {arXiv preprint arXiv:2509.24650},
  year = {2025}
}
```

## 📄 License

VoxCPM model weights and code are open-sourced under the [Apache-2.0](LICENSE) license.

## 🙏 Acknowledgments

- [DiTAR](https://arxiv.org/abs/2502.03930) for the diffusion autoregressive backbone
- [MiniCPM-4](https://github.com/OpenBMB/MiniCPM) for the language model foundation
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) for the Flow Matching-based LocDiT implementation
- [DAC](https://github.com/descriptinc/descript-audio-codec) for the Audio VAE backbone
- Our community users for trying VoxCPM, reporting issues, sharing ideas, and contributing—your support helps the project keep getting better

## Institutions

[ModelBest](https://modelbest.cn/)     [THUHCSI](https://github.com/thuhcsi)

## ⭐ Star History

[Star History Chart](https://star-history.com/#OpenBMB/VoxCPM&Date)