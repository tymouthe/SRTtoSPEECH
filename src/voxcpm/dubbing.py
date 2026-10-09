"""Multi-speaker video dubbing on top of VoxCPM.

Pipeline:
    video (+ optional SRT)
      -> extract original audio (ffmpeg); optionally split it into vocals / instrumental (Hybrid Demucs)
      -> parse / auto-transcribe subtitles, detect ``[Speaker]`` / ``Speaker:`` tags
      -> analyse each line of the clean vocals: gender (pitch), emotion (SenseVoice) and prosody
      -> build one voice profile per character; tones are described relative to that character's own
         baseline and every character keeps one reference voice and one seed, so it stays consistent
      -> synthesise every subtitle line with VoxCPM, then denoise and loudness-level it
      -> fit each clip to its subtitle slot (trim silence, time-stretch, trim overflow)
      -> replace the original audio with the generated voices (muted by default) and mux back into the video

Only ``ffmpeg``/``ffprobe`` on PATH plus the package's existing dependencies are required. Vocal removal
downloads the torchaudio Hybrid Demucs weights on first use; emotion detection uses FunASR SenseVoiceSmall.
Auto-transcription (when no SRT is given) needs the optional ``stable-ts`` extra.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import zlib
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

logger = logging.getLogger(__name__)

VOICE_MODES = ("clone_first", "clone_line", "clone_speaker", "design")
BACKGROUND_MODES = ("mute", "instrumental", "duck", "keep")

ANALYSIS_SR = 16_000
SEPARATION_SR = 44_100
MALE_FEMALE_F0_THRESHOLD = 165.0  # Hz; typical adult male F0 ~85-155, female ~165-255
MIN_REFERENCE_SECONDS = 1.0
# Median F0 range (Hz) a designed voice must land in to sound like an adult of that gender (not a child).
ADULT_PITCH_RANGE = {"male": (85.0, 160.0), "female": (165.0, 255.0)}


# -----------------------------
# Data model
# -----------------------------


@dataclass
class SubtitleLine:
    index: int
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    tone: str = ""
    gender: str = "unknown"
    emotion: str = ""
    features: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def duration(self) -> float:
        return max(self.end - self.start, 0.0)


@dataclass
class SpeakerProfile:
    name: str
    gender: str = "unknown"  # male | female | unknown
    # clone_first: the character's first line clones its original audio; every later line clones that
    # generated first line, so the whole character is one consistent voice. See VOICE_MODES.
    voice_mode: str = "clone_line"
    description: str = ""  # voice description used for design mode (and as style hint)
    age: str = "adult"  # adult | kid
    reference_wav: Optional[str] = None  # explicit reference audio for clone_speaker
    reference_text: Optional[str] = None


@dataclass
class DubOptions:
    background: str = "mute"  # only the generated voices; see BACKGROUND_MODES
    duck_db: float = -18.0
    max_speedup: float = 1.35
    cfg_value: float = 2.0
    inference_timesteps: int = 10
    denoise_reference: bool = True
    clean_generated: bool = True
    target_lufs: float = -20.0
    normalize: bool = False
    seed: Optional[int] = None  # base seed; each character gets its own fixed seed derived from it
    embed_subtitles: bool = True
    use_tone_control: bool = True
    # Check each generated line against the character's voice and regenerate when it drifts. A line passes
    # when it is at least ``min_voice_similarity`` close to its own character's reference and at least
    # ``voice_margin`` closer to it than to any other character's (cloned speech, especially across
    # languages, rarely scores as high as the original recording, so the margin matters most).
    verify_voice: bool = True
    min_voice_similarity: float = 0.3
    voice_margin: float = 0.03
    # Keep each character's pitch steady: a try whose pitch is more than ``pitch_tolerance`` semitones from
    # the character's (emotion-adjusted) pitch is retried, and the kept line is nudged back towards it by
    # at most ``max_pitch_correction`` semitones.
    pitch_tolerance: float = 3.0
    max_pitch_correction: float = 2.0  # larger shifts also move the formants and change the voice
    # Retry a line (new seed each time) until it is consistent with its speaker, up to ``max_attempts``
    # tries; a speaker's first line, which all their other lines copy, gets ``first_line_attempts``.
    max_attempts: int = 8
    first_line_attempts: int = 12
    # Lines that clone a speaker's *generated* first line must be at least this close to it, and to the
    # speaker's other accepted lines (cloned speech of the same voice normally scores 0.6-0.85).
    min_consistency: float = 0.55
    # Short lines (slot up to ``short_line_seconds``, e.g. a one-word reply) get only their emotion as a style
    # hint: with a long voice prompt and one word of text VoxCPM tends to babble on for seconds (that babble,
    # cut to fit the slot, is what sounds like someone else). Their voice check uses ``short_line_consistency``
    # since a voice embedding of one word is noisy.
    short_line_seconds: float = 1.6
    short_line_consistency: float = 0.45
    # SRT → Speech: pad a line file shorter than its subtitle slot with silence (after the speech) to the
    # slot's length. Off by default: every line file keeps its natural length.
    pad_to_slot: bool = False
    # SRT → Speech: when a line is finalized, cut the silence before and after the speech and shorten every
    # pause inside it that is longer than ``max_pause`` seconds to ``max_pause``.
    remove_silence: bool = True
    max_pause: float = 0.2
    # Every line of a designed voice clones its first, designed reading, so that reading gets more tries.
    design_attempts: int = 5


@dataclass
class DubResult:
    video_path: Optional[str]
    audio_path: str
    srt_path: str
    speaker_srt_path: str
    report: list[dict] = field(default_factory=list)
    instrumental_path: Optional[str] = None


# -----------------------------
# SRT
# -----------------------------

_TIME_RE = re.compile(
    r"(\d+):(\d{1,2}):(\d{1,2})[,.](\d{1,3})\s*-->\s*(\d+):(\d{1,2}):(\d{1,2})[,.](\d{1,3})"
)
_TAG_RE = re.compile(r"<[^>]+>|\{\\[^}]*\}")
# Up to 80 characters: a voice tag like "[Chen Dayong|male|adult|indifferent]" is longer than a bare name.
_BRACKET_SPEAKER_RE = re.compile(r"^\s*[\[【]\s*([^\]】]{1,80}?)\s*[\]】]\s*[:：]?\s*(.*)$", re.S)
_COLON_SPEAKER_RE = re.compile(r"^\s*([^\s:：\-][^:：\n]{0,23}?)\s*[:：]\s+(.+)$", re.S)


def parse_timestamp(value) -> float:
    """Accept seconds (``12.5``) or SRT-style ``00:00:12,500`` / ``00:12.5``."""
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", ".")
    seconds = 0.0
    for part in text.split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


def _ts_to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000.0


def seconds_to_srt_time(t: float) -> str:
    total_ms = max(int(round(t * 1000)), 0)
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _looks_like_name(candidate: str) -> bool:
    words = candidate.split()
    if not words or len(words) > 3 or candidate.strip().isdigit():
        return False
    # Latin-script names must be capitalised ("Mary Jane", "Dr. Who"); others (e.g. "老王") pass.
    return all(not w[0].isalpha() or not w[0].isascii() or w[0].isupper() for w in words)


def split_speaker_tag(text: str) -> tuple[Optional[str], str]:
    """Split ``[Alice] hi`` / ``Alice: hi`` into ``("Alice", "hi")``.

    A ``Name:`` prefix is only treated as a speaker when it is short (<= 3 capitalised words) so
    ordinary sentences containing a colon are left untouched.
    """
    m = _BRACKET_SPEAKER_RE.match(text)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m = _COLON_SPEAKER_RE.match(text)
    if m and _looks_like_name(m.group(1)):
        return m.group(1).strip(), m.group(2).strip()
    return None, text.strip()


def parse_srt(content: str) -> list[SubtitleLine]:
    content = content.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    lines: list[SubtitleLine] = []
    for block in re.split(r"\n\s*\n", content.strip()):
        rows = [r for r in block.split("\n") if r.strip()]
        time_idx = next((i for i, r in enumerate(rows) if _TIME_RE.search(r)), None)
        if time_idx is None:
            continue
        g = _TIME_RE.search(rows[time_idx]).groups()
        start, end = _ts_to_seconds(*g[:4]), _ts_to_seconds(*g[4:])
        raw = " ".join(_TAG_RE.sub("", r).strip() for r in rows[time_idx + 1 :]).strip()
        raw = re.sub(r"^-\s*", "", raw)
        if not raw:
            continue
        speaker, text = split_speaker_tag(raw)
        if not text:
            continue
        lines.append(SubtitleLine(index=len(lines) + 1, start=start, end=end, text=text, speaker=speaker))
    lines.sort(key=lambda x: x.start)
    for i, line in enumerate(lines, 1):
        line.index = i
    return lines


def load_srt(path: str | os.PathLike) -> list[SubtitleLine]:
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "utf-16", "gb18030", "latin-1"):
        try:
            return parse_srt(raw.decode(enc))
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode subtitle file: {path}")


def format_srt(lines: Iterable[SubtitleLine], with_speaker: bool = False) -> str:
    out = []
    for i, line in enumerate(lines, 1):
        text = f"[{line.speaker}] {line.text}" if with_speaker and line.speaker else line.text
        out.append(f"{i}\n{seconds_to_srt_time(line.start)} --> {seconds_to_srt_time(line.end)}\n{text}\n")
    return "\n".join(out)


# -----------------------------
# ffmpeg helpers
# -----------------------------


def _require_binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"`{name}` was not found on PATH. Install ffmpeg to use video dubbing.")
    return path


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr[-2000:]}")


def probe_duration(path: str) -> float:
    proc = subprocess.run(
        [_require_binary("ffprobe"), "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        return float(json.loads(proc.stdout)["format"]["duration"])
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read duration of {path}: {proc.stderr}") from exc


def has_video_stream(path: str) -> bool:
    proc = subprocess.run(
        [_require_binary("ffprobe"), "-v", "error", "-select_streams", "v", "-show_entries", "stream=index", "-of", "csv=p=0", path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return bool(proc.stdout.strip())


def extract_audio(media_path: str, out_wav: str, sample_rate: int) -> str:
    _run([_require_binary("ffmpeg"), "-y", "-v", "error", "-i", media_path, "-vn", "-ac", "1", "-ar", str(sample_rate), out_wav])
    return out_wav


def mux_video(
    video_path: str,
    audio_path: str,
    out_path: str,
    srt_path: Optional[str] = None,
) -> str:
    cmd = [_require_binary("ffmpeg"), "-y", "-v", "error", "-i", video_path, "-i", audio_path]
    if srt_path:
        cmd += ["-i", srt_path]
    cmd += ["-map", "0:v:0", "-map", "1:a:0"]
    if srt_path:
        sub_codec = "srt" if out_path.lower().endswith(".mkv") else "mov_text"
        cmd += ["-map", "2:s:0", "-c:s", sub_codec, "-metadata:s:s:0", "title=Dub"]
    cmd += ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k", out_path]
    _run(cmd)
    return out_path


# -----------------------------
# Vocal removal (Hybrid Demucs)
# -----------------------------


def _separate_chunks(model, mix, sr: int, segment: float, overlap: float, device) -> "torch.Tensor":
    """Overlap-add source separation of ``mix`` (batch, channels, time) in ``segment``-second chunks."""
    import torch
    from torchaudio.transforms import Fade

    batch, channels, length = mix.shape
    chunk_len = int(sr * segment * (1 + overlap))
    overlap_frames = int(overlap * sr)
    fade = Fade(fade_in_len=0, fade_out_len=overlap_frames, fade_shape="linear")
    final = torch.zeros(batch, len(model.sources), channels, length)
    start, end = 0, chunk_len
    while start < length - overlap_frames:
        chunk = mix[:, :, start:end].to(device)
        with torch.no_grad():
            out = fade(model(chunk)).cpu()
        final[:, :, :, start : start + out.shape[-1]] += out
        if start == 0:
            fade.fade_in_len = overlap_frames
            start += chunk_len - overlap_frames
        else:
            start += chunk_len
        end += chunk_len
        if end >= length:
            fade.fade_out_len = 0
    return final


def separate_vocals(
    media_path: str,
    vocals_out: str,
    instrumental_out: str,
    device: Optional[str] = None,
    segment: float = 10.0,
    overlap: float = 0.1,
) -> tuple[str, str]:
    """Split a soundtrack into ``vocals`` and ``instrumental`` (music + effects) stereo wavs."""
    import soundfile as sf
    import torch

    mix_path = str(Path(instrumental_out).with_name("original_44k_stereo.wav"))
    _run([_require_binary("ffmpeg"), "-y", "-v", "error", "-i", media_path, "-vn", "-ac", "2", "-ar", str(SEPARATION_SR), mix_path])
    audio, sr = sf.read(mix_path, dtype="float32", always_2d=True)

    from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS

    try:
        model = HDEMUCS_HIGH_MUSDB_PLUS.get_model().eval()
    except Exception as exc:
        raise RuntimeError(
            "Could not load the Hybrid Demucs weights used for vocal removal "
            "(downloaded once from download.pytorch.org into the torch hub cache). "
            f"Check your network, or disable vocal removal. Cause: {exc}"
        ) from exc

    mix = torch.from_numpy(np.ascontiguousarray(audio.T)).unsqueeze(0)
    ref = mix.mean(1)
    mean, std = ref.mean(), ref.std() + 1e-8
    mix = (mix - mean) / std

    sources = None
    for dev in dict.fromkeys([device or "cpu", "cpu"]):
        try:
            sources = _separate_chunks(model.to(dev), mix, sr, segment, overlap, torch.device(dev))
            break
        except (RuntimeError, NotImplementedError) as exc:
            if dev == "cpu":
                raise
            logger.warning("Vocal separation failed on %s (%s); retrying on CPU", dev, exc)
    sources = sources[0] * std
    vocal_idx = list(model.sources).index("vocals")
    vocals = sources[vocal_idx]
    instrumental = sources.sum(0) - vocals + mean
    sf.write(vocals_out, (vocals + mean).T.numpy(), sr)
    sf.write(instrumental_out, instrumental.T.numpy(), sr)
    return vocals_out, instrumental_out


# -----------------------------
# Emotion detection (SenseVoice)
# -----------------------------

_EMOTION_HINTS = {
    "happy": "happy and cheerful",
    "sad": "sad and downcast",
    "angry": "angry and intense",
    "fearful": "fearful and nervous",
    "disgusted": "disgusted",
    "surprised": "surprised",
}
_EMOTION_TAG_RE = re.compile(r"<\|(HAPPY|SAD|ANGRY|NEUTRAL|FEARFUL|DISGUSTED|SURPRISED)\|>")


def parse_sensevoice_emotion(raw_text: str) -> str:
    """``"<|en|><|HAPPY|><|Speech|><|woitn|>hi"`` -> ``"happy"``; unknown -> ``"neutral"``."""
    m = _EMOTION_TAG_RE.search(raw_text or "")
    return m.group(1).lower() if m else "neutral"


def load_emotion_model(device: str = "cpu"):
    from funasr import AutoModel

    return AutoModel(model="iic/SenseVoiceSmall", disable_update=True, device=device, log_level="ERROR")


def detect_emotions(paths: list[str], model) -> list[str]:
    if not paths:
        return []
    results = model.generate(input=paths, language="auto", use_itn=False)
    by_key = {str(r.get("key")): parse_sensevoice_emotion(r.get("text", "")) for r in results}
    ordered = [parse_sensevoice_emotion(r.get("text", "")) for r in results]
    return [by_key.get(Path(p).stem, ordered[i] if i < len(ordered) else "neutral") for i, p in enumerate(paths)]


# -----------------------------
# Speaker identification (CAM++ voice embeddings)
# -----------------------------

SPEAKER_MODEL_ID = "iic/speech_campplus_sv_zh-cn_16k-common"
MIN_EMBED_SECONDS = 0.6


def load_speaker_model(device: str = "cpu"):
    from funasr import AutoModel

    return AutoModel(model=SPEAKER_MODEL_ID, disable_update=True, device=device, log_level="ERROR", disable_pbar=True)


def speaker_embedding(model, wav16: np.ndarray) -> Optional[np.ndarray]:
    """L2-normalised voice embedding of a 16 kHz clip, or ``None`` if it is too short to be reliable."""
    if len(wav16) < MIN_EMBED_SECONDS * ANALYSIS_SR:
        return None
    result = model.generate(input=np.ascontiguousarray(wav16, dtype=np.float32))
    emb = np.asarray(result[0]["spk_embedding"], dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(emb))
    return emb / norm if norm > 0 else None


def _centroid(vectors: list[np.ndarray]) -> np.ndarray:
    c = np.mean(vectors, axis=0)
    return c / (np.linalg.norm(c) + 1e-9)


def cluster_speakers(
    embeddings: list[Optional[np.ndarray]],
    num_speakers: Optional[int] = None,
    threshold: float = 0.55,
    merge_similarity: float = 0.35,
) -> list[int]:
    """Group lines by voice; returns a cluster id per line, numbered by first appearance.

    Agglomerative clustering on cosine distance. With ``num_speakers`` the number of characters is fixed,
    otherwise clusters merge until voices are less similar than ``1 - threshold``; single-line clusters are
    then folded into the closest real character. Lines too short to embed join the nearest line in time.
    """
    known = [i for i, e in enumerate(embeddings) if e is not None]
    labels = [-1] * len(embeddings)
    if not known:
        return [0] * len(embeddings)
    if len(known) == 1:
        labels[known[0]] = 0
    else:
        from scipy.cluster.hierarchy import fcluster, linkage

        matrix = np.stack([embeddings[i] for i in known])
        tree = linkage(matrix, method="average", metric="cosine")
        if num_speakers:
            found = fcluster(tree, t=num_speakers, criterion="maxclust")
        else:
            found = fcluster(tree, t=threshold, criterion="distance")
        for i, label in zip(known, found):
            labels[i] = int(label)
        if not num_speakers and len(known) >= 6:
            counts = Counter(labels[i] for i in known)
            centroids = {
                c: _centroid([embeddings[i] for i in known if labels[i] == c]) for c in counts if counts[c] > 1
            }
            for i in known:
                if counts[labels[i]] == 1 and centroids:
                    best = max(centroids, key=lambda c: float(embeddings[i] @ centroids[c]))
                    if float(embeddings[i] @ centroids[best]) >= merge_similarity:
                        counts[labels[i]] -= 1
                        labels[i] = best
                        counts[best] += 1
    for i in range(len(labels)):
        if labels[i] == -1:
            nearest = min(known, key=lambda j: abs(j - i))
            labels[i] = labels[nearest]
    order: dict[int, int] = {}
    for label in labels:
        order.setdefault(label, len(order))
    return [order[label] for label in labels]


def score_voice_consistency(lines: list[SubtitleLine]) -> None:
    """Store how close each line's voice is to its character's average voice (``features["voice_match"]``)."""
    for name in dict.fromkeys(l.speaker for l in lines):
        own = [l for l in lines if l.speaker == name and l.features.get("embedding") is not None]
        if not own:
            continue
        for line in own:
            # Compare with the character's *other* lines so a mismatched line cannot vouch for itself.
            others = [l.features["embedding"] for l in own if l is not line]
            line.features["voice_match"] = float(line.features["embedding"] @ _centroid(others)) if others else 1.0


# -----------------------------
# Audio analysis: gender & tone
# -----------------------------


def _segment(audio: np.ndarray, sr: int, start: float, end: float, pad: float = 0.0) -> np.ndarray:
    a = max(int((start - pad) * sr), 0)
    b = min(int((end + pad) * sr), len(audio))
    return audio[a:b] if b > a else np.zeros(0, dtype=audio.dtype)


def estimate_f0(segment: np.ndarray, sr: int) -> np.ndarray:
    """Return voiced F0 values (Hz) of a mono segment; empty if unvoiced/too short."""
    if len(segment) < int(0.2 * sr):
        return np.zeros(0)
    import librosa

    f0, voiced, _ = librosa.pyin(segment.astype(np.float32), fmin=65.0, fmax=450.0, sr=sr, frame_length=1024)
    f0 = f0[voiced & ~np.isnan(f0)]
    return f0


def classify_gender(f0: np.ndarray) -> str:
    if len(f0) < 5:
        return "unknown"
    return "male" if float(np.median(f0)) < MALE_FEMALE_F0_THRESHOLD else "female"


_CJK_RE = re.compile(r"[぀-ヿ㐀-鿿가-힯]")


def estimate_speech_seconds(text: str) -> float:
    """Rough time needed to say ``text`` at a normal pace, for any script: CJK characters (~4.5/s), Latin or
    Cyrillic words (~2.7/s), letters of other scripts such as Khmer or Thai (~8 base letters/s; vowel signs,
    tone marks and Khmer subscript consonants are not counted) and numbers (~0.4 s each)."""
    import unicodedata

    cjk = len(_CJK_RE.findall(text))
    words = len(re.findall(r"[A-Za-zÀ-ɏЀ-ӿ']+", text))
    letters = sum(
        1
        for ch in text
        if unicodedata.category(ch) == "Lo" and not _CJK_RE.match(ch)
    ) - text.count("\u17d2")  # a consonant after Khmer COENG is a subscript, not a new syllable
    numbers = len(re.findall(r"\d+", text))
    return cjk / 4.5 + words / 2.7 + max(letters, 0) / 8.0 + numbers * 0.4


def _speech_units(text: str) -> int:
    cjk = len(re.findall(r"[぀-ヿ㐀-鿿가-힯]", text))
    words = len(re.findall(r"[A-Za-z0-9À-ɏЀ-ӿ']+", text))
    # One CJK character is roughly half an English word in speaking time.
    return words + cjk // 2 if cjk else words


def measure_line(segment: np.ndarray, sr: int) -> dict:
    f0 = estimate_f0(segment, sr)
    rms = float(np.sqrt(np.mean(segment**2))) if len(segment) else 0.0
    spread = float(np.std(12 * np.log2(f0 / np.median(f0)))) if len(f0) >= 5 else None
    return {
        "f0": f0,
        "f0_median": float(np.median(f0)) if len(f0) >= 5 else None,
        "pitch_spread": spread,
        "rms_db": 20 * np.log10(rms + 1e-8),
    }


def analyze_lines(lines: list[SubtitleLine], audio: np.ndarray, sr: int) -> list[SubtitleLine]:
    """Measure pitch/energy of each line (from clean vocals if available) and set ``gender`` (in place)."""
    for line in lines:
        line.features = measure_line(_segment(audio, sr, line.start, line.end), sr)
        units = _speech_units(line.text)
        line.features["rate"] = units / line.duration if line.duration > 0.3 and units else None
        line.gender = classify_gender(line.features["f0"])
    return lines


def _median(values) -> Optional[float]:
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else None


def describe_tone(line: SubtitleLine, baseline: dict) -> str:
    """Prosody hint for one line: how it differs from the speaker's *own* usual delivery.

    Comparing with the speaker's baseline (instead of the whole video) keeps a naturally loud or fast
    character from being labelled "loud"/"fast" on every line, which keeps their voice consistent.
    The detected emotion is kept separately in ``line.emotion`` (see :func:`line_style`).
    """
    parts = []
    f = line.features or {}
    if baseline.get("lines", 0) < 2:
        return ", ".join(parts)
    if f.get("rms_db") is not None and baseline.get("rms_db") is not None and f["rms_db"] > -60:
        delta = f["rms_db"] - baseline["rms_db"]
        if delta > 5:
            parts.append("louder and more forceful")
        elif delta < -6:
            parts.append("softer and quieter")
    if f.get("pitch_spread") is not None and baseline.get("pitch_spread") is not None:
        delta = f["pitch_spread"] - baseline["pitch_spread"]
        if delta > 2.0:
            parts.append("more animated")
        elif delta < -1.5 and baseline["pitch_spread"] > 2.0:
            parts.append("flatter and more subdued")
    if f.get("rate") and baseline.get("rate"):
        ratio = f["rate"] / baseline["rate"]
        if ratio > 1.3:
            parts.append("speaking faster")
        elif ratio < 0.75:
            parts.append("speaking slower")
    return ", ".join(parts)


def emotion_hint(emotion: str) -> str:
    """Known SenseVoice labels map to a phrase; any other non-neutral text is used as written."""
    emotion = (emotion or "").strip()
    if emotion.lower() in ("", "neutral", "none"):
        return ""
    return _EMOTION_HINTS.get(emotion.lower(), emotion)


def line_style(line: SubtitleLine) -> str:
    return ", ".join(p for p in (emotion_hint(line.emotion), line.tone) if p)


def speaker_baseline(lines: list[SubtitleLine]) -> dict:
    voiced = [l.features for l in lines if l.features and l.features.get("rms_db", -100) > -60]
    return {
        "lines": len(voiced),
        "rms_db": _median(f.get("rms_db") for f in voiced),
        "pitch_spread": _median(f.get("pitch_spread") for f in voiced),
        "f0_median": _median(f.get("f0_median") for f in voiced),
        "rate": _median(f.get("rate") for f in voiced),
    }


def assign_tones(lines: list[SubtitleLine]) -> list[SubtitleLine]:
    for name in dict.fromkeys(l.speaker for l in lines):
        own = [l for l in lines if l.speaker == name]
        baseline = speaker_baseline(own)
        for line in own:
            line.tone = describe_tone(line, baseline)
    return lines


def speaking_style(gender: str, lines: list[SubtitleLine], reference: Optional[dict] = None) -> list[str]:
    """How a character usually speaks: pitch register and delivery, plus (given ``reference``, the
    baseline of everyone in the video) whether they speak faster/slower or louder/softer than the others."""
    parts = []
    baseline = speaker_baseline(lines)
    f0 = baseline.get("f0_median")
    if f0 is not None:
        if gender == "male":
            parts.append("deep pitch" if f0 < 100 else "light, higher pitch" if f0 > 140 else "medium pitch")
        elif gender == "female":
            parts.append("low pitch" if f0 < 185 else "high, bright pitch" if f0 > 240 else "medium pitch")
    spread = baseline.get("pitch_spread")
    if spread is not None:
        if spread > 3.5:
            parts.append("expressive delivery")
        elif spread < 1.8:
            parts.append("steady, calm delivery")
    if reference:
        if baseline.get("rate") and reference.get("rate"):
            ratio = baseline["rate"] / reference["rate"]
            if ratio > 1.2:
                parts.append("fast-paced")
            elif ratio < 0.83:
                parts.append("slow, deliberate pace")
        if baseline.get("rms_db") is not None and reference.get("rms_db") is not None:
            delta = baseline["rms_db"] - reference["rms_db"]
            if delta > 4:
                parts.append("strong, projecting voice")
            elif delta < -4:
                parts.append("soft-spoken")
    return parts


def describe_speaker(gender: str, lines: list[SubtitleLine], cast: Optional[list[SubtitleLine]] = None) -> str:
    """A stable voice prompt for a character, scanned from their own original lines: gender, pitch
    register, usual delivery, (compared with the rest of the ``cast``) pace and loudness, and mood."""
    reference = speaker_baseline(cast) if cast and len({l.speaker for l in cast}) > 1 else None
    parts = [default_description(gender)] + speaking_style(gender, lines, reference)
    moods = Counter(l.emotion for l in lines if l.emotion in _EMOTION_HINTS)
    if moods:
        mood, count = moods.most_common(1)[0]
        if count * 2 > len(lines):
            parts.append(f"generally {_EMOTION_HINTS[mood].split(' and ')[0]}")
    return ", ".join(parts)


def default_description(gender: str) -> str:
    return {"male": "Adult male voice", "female": "Adult female voice"}.get(gender, "Natural adult voice")


_GENERIC_SPEAKERS = {"male": "Male", "female": "Female"}
ADULT_VOICE_DESCRIPTIONS = {
    "male": "Adult male voice, mature and warm, medium-low pitch",
    "female": "Adult female voice, mature and warm, medium pitch",
}


def collapse_to_two_voices(
    lines: list[SubtitleLine],
    profiles: Optional[dict[str, SpeakerProfile]] = None,
    speaker_style: bool = True,
    voice_mode: str = "clone_first",
) -> dict[str, SpeakerProfile]:
    """Voice the whole video with just two voices: one adult male (``"Male"``) and one adult female
    (``"Female"``).

    Every line goes to the voice of its character's gender (or its own detected gender, or the video's
    most common gender when that is unknown). By default (``clone_first``) each voice copies the first
    line of that gender spoken by an adult in the original, and every later line clones that generated
    first line; ``voice_mode="design"`` instead designs both voices from a fixed adult description.
    Per-line emotion and tone are kept and still passed as style hints.

    With ``speaker_style`` the voice keeps its timbre but each line is *styled* after its original
    character: their pitch register, delivery, pace and loudness (see :func:`speaking_style`) are put in
    front of the line's tone, so two characters sharing the male voice still speak differently.
    Updates ``lines`` in place and returns the two profiles.
    """
    profiles = profiles or {}
    known = Counter(l.gender for l in lines if l.gender in _GENERIC_SPEAKERS)
    fallback = known.most_common(1)[0][0] if known else "male"
    if speaker_style:
        everyone = speaker_baseline(lines)
        for name in dict.fromkeys(l.speaker for l in lines):
            own = [l for l in lines if l.speaker == name]
            profile = profiles.get(name)
            genders = Counter(l.gender for l in own if l.gender in _GENERIC_SPEAKERS)
            gender = profile.gender if profile is not None else genders.most_common(1)[0][0] if genders else "unknown"
            style = ", ".join(speaking_style(gender, own, everyone))
            for line in own:
                line.tone = ", ".join(p for p in (style, line.tone) if p)
    for line in lines:
        profile = profiles.get(line.speaker)
        if profile is not None and profile.gender in _GENERIC_SPEAKERS:
            gender = profile.gender
        else:
            gender = line.gender if line.gender in _GENERIC_SPEAKERS else fallback
        line.speaker, line.gender = _GENERIC_SPEAKERS[gender], gender
    return {
        _GENERIC_SPEAKERS[g]: SpeakerProfile(
            name=_GENERIC_SPEAKERS[g], gender=g, voice_mode=voice_mode, description=ADULT_VOICE_DESCRIPTIONS[g]
        )
        for g in dict.fromkeys(l.gender for l in lines)
    }


def build_speaker_profiles(lines: list[SubtitleLine], default_mode: Optional[str] = None) -> dict[str, SpeakerProfile]:
    """Assign speakers to untagged lines (by gender) and derive one profile per speaker."""
    for line in lines:
        if not line.speaker:
            line.speaker = _GENERIC_SPEAKERS.get(line.gender, "Speaker")

    if default_mode is None:
        # One consistent voice per speaker: the first line copies the original, later lines clone it.
        default_mode = "clone_first"

    profiles: dict[str, SpeakerProfile] = {}
    for name in dict.fromkeys(line.speaker for line in lines):
        genders = Counter(l.gender for l in lines if l.speaker == name and l.gender != "unknown")
        gender = genders.most_common(1)[0][0] if genders else "unknown"
        own = [l for l in lines if l.speaker == name]
        profiles[name] = SpeakerProfile(
            name=name, gender=gender, voice_mode=default_mode, description=describe_speaker(gender, own, lines)
        )
    return profiles


def apply_speaker_overrides(profiles: dict[str, SpeakerProfile], overrides: dict) -> dict[str, SpeakerProfile]:
    """Merge a ``{"Alice": {"gender": "female", "voice_mode": "design", ...}}`` mapping into profiles."""
    for name, cfg in (overrides or {}).items():
        profile = profiles.setdefault(name, SpeakerProfile(name=name))
        for key, value in cfg.items():
            key = {"mode": "voice_mode", "reference": "reference_wav", "voice": "description"}.get(key, key)
            if hasattr(profile, key):
                setattr(profile, key, value)
        if "description" not in cfg and "voice" not in cfg and "gender" in cfg:
            profile.description = default_description(profile.gender)
    for profile in profiles.values():
        if profile.voice_mode not in VOICE_MODES:
            raise ValueError(f"Unknown voice mode {profile.voice_mode!r} for {profile.name}; use one of {VOICE_MODES}")
    return profiles


# -----------------------------
# Cleaning generated voices
# -----------------------------


def spectral_gate(
    wav: np.ndarray, sr: int, n_std: float = 1.5, threshold_db: float = 6.0, floor_db: float = -20.0
) -> np.ndarray:
    """Stationary spectral-gating denoiser.

    The noise profile is taken from the quietest 10% of frames (pauses and lead-in/out of a TTS clip);
    time-frequency bins that do not rise ``n_std`` standard deviations (at least ``threshold_db``) above
    it are attenuated to ``floor_db``.
    Clips without a clear noise floor (no pauses) are returned unchanged rather than over-suppressed.
    """
    import librosa
    from scipy.ndimage import uniform_filter

    n_fft = 2048 if sr >= 32_000 else 1024
    hop = n_fft // 4
    if len(wav) < n_fft * 4:
        return wav
    spec = librosa.stft(wav.astype(np.float32), n_fft=n_fft, hop_length=hop)
    mag = np.abs(spec)
    power = (mag**2).sum(axis=0)
    quiet = power <= np.percentile(power, 10)
    noise_power = float(power[quiet].mean()) + 1e-12
    if 10 * np.log10(float(np.median(power)) / noise_power + 1e-12) < 12:
        return wav
    mag_db = 20 * np.log10(mag + 1e-10)
    noise_db = mag_db[:, quiet]
    threshold = noise_db.mean(axis=1, keepdims=True) + np.maximum(n_std * noise_db.std(axis=1, keepdims=True), threshold_db)
    # Soft 6 dB ramp + wide time smoothing instead of a hard on/off mask: a hard mask flickers as a voice
    # fades out, which is heard as chirpy "musical noise" at the end of words.
    mask = np.clip((mag_db - threshold) / 6.0 + 0.5, 0.0, 1.0).astype(np.float32)
    mask = uniform_filter(mask, size=(5, 9))
    floor = 10 ** (floor_db / 20)
    gain = floor + (1 - floor) * mask
    out = librosa.istft(spec * gain, hop_length=hop, length=len(wav))
    return out.astype(np.float32)


def normalize_loudness(wav: np.ndarray, sr: int, target_lufs: float = -20.0, peak: float = 0.97) -> np.ndarray:
    loudness = None
    if len(wav) >= int(0.4 * sr):
        try:
            import torch
            import torchaudio

            loudness = float(torchaudio.functional.loudness(torch.from_numpy(wav).unsqueeze(0), sr))
        except Exception:
            loudness = None
    if loudness is None or not np.isfinite(loudness):
        rms = float(np.sqrt(np.mean(wav**2))) if len(wav) else 0.0
        if rms < 1e-6:
            return wav
        loudness = 20 * np.log10(rms) - 0.7
    wav = wav * 10 ** ((target_lufs - loudness) / 20)
    max_abs = float(np.max(np.abs(wav))) if len(wav) else 0.0
    if max_abs > peak:
        wav = wav * (peak / max_abs)
    return wav.astype(np.float32)


def clean_voice(wav: np.ndarray, sr: int, target_lufs: float = -20.0, highpass_hz: float = 70.0) -> np.ndarray:
    """Remove rumble/hum and background hiss from a generated line, then level its loudness."""
    from scipy.signal import butter, sosfiltfilt

    wav = np.asarray(wav, dtype=np.float32)
    if len(wav) < int(0.1 * sr):
        return wav
    sos = butter(4, highpass_hz, btype="highpass", fs=sr, output="sos")
    wav = sosfiltfilt(sos, wav).astype(np.float32)
    wav = spectral_gate(wav, sr)
    return normalize_loudness(wav, sr, target_lufs)


# -----------------------------
# Timing: fit generated clips into subtitle slots
# -----------------------------


def trim_silence(wav: np.ndarray, top_db: float = 40.0) -> np.ndarray:
    if len(wav) == 0:
        return wav
    import librosa

    trimmed, _ = librosa.effects.trim(wav, top_db=top_db)
    return trimmed if len(trimmed) else wav


def _fade_out(wav: np.ndarray, sr: int, seconds: float = 0.06) -> np.ndarray:
    n = min(len(wav), int(seconds * sr))
    if n > 0:
        wav = wav.copy()
        wav[-n:] *= (0.5 + 0.5 * np.cos(np.linspace(0.0, np.pi, n))).astype(wav.dtype)
    return wav


def _fade_in(wav: np.ndarray, sr: int, seconds: float = 0.01) -> np.ndarray:
    n = min(len(wav), int(seconds * sr))
    if n > 0:
        wav = wav.copy()
        wav[:n] *= np.linspace(0.0, 1.0, n, dtype=wav.dtype)
    return wav


def remove_silences(
    wav: np.ndarray,
    sr: int,
    max_pause: float = 0.2,
    top_db: float = 45.0,
    lead: float = 0.01,
    tail: float = 0.04,
) -> np.ndarray:
    """Cut the silence before and after the speech in ``wav`` and shorten each pause inside it that is longer
    than ``max_pause`` seconds to ``max_pause`` (pauses are kept, so words do not run together).

    Silence is anything ``top_db`` below the loudest part. ``lead`` / ``tail`` seconds are kept around the speech so
    soft starts and word endings are not clipped; every cut is faded so it never clicks.
    """
    import librosa

    wav = np.asarray(wav, dtype=np.float32)
    if len(wav) < int(0.05 * sr):
        return wav
    frame = max(int(0.02 * sr), 64)
    hop = frame // 4
    spans = librosa.effects.split(wav, top_db=top_db, frame_length=frame, hop_length=hop)
    if not len(spans):
        return wav
    keep_pause = int(max_pause * sr)
    fade = 0.008
    # Stretches to keep, split where a pause is too long; each split keeps half of ``max_pause`` on either side.
    stretches, start = [], max(int(spans[0][0]) - int(lead * sr), 0)
    for (_, end), (next_start, _) in zip(spans[:-1], spans[1:]):
        if next_start - end > keep_pause:
            stretches.append((start, int(end) + keep_pause // 2))
            start = int(next_start) - keep_pause // 2
    stretches.append((start, min(int(spans[-1][1]) + int(tail * sr), len(wav))))
    pieces = [_fade_out(_fade_in(wav[a:b], sr, fade), sr, fade) for a, b in stretches]
    return _fade_out(_fade_in(np.concatenate(pieces), sr), sr, 0.03)


def finalize_edges(
    wav: np.ndarray,
    sr: int,
    floor_db: float = -35.0,
    min_gap: float = 0.15,
    max_artifact: float = 0.6,
) -> np.ndarray:
    """Remove trailing junk after the last word and fade both edges.

    TTS clips often end with a breath, mumble, chirp or hum *after* the last word, separated from it by
    a short pause. The clip is split into sound segments at pauses of at least ``min_gap`` seconds; trailing
    segments that are short (< ``max_artifact`` s) and contain no voiced speech are dropped, as is any
    leading/trailing sound ``floor_db`` below the speech level. Continuous speech is never cut. The result
    gets a 10 ms fade-in and a 60 ms cosine fade-out so it never starts or stops with a click.
    """
    import librosa

    wav = np.asarray(wav, dtype=np.float32)
    frame = max(int(0.02 * sr), 16)
    hop = frame // 2
    if len(wav) < frame * 4:
        return _fade_out(_fade_in(wav, sr), sr)
    rms = librosa.feature.rms(y=wav, frame_length=frame, hop_length=hop)[0]
    level = float(np.percentile(rms, 95))
    active = rms > level * 10 ** (floor_db / 20)
    if not active.any():
        return wav

    # sound segments (in frames), split at pauses >= min_gap
    gap_frames = max(int(min_gap * sr / hop), 1)
    idx = np.where(active)[0]
    segments = [[idx[0], idx[0]]]
    for i in idx[1:]:
        if i - segments[-1][1] > gap_frames:
            segments.append([i, i])
        else:
            segments[-1][1] = i

    y16 = librosa.resample(wav, orig_sr=sr, target_sr=ANALYSIS_SR) if sr != ANALYSIS_SR else wav
    _, voiced, _ = librosa.pyin(y16, fmin=65.0, fmax=500.0, sr=ANALYSIS_SR, frame_length=1024, hop_length=160)

    def has_voice(seg) -> bool:
        a = int(seg[0] * hop / sr * ANALYSIS_SR / 160)
        b = int((seg[1] * hop + frame) / sr * ANALYSIS_SR / 160) + 1
        return bool(voiced[a:b].any())

    while len(segments) > 1:
        last = segments[-1]
        if (last[1] - last[0]) * hop / sr < max_artifact and not has_voice(last):
            segments.pop()
        else:
            break

    start = max(int(segments[0][0] * hop) - int(0.02 * sr), 0)
    end = min(int(segments[-1][1] * hop) + frame + int(0.03 * sr), len(wav))
    return _fade_out(_fade_in(wav[start:end], sr), sr)


def sound_bursts(wav: np.ndarray, sr: int, gap: float = 0.12, floor_db: float = 30.0) -> list[tuple[int, int]]:
    """``(start, end)`` sample ranges of the separate bursts of sound in a clip: stretches within ``floor_db`` of
    its loudest part, split at pauses of at least ``gap`` seconds."""
    import librosa

    if len(wav) == 0:
        return []
    hop = max(int(0.01 * sr), 1)
    rms = librosa.feature.rms(y=np.asarray(wav, dtype=np.float32), frame_length=hop * 4, hop_length=hop)[0]
    db = 20 * np.log10(rms + 1e-9)
    active = np.where(db > db.max() - floor_db)[0]
    if not len(active):
        return []
    bursts, start, prev = [], active[0], active[0]
    for frame in active[1:]:
        if (frame - prev) * hop >= gap * sr:
            bursts.append((start * hop, prev * hop + hop * 4))
            start = frame
        prev = frame
    bursts.append((start * hop, min(prev * hop + hop * 4, len(wav))))
    return bursts


def allowed_bursts(expected_seconds: float) -> int:
    """How many separate bursts of sound a short line of ``expected_seconds`` may have (about 4 syllables a
    second, plus one), so a word that is repeated or stuttered is caught."""
    return max(2, int(np.ceil(expected_seconds * 4.0)) + 1)


def median_pitch(wav: np.ndarray, sr: int) -> Optional[float]:
    """Median F0 (Hz) of a clip, or ``None`` if it has too little voiced speech."""
    import librosa

    y16 = librosa.resample(wav, orig_sr=sr, target_sr=ANALYSIS_SR) if sr != ANALYSIS_SR else wav
    f0 = estimate_f0(y16, ANALYSIS_SR)
    return float(np.median(f0)) if len(f0) >= 5 else None


def fold_octave(f: Optional[float], ref: Optional[float], strict: bool = False) -> Optional[float]:
    """Undo pitch-tracker octave errors relative to ``ref``.

    Default (for the original, often music-laden audio): move ``f`` by whole octaves to within ~half an
    octave of ``ref``. ``strict`` (for clean generated speech, where tracking is reliable): only undo a
    jump of almost exactly one octave, so a genuinely too-low (e.g. male-sounding) line is still caught.
    """
    if not f or not ref:
        return f
    if strict:
        off = 12 * np.log2(f / ref)
        if abs(abs(off) - 12) <= 1.5:
            return f / 2 if off > 0 else f * 2
        return f
    while f < ref / 1.6:
        f *= 2
    while f > ref * 1.6:
        f /= 2
    return f


def semitones(f: Optional[float], ref: Optional[float]) -> Optional[float]:
    return None if not f or not ref else float(12 * np.log2(f / ref))


def loop_to_length(wav: np.ndarray, sr: int, seconds: float, gap: float = 0.1) -> np.ndarray:
    """Repeat a short clip (with short pauses) until it lasts ``seconds``, so a voice embedding can be
    taken from it; clips that are already long enough are returned unchanged."""
    target = int(seconds * sr)
    if len(wav) == 0 or len(wav) >= target:
        return wav
    pause = np.zeros(int(gap * sr), dtype=np.float32)
    reps = int(np.ceil(target / (len(wav) + len(pause))))
    return np.concatenate([np.concatenate([wav, pause])] * reps)[: max(target, len(wav))].astype(np.float32)


def pitch_shift(wav: np.ndarray, sr: int, steps: float) -> np.ndarray:
    """Shift pitch by ``steps`` semitones keeping duration (ffmpeg ``asetrate`` + ``atempo``).

    This also moves the formants, which is inaudible for the small (<= 3 semitone) corrections used here.
    """
    if abs(steps) < 0.25:
        return wav
    import soundfile as sf

    ratio = 2 ** (steps / 12)
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = os.path.join(tmp, "in.wav"), os.path.join(tmp, "out.wav")
        sf.write(src, wav, sr, subtype="FLOAT")
        filt = f"asetrate={sr * ratio:.3f},aresample={sr},atempo={1 / ratio:.5f}"
        _run([_require_binary("ffmpeg"), "-y", "-v", "error", "-i", src, "-filter:a", filt, "-c:a", "pcm_f32le", dst])
        out, _ = sf.read(dst, dtype="float32")
    return out


def time_stretch(wav: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """Speed speech up by ``rate`` without changing pitch using ffmpeg's ``atempo`` (WSOLA).

    WSOLA keeps speech crisp; a phase vocoder (``librosa.effects.time_stretch``) smears it into a metallic,
    "phasey" sound that is most audible at the end of words.
    """
    import soundfile as sf

    with tempfile.TemporaryDirectory() as tmp:
        src, dst = os.path.join(tmp, "in.wav"), os.path.join(tmp, "out.wav")
        sf.write(src, wav, sr, subtype="FLOAT")
        _run([_require_binary("ffmpeg"), "-y", "-v", "error", "-i", src, "-filter:a", f"atempo={rate:.5f}", "-c:a", "pcm_f32le", dst])
        out, _ = sf.read(dst, dtype="float32")
    return out


def fit_to_slot(
    wav: np.ndarray,
    sr: int,
    slot: float,
    available: float,
    max_speedup: float = 1.35,
) -> tuple[np.ndarray, float]:
    """Make ``wav`` fit a subtitle slot.

    * shorter than ``slot`` -> unchanged
    * longer -> sped up (pitch preserved) by at most ``max_speedup`` to reach ``slot``
    * still longer than ``available`` (time until the next line) -> trimmed with a fade-out

    Returns the fitted audio and the speed factor that was applied.
    """
    duration = len(wav) / sr
    rate = 1.0
    if slot > 0 and duration > slot:
        rate = min(duration / slot, max(max_speedup, 1.0))
        if rate > 1.01:
            wav = time_stretch(wav.astype(np.float32), sr, rate)
        else:
            rate = 1.0
    limit = int(max(available, slot) * sr)
    if limit > 0 and len(wav) > limit:
        wav = wav[:limit]
    return _fade_out(_fade_in(wav.astype(np.float32), sr), sr), rate


def mix_timeline(
    clips: list[tuple[float, np.ndarray]],
    total_seconds: float,
    sr: int,
    background: Optional[np.ndarray] = None,
    background_mode: str = "duck",
    duck_db: float = -18.0,
    speech_regions: Optional[list[tuple[float, float]]] = None,
    ramp_seconds: float = 0.12,
) -> np.ndarray:
    n = int(round(total_seconds * sr))
    out = np.zeros(n, dtype=np.float32)

    if background is not None and background_mode != "mute":
        bg = np.zeros(n, dtype=np.float32)
        m = min(n, len(background))
        bg[:m] = background[:m]
        if background_mode == "duck" and speech_regions:
            gain = np.ones(n, dtype=np.float32)
            low = 10 ** (duck_db / 20)
            for start, end in speech_regions:
                a, b = max(int(start * sr), 0), min(int(end * sr), n)
                if b > a:
                    gain[a:b] = low
            # Smooth the gain envelope to avoid clicks.
            k = max(int(ramp_seconds * sr), 1)
            gain = np.convolve(gain, np.ones(k, dtype=np.float32) / k, mode="same")
            bg *= gain
        out += bg

    for start, wav in clips:
        a = max(int(round(start * sr)), 0)
        if a >= n:
            continue
        b = min(a + len(wav), n)
        out[a:b] += wav[: b - a]

    peak = float(np.max(np.abs(out))) if n else 0.0
    if peak > 0.99:
        out *= 0.99 / peak
    return out


# -----------------------------
# Transcription (optional)
# -----------------------------


def transcribe_to_lines(audio_path: str, model_name: str = "base", language: Optional[str] = None, device=None):
    try:
        import stable_whisper
    except ImportError as exc:
        raise ImportError(
            "No subtitle file was given and stable-ts is not installed. "
            'Upload an SRT or install with: pip install "voxcpm[timestamps]"'
        ) from exc
    model = stable_whisper.load_model(model_name, device=device)
    result = model.transcribe(audio_path, language=language)
    lines = []
    for seg in result.segments:
        text = str(seg.text).strip()
        if text:
            lines.append(SubtitleLine(index=len(lines) + 1, start=float(seg.start), end=float(seg.end), text=text))
    return lines


# -----------------------------
# Pipeline
# -----------------------------


def _clean_control(text: str) -> str:
    return re.sub(r"[()（）]", "", text or "").strip(" ,")


def require_voxcpm2(model) -> None:
    """Dubbing only runs on VoxCPM2: it relies on reference-audio cloning plus style control, 48 kHz output
    and 30-language text, none of which VoxCPM 1.x provides."""
    from .model.voxcpm2 import VoxCPM2Model

    if not isinstance(getattr(model, "tts_model", None), VoxCPM2Model):
        kind = type(getattr(model, "tts_model", model)).__name__
        raise ValueError(
            f"Video dubbing requires a VoxCPM2 model (e.g. openbmb/VoxCPM2), but a {kind} was loaded. "
            "Load VoxCPM2 with --model-id / --hf-model-id openbmb/VoxCPM2."
        )


def speaker_seed(name: str, base: Optional[int] = None) -> int:
    """A fixed seed per character so every line of that character is sampled the same way."""
    return ((base or 0) + zlib.crc32(name.encode("utf-8"))) & 0x7FFFFFFF


class VideoDubber:
    """Runs the dubbing pipeline with an already loaded ``voxcpm.VoxCPM`` instance.

    ``model`` may be ``None`` while only calling :meth:`prepare` (analysis needs no TTS model).
    """

    def __init__(self, model, workdir: str | os.PathLike):
        if model is not None:
            require_voxcpm2(model)
        self.model = model
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.original_16k = self.workdir / "original_16k.wav"
        self.vocals_44k = self.workdir / "vocals_44k.wav"
        self.vocals_16k = self.workdir / "vocals_16k.wav"
        self.instrumental_44k = self.workdir / "instrumental_44k.wav"
        self.warnings: list[str] = []
        self.speaker_model = None
        self._embedding_cache: dict[str, Optional[np.ndarray]] = {}

    @property
    def sample_rate(self) -> int:
        return int(self.model.tts_model.sample_rate)

    # ---- preparation ----

    def separate(self, media_path: str, device: Optional[str] = None) -> None:
        separate_vocals(media_path, str(self.vocals_44k), str(self.instrumental_44k), device=device)
        extract_audio(str(self.vocals_44k), str(self.vocals_16k), ANALYSIS_SR)

    def prepare(
        self,
        media_path: str,
        srt_path: Optional[str] = None,
        transcribe_model: str = "base",
        language: Optional[str] = None,
        remove_vocals: bool = False,
        detect_emotion: bool = True,
        emotion_model=None,
        separation_device: Optional[str] = None,
        identify_speakers: bool = True,
        num_speakers: Optional[int] = None,
        speaker_model=None,
        two_voices: bool = False,
        speaker_style: bool = True,
    ) -> tuple[list[SubtitleLine], dict[str, SpeakerProfile]]:
        """Analyse the video. With ``two_voices`` every character is mapped to one adult male or one adult
        female voice, styled after how that character speaks unless ``speaker_style`` is off (see
        :func:`collapse_to_two_voices`)."""
        extract_audio(media_path, str(self.original_16k), ANALYSIS_SR)
        if remove_vocals:
            try:
                self.separate(media_path, separation_device)
            except Exception as exc:  # analysis still works on the original mix
                message = f"Vocal removal skipped, analysing the original audio instead: {exc}"
                logger.warning(message)
                self.warnings.append(message)
        speech_path = self._speech_path()
        if srt_path:
            lines = load_srt(srt_path)
        else:
            lines = transcribe_to_lines(str(speech_path), model_name=transcribe_model, language=language)
        if not lines:
            raise ValueError("No subtitle lines found.")

        audio = self._load_speech_audio()
        analyze_lines(lines, audio, ANALYSIS_SR)
        if detect_emotion:
            self._detect_emotions(lines, audio, emotion_model)
        if identify_speakers:
            self._identify_speakers(lines, audio, speaker_model, num_speakers)
        profiles = build_speaker_profiles(lines)
        assign_tones(lines)  # relative to each original character, before characters are merged
        if two_voices:
            profiles = collapse_to_two_voices(lines, profiles, speaker_style=speaker_style)
        return lines, profiles

    def _detect_emotions(self, lines: list[SubtitleLine], audio: np.ndarray, emotion_model) -> None:
        import soundfile as sf

        clip_dir = self.workdir / "emotion"
        clip_dir.mkdir(exist_ok=True)
        paths, targets = [], []
        for line in lines:
            seg = _segment(audio, ANALYSIS_SR, line.start, line.end, pad=0.1)
            if len(seg) < int(0.3 * ANALYSIS_SR):
                continue
            path = clip_dir / f"line_{line.index:04d}.wav"
            sf.write(str(path), seg, ANALYSIS_SR)
            paths.append(str(path))
            targets.append(line)
        try:
            model = emotion_model or load_emotion_model()
            for line, emotion in zip(targets, detect_emotions(paths, model)):
                line.emotion = emotion
        except Exception as exc:  # emotion is a refinement; never fail the whole dub because of it
            logger.warning("Emotion detection skipped: %s", exc)

    def _get_speaker_model(self, model=None):
        if model is not None:
            self.speaker_model = model
        if self.speaker_model is None:
            self.speaker_model = load_speaker_model()
        return self.speaker_model

    def _identify_speakers(self, lines: list[SubtitleLine], audio: np.ndarray, model, num_speakers) -> None:
        """Embed every line's voice; name untagged lines by voice cluster and score tagged ones."""
        try:
            model = self._get_speaker_model(model)
            for line in lines:
                seg = _segment(audio, ANALYSIS_SR, line.start, line.end)
                line.features["embedding"] = speaker_embedding(model, seg)
        except Exception as exc:
            message = f"Voice-based speaker detection skipped ({exc}); untagged lines are grouped by pitch."
            logger.warning(message)
            self.warnings.append(message)
            return
        untagged = [l for l in lines if not l.speaker]
        if untagged:
            labels = cluster_speakers([l.features.get("embedding") for l in untagged], num_speakers=num_speakers)
            for line, label in zip(untagged, labels):
                line.speaker = f"Speaker {label + 1}"
        score_voice_consistency(lines)

    def _score_original_voices(self, lines: list[SubtitleLine], audio16: np.ndarray) -> None:
        """Voice-match every original line against its speaker when the analysis is not at hand (e.g. the
        lines were edited in the web demo), so a speaker's reference leaves out lines that sound like
        someone else."""
        if all("voice_match" in l.features for l in lines):
            return
        try:
            model = self._get_speaker_model()
            for line in lines:
                if line.features.get("embedding") is None:
                    line.features["embedding"] = speaker_embedding(
                        model, _segment(audio16, ANALYSIS_SR, line.start, line.end)
                    )
        except Exception as exc:
            logger.warning("Could not voice-match the original lines: %s", exc)
            return
        score_voice_consistency(lines)

    def _embedding_of_file(self, path: str) -> Optional[np.ndarray]:
        if path not in self._embedding_cache:
            import soundfile as sf

            wav, sr = sf.read(path, dtype="float32", always_2d=False)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            self._embedding_cache[path] = self._embed(wav, sr)
        return self._embedding_cache[path]

    def _embed(self, wav: np.ndarray, sr: int) -> Optional[np.ndarray]:
        import librosa

        if sr != ANALYSIS_SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=ANALYSIS_SR)
        return speaker_embedding(self._get_speaker_model(), wav)

    def _speech_path(self) -> Path:
        """Clean vocals when vocal removal ran, otherwise the original mix."""
        return self.vocals_16k if self.vocals_16k.exists() else self.original_16k

    def _load_speech_audio(self) -> np.ndarray:
        import soundfile as sf

        audio, sr = sf.read(str(self._speech_path()), dtype="float32")
        assert sr == ANALYSIS_SR
        return audio

    def _write_reference(self, name: str, wav: np.ndarray, sr: int = ANALYSIS_SR) -> Optional[str]:
        import soundfile as sf

        if len(wav) < MIN_REFERENCE_SECONDS * sr:
            return None
        safe = re.sub(r"[^\w\-]+", "_", name)[:40] or "ref"
        path = self.workdir / "refs" / f"{safe}.wav"
        path.parent.mkdir(exist_ok=True)
        sf.write(str(path), wav, sr)
        return str(path)

    def build_speaker_reference(
        self, profile: SpeakerProfile, lines: list[SubtitleLine], audio: np.ndarray, max_seconds: float = 8.0
    ) -> Optional[str]:
        """Concatenate the speaker's most typical lines into one reference clip.

        Lines whose emotion matches the speaker's usual mood are preferred, then longer lines, so the
        reference captures the character's normal voice rather than a one-off shout or whisper. Lines whose
        voice does not match the character (mis-tagged lines, overlapping speakers) are left out.
        """
        own = [l for l in lines if l.speaker == profile.name]
        matching = [l for l in own if l.features.get("voice_match", 1.0) >= 0.4]
        own = matching or own  # lines that sound like someone else must not leak into the reference
        usual = Counter(l.emotion or "neutral" for l in own).most_common(1)
        usual_emotion = usual[0][0] if usual else "neutral"
        own.sort(key=lambda l: ((l.emotion or "neutral") != usual_emotion, -l.duration))
        pieces, total = [], 0.0
        gap = np.zeros(int(0.2 * ANALYSIS_SR), dtype=np.float32)
        for line in own:
            if total >= max_seconds:
                break
            seg = _segment(audio, ANALYSIS_SR, line.start, line.end, pad=0.1)
            if len(seg) == 0:
                continue
            pieces += [seg, gap]
            total += len(seg) / ANALYSIS_SR
        if not pieces:
            return None
        return self._write_reference(f"speaker_{profile.name}", np.concatenate(pieces[:-1]))

    # ---- generation ----

    def design_reference(
        self,
        profile: SpeakerProfile,
        lines: list[SubtitleLine],
        options: DubOptions,
        audio16: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        """Create a designed character's voice once: a calm, neutral reading of one of its lines from the
        voice description alone.

        Every line of the character then clones this clip, so the voice stays the same person throughout
        while each line's emotion still comes from its style hint (cloning the first *emotional* line
        would bake that emotion into every later line).

        Because every line inherits this reading, it is checked against the original speech (``audio16``)
        before it is used: its pitch must be in the adult range for the gender and within
        ``pitch_tolerance`` semitones of the original speakers' pitch, and (with ``verify_voice``) its voice
        should be at least ``min_voice_similarity`` close to theirs. Up to ``design_attempts`` seeds are
        tried; the best reading is kept, nudged towards the original pitch, and a warning naming the clip
        is recorded when none passed. A ``reference_wav`` on the profile replaces the designed voice.
        """
        if profile.reference_wav:
            return profile.reference_wav
        own = [l for l in lines if l.speaker == profile.name]
        if not own:
            return None
        sample = max(own, key=lambda l: (emotion_hint(l.emotion) == "", len(l.text)))
        description = _clean_control(profile.description or default_description(profile.gender))
        text = f"({description}, calm and neutral){sample.text}"
        low, high = ADULT_PITCH_RANGE.get(profile.gender, (ADULT_PITCH_RANGE["male"][0], ADULT_PITCH_RANGE["female"][1]))
        target_pitch, target_voice = self._original_voice(own, audio16, options)
        if target_pitch is not None:
            target_pitch = float(np.clip(fold_octave(target_pitch, (low + high) / 2), low, high))
        sr = self.sample_rate

        def check(wav):
            pitch = median_pitch(wav, sr)
            if pitch is not None and target_pitch:
                pitch = fold_octave(pitch, target_pitch, strict=True)
            in_range = pitch is None or low <= pitch <= high
            offset = semitones(pitch, target_pitch)
            similarity = None
            if target_voice is not None:
                emb = self._embed(wav, sr)
                similarity = None if emb is None else float(emb @ target_voice)
            ok = (
                in_range
                and (offset is None or abs(offset) <= options.pitch_tolerance)
                and (similarity is None or similarity >= options.min_voice_similarity)
            )
            rank = (ok, in_range, -round(abs(offset or 0.0), 1), similarity if similarity is not None else 0.0)
            return {"wav": wav, "pitch": pitch, "offset": offset, "similarity": similarity, "ok": ok, "rank": rank}

        best = None
        tries = max(1, options.design_attempts, options.max_attempts)
        for attempt in range(tries):
            seed = (speaker_seed(profile.name, options.seed) + attempt * 7919) & 0x7FFFFFFF
            candidate = check(self._generate_once(text, None, seed, options))
            if best is None or candidate["rank"] > best["rank"]:
                best = candidate
            if best["ok"]:
                break
        wav = best["wav"]
        if best["offset"] is not None and abs(best["offset"]) > 1.5:
            wav = pitch_shift(wav, sr, float(np.clip(-best["offset"], -options.max_pitch_correction, options.max_pitch_correction)))
        path = self._write_reference(f"designed_{profile.name}", wav, sr)
        if not best["ok"]:
            details = []
            if best["pitch"] is not None and target_pitch:
                details.append(f"pitch {best['pitch']:.0f} Hz vs original {target_pitch:.0f} Hz")
            if best["similarity"] is not None:
                details.append(f"voice match {best['similarity']:.2f}")
            message = (
                f"The designed {profile.name} voice did not match the original after {tries} tries "
                f"({', '.join(details) or 'outside the adult range'}); using the closest one ({path}). "
                "Listen to it and, if needed, change the voice description or the seed, or give this voice a "
                "reference clip (\"reference\" in --speakers)."
            )
            logger.warning(message)
            self.warnings.append(message)
        return path

    def first_line(self, name: str, gender: str, lines: list[SubtitleLine], audio16: np.ndarray) -> SubtitleLine:
        """The line a ``clone_first`` voice is copied from: the character's first line that is long enough
        to clone (>= ``MIN_REFERENCE_SECONDS``) and, for a known gender, spoken at an adult pitch."""
        own = [l for l in lines if l.speaker == name]
        long_enough = [l for l in own if l.duration >= MIN_REFERENCE_SECONDS] or own
        low, high = ADULT_PITCH_RANGE.get(gender, (0.0, float("inf")))
        for line in long_enough:
            pitch = median_pitch(_segment(audio16, ANALYSIS_SR, line.start, line.end), ANALYSIS_SR)
            if pitch is None or low <= pitch <= high:
                return line
        return long_enough[0]

    def _original_voice(
        self, lines: list[SubtitleLine], audio16: Optional[np.ndarray], options: DubOptions, max_lines: int = 12
    ) -> tuple[Optional[float], Optional[np.ndarray]]:
        """Median pitch and average voice embedding of the original speech of ``lines`` (longest first)."""
        if audio16 is None or not len(audio16):
            return None, None
        longest = sorted(lines, key=lambda l: -l.duration)[:max_lines]
        segments = [_segment(audio16, ANALYSIS_SR, l.start, l.end) for l in longest]
        f0 = [estimate_f0(seg, ANALYSIS_SR) for seg in segments]
        f0 = np.concatenate([f for f in f0 if len(f)]) if any(len(f) for f in f0) else np.zeros(0)
        pitch = float(np.median(f0)) if len(f0) >= 5 else None
        voice = None
        if options.verify_voice:
            try:
                embs = [e for e in (speaker_embedding(self._get_speaker_model(), seg) for seg in segments) if e is not None]
                voice = _centroid(embs) if embs else None
            except Exception as exc:
                logger.warning("Voice check of the designed voice skipped: %s", exc)
        return pitch, voice

    def _line_text(
        self,
        line: SubtitleLine,
        profile: SpeakerProfile,
        mode: str,
        options: DubOptions,
        faster: bool = False,
        short: bool = False,
    ) -> str:
        # The character's description (gender, pitch register, delivery) is given on every line, also when
        # cloning, so VoxCPM keeps the same kind of voice instead of drifting between lines.
        # A short line (one or two words) gets only its emotion: a long voice prompt makes VoxCPM babble on
        # past the words. Its voice comes from the reference clip instead.
        hints = [] if short else [profile.description or default_description(profile.gender)]
        if options.use_tone_control:
            hints.append(emotion_hint(line.emotion) if short else line_style(line))
        text = line.text
        if short:
            # One word trailing off ("He...") tends to come out mumbled: ask for it clearly, and say a trailing
            # ellipsis as a full stop (the subtitle text itself is unchanged).
            hints.append("spoken clearly")
            text = re.sub(r"\s*(?:\.{2,}|…+|។{2,})\s*$", ".", text) or line.text
        if faster:
            hints.append("speaking quickly")
        control = _clean_control(", ".join(h for h in hints if h))
        return f"({control}){text}" if control else text

    def _max_len(self, slot: float, expected: Optional[float] = None) -> Optional[int]:
        """Generation length cap (audio tokens) for a line with a ``slot``-second subtitle slot, so the model
        cannot run on for many seconds past the line (it otherwise sometimes generates 10-25 s for one word).
        For a short line the cap also follows how long its words take to say (``expected`` seconds)."""
        tts = getattr(self.model, "tts_model", None)
        try:
            rate = tts._encode_sample_rate / (tts.patch_size * tts.chunk_size)
        except (AttributeError, TypeError, ZeroDivisionError):
            return None
        if slot <= 0:
            return None
        seconds = slot * 2.0 + 1.5
        if expected:
            seconds = min(seconds, expected * 3.0 + 1.2)
        return int(np.ceil(rate * seconds))

    def _generate_once(self, text, reference, seed, options: DubOptions, max_len: Optional[int] = None) -> np.ndarray:
        extra = {"max_len": max_len} if max_len else {}
        wav = self.model.generate(
            text=text,
            reference_wav_path=reference,
            **extra,
            cfg_value=options.cfg_value,
            inference_timesteps=options.inference_timesteps,
            normalize=options.normalize,
            denoise=False,  # references are denoised once up front (see _prepared_reference)
            seed=seed,
        )
        wav = np.asarray(wav, dtype=np.float32)
        if options.clean_generated:
            wav = clean_voice(wav, self.sample_rate, target_lufs=options.target_lufs)
        return finalize_edges(trim_silence(wav), self.sample_rate)

    def _prepared_reference(self, path: Optional[str], options: DubOptions) -> Optional[str]:
        """Denoise a reference clip once (ZipEnhancer) instead of on every generation call."""
        denoiser = getattr(self.model, "denoiser", None)
        if not path or not options.denoise_reference or denoiser is None:
            return path
        cache = self.__dict__.setdefault("_denoised_refs", {})
        if path not in cache:
            out = str(Path(path).with_name(Path(path).stem + "_denoised.wav"))
            try:
                denoiser.enhance(path, output_path=out)
                cache[path] = out
            except Exception as exc:
                logger.warning("Reference denoising failed for %s: %s", path, exc)
                cache[path] = path
        return cache[path]

    def _generate_best(
        self,
        line,
        profile,
        mode,
        reference,
        target,
        options: DubOptions,
        others=(),
        target_pitch=None,
        attempts: Optional[int] = None,
        min_similarity: Optional[float] = None,
        peers: Optional[np.ndarray] = None,
        pitch_range: Optional[tuple[float, float]] = None,
        pitch_limits: Optional[tuple[float, float]] = None,
    ) -> dict:
        """Generate a line, retrying (new seed, faster pace if too long) until the voice matches the
        character's reference, its pitch matches the character's, and it fits its slot, or ``attempts``
        (default ``max_attempts``) is reached; keep the best and nudge its pitch towards ``target_pitch``.

        ``others`` are the reference embeddings of the *other* characters: a line must sound more like its
        own character than like any of them. ``min_similarity`` raises the required closeness to the
        reference, and ``peers`` (the average voice of the character's lines accepted so far) must also be
        at least ``min_consistency`` close, so every line sounds like the lines before it.

        A try that runs on far past its slot (babble) or is nearly silent never counts as consistent.
        ``pitch_range`` (Hz, instead of ``target_pitch``) requires the pitch to lie within it, e.g. a kid's range.
        ``pitch_limits`` (Hz) are hard bounds every try must also stay within, e.g. so a woman's line never drops
        to a pitch that sounds male (``best["gender_ok"]``).
        """
        threshold = options.min_voice_similarity if min_similarity is None else min_similarity
        sr = self.sample_rate
        slot = line.duration
        reference = self._prepared_reference(reference, options)
        # A line is short when its slot is, or when its words are (one word in a 1.7 s slot is still one word).
        expected = estimate_speech_seconds(line.text)
        short = 0 < slot <= options.short_line_seconds or 0 < expected <= 0.8
        consistency_needed = options.short_line_consistency if short else options.min_consistency
        if short and min_similarity is not None:
            threshold = min(threshold, options.short_line_consistency)
        max_len = self._max_len(slot, expected if short else None)

        def problems(candidate) -> list[str]:
            """What is wrong with a try (empty when it is consistent), in plain words."""
            sim = candidate["similarity"]
            margin = candidate["margin"]
            consistency = candidate["consistency"]
            found = []
            longest = slot * 1.5 + 0.3 if short else slot * 1.8 + 0.4  # past this it is babble, not the words
            if short and expected > 0:
                longest = min(longest, expected * 2.5 + 0.8)
            if candidate["duration"] < 0.15:
                found.append("almost silent")
            elif slot > 0 and candidate["duration"] > longest:
                found.append("runs on too long")
            elif short and candidate["bursts"] > allowed_bursts(expected):
                found.append("repeats words")  # "He... he... he..." is stuttering, not the line
            if sim is not None and sim < threshold:
                found.append("doesn't match the speaker's voice")
            elif margin is not None and margin < options.voice_margin:
                found.append("sounds like another speaker")
            if consistency is not None and consistency < consistency_needed:
                found.append("differs from the speaker's other lines")
            off = candidate["pitch_offset"]
            if off is not None and abs(off) > (0.5 if pitch_range else options.pitch_tolerance):
                found.append("pitch too high" if off > 0 else "pitch too low")
            if not gender_clear(candidate):
                found.append("unclear gender")
            return found

        def score(candidate):
            off = candidate["pitch_offset"]
            overflow = max(candidate["duration"] / slot - 1.0, 0.0) if slot > 0 else 0.0
            margin, sim = candidate["margin"], candidate["similarity"]
            closeness = margin if margin is not None else (sim if sim is not None else 0.0)
            return (not problems(candidate), -round(overflow, 2), closeness - 0.03 * abs(off or 0.0))

        def gender_clear(candidate):
            pitch = candidate["pitch"]
            return pitch_limits is None or pitch is None or pitch_limits[0] <= pitch <= pitch_limits[1]

        best = None
        for attempt in range(max(1, attempts or options.max_attempts)):
            faster = best is not None and slot > 0 and best["duration"] > slot * 1.15
            text = self._line_text(line, profile, mode, options, faster=faster, short=short)
            seed = (speaker_seed(line.speaker, options.seed) + attempt * 7919) & 0x7FFFFFFF
            wav = self._generate_once(text, reference, seed, options, max_len=max_len)
            similarity = margin = consistency = emb = None
            if target is not None and options.verify_voice:
                emb = self._embed(loop_to_length(wav, sr, 1.5), sr)  # short clips are checked too
                if emb is not None:
                    similarity = float(emb @ target)
                    if len(others):
                        margin = similarity - max(float(emb @ o) for o in others)
                    if peers is not None:
                        consistency = float(emb @ peers)
            if pitch_range:
                pitch = median_pitch(wav, sr)
                low, high = pitch_range
                offset = None if pitch is None else 0.0 if low <= pitch <= high else semitones(pitch, low if pitch < low else high)
            else:
                pitch = fold_octave(median_pitch(wav, sr), target_pitch, strict=True) if target_pitch else None
                offset = semitones(pitch, target_pitch)
            candidate = {
                "wav": wav,
                "similarity": similarity,
                "margin": margin,
                "consistency": consistency,
                "embedding": emb,
                "pitch": pitch,
                "pitch_offset": offset,
                "duration": len(wav) / sr,
                "bursts": len(sound_bursts(wav, sr)) if short else 0,
                "attempt": attempt + 1,
            }
            if best is None or score(candidate) > score(best):
                best = candidate
            voice_ok, neg_overflow, _ = score(best)
            # Once consistent, only keep retrying (briefly) to get a line that fits its slot.
            if voice_ok and (-neg_overflow <= 0.15 or attempt + 1 >= 3):
                break
        best["tries"] = attempt + 1
        best["voice_ok"] = score(best)[0]
        best["gender_ok"] = gender_clear(best)
        best["issues"] = problems(best)
        best["trimmed"] = False
        if short and best["bursts"] > allowed_bursts(expected):
            # Every try stuttered: keep the first bursts (the words themselves) and fade out the repeats.
            keep = sound_bursts(best["wav"], sr)[allowed_bursts(expected) - 1][1]
            best["wav"] = _fade_out(best["wav"][: min(keep + int(0.05 * sr), len(best["wav"]))], sr)
            best["duration"] = len(best["wav"]) / sr
            best["trimmed"] = True
        best["pitch_shift"] = 0.0
        off = best["pitch_offset"]
        # Pitch of a clip under ~1 s is too unreliable to correct (and shifting it changes the voice).
        if off is not None and abs(off) > (0.25 if pitch_range else 1.5) and best["duration"] >= 1.0:
            steps = float(np.clip(-off, -options.max_pitch_correction, options.max_pitch_correction))
            best["wav"] = pitch_shift(best["wav"], sr, steps)
            best["pitch_shift"] = steps
        return best

    def _target_pitch(
        self,
        line: SubtitleLine,
        reference: Optional[str],
        audio16: np.ndarray,
        speaker_pitch: Optional[float] = None,
    ) -> Optional[float]:
        """The pitch this line should have: the character's reference pitch, moved by how far the original
        line sits above or below the character's *own* usual pitch (``speaker_pitch``), so a shout stays
        higher than a calm line. The move is kept within -2/+3 semitones so the character always sounds
        like the same person. Without ``speaker_pitch`` (a line cloned from itself) it is the reference pitch."""
        if not reference:
            return None
        cache = self.__dict__.setdefault("_ref_pitch", {})
        if reference not in cache:
            import soundfile as sf

            wav, sr = sf.read(reference, dtype="float32")
            cache[reference] = median_pitch(wav if wav.ndim == 1 else wav.mean(axis=1), sr)
        ref_pitch = cache[reference]
        if ref_pitch is None or not speaker_pitch:
            return ref_pitch
        line_pitch = median_pitch(_segment(audio16, ANALYSIS_SR, line.start, line.end), ANALYSIS_SR)
        if line_pitch is None:
            return ref_pitch
        offset = semitones(fold_octave(line_pitch, speaker_pitch), speaker_pitch)
        return float(ref_pitch * 2 ** (np.clip(offset, -2.0, 3.0) / 12))

    def _speaker_pitches(self, lines: list[SubtitleLine], audio16: np.ndarray) -> dict[str, Optional[float]]:
        """Each character's usual pitch in the original: the median of their lines' pitches."""
        found: dict[str, list[float]] = {}
        for line in lines:
            pitch = median_pitch(_segment(audio16, ANALYSIS_SR, line.start, line.end), ANALYSIS_SR)
            if pitch is not None:
                found.setdefault(line.speaker, []).append(pitch)
        return {name: _median(values) for name, values in found.items()}

    def _background(self, media_path: str, mode: str, sr: int) -> Optional[np.ndarray]:
        import soundfile as sf

        if mode == "mute":
            return None
        if mode == "instrumental":
            if not self.instrumental_44k.exists():
                try:
                    self.separate(media_path)
                except Exception as exc:
                    raise RuntimeError(
                        f"The 'instrumental' background needs vocal removal, which failed: {exc}\n"
                        "Use the 'mute' background to keep only the generated voices."
                    ) from exc
            source = str(self.instrumental_44k)
        else:
            source = media_path
        bg_path = self.workdir / f"background_{mode}_{sr}.wav"
        extract_audio(source, str(bg_path), sr)
        audio, _ = sf.read(str(bg_path), dtype="float32")
        return audio

    def run(
        self,
        media_path: str,
        lines: list[SubtitleLine],
        profiles: dict[str, SpeakerProfile],
        output_path: str,
        options: Optional[DubOptions] = None,
        progress: Optional[Callable[[int, int, SubtitleLine], None]] = None,
    ) -> DubResult:
        import soundfile as sf

        options = options or DubOptions()
        if options.background not in BACKGROUND_MODES:
            raise ValueError(f"background must be one of {BACKGROUND_MODES}")
        if not self.original_16k.exists():
            extract_audio(media_path, str(self.original_16k), ANALYSIS_SR)
        audio16 = self._load_speech_audio()
        lines = sorted(lines, key=lambda l: l.start)

        for line in lines:
            if line.speaker not in profiles:
                profiles[line.speaker] = SpeakerProfile(name=line.speaker or "Speaker", gender=line.gender)

        speaker_refs: dict[str, Optional[str]] = {}
        for name, profile in profiles.items():
            if profile.voice_mode == "clone_speaker":
                speaker_refs[name] = profile.reference_wav or self.build_speaker_reference(profile, lines, audio16)
            elif profile.voice_mode == "design":
                speaker_refs[name] = self.design_reference(profile, lines, options, audio16)
            elif profile.voice_mode == "clone_first" and profile.reference_wav:
                speaker_refs[name] = profile.reference_wav  # your own reference replaces the first line
        # clone_first: generate each character's first line before the others, so it can be their reference.
        anchors = {
            name: self.first_line(name, profile.gender, lines, audio16)
            for name, profile in profiles.items()
            if profile.voice_mode == "clone_first" and not profile.reference_wav and any(l.speaker == name for l in lines)
        }
        anchor_ids = {id(l) for l in anchors.values()}
        speaker_pitches = self._speaker_pitches(lines, audio16)
        if anchors and options.verify_voice:
            self._score_original_voices(lines, audio16)
        accepted: dict[str, list[np.ndarray]] = {}  # voices of each speaker's consistent lines so far
        inconsistent: list[str] = []
        order = [i for i, l in enumerate(lines) if id(l) in anchor_ids] + [
            i for i, l in enumerate(lines) if id(l) not in anchor_ids
        ]

        total = probe_duration(media_path)
        total = max(total, max(l.end for l in lines))
        sr = self.sample_rate
        clips: list[tuple[float, np.ndarray]] = []
        report = []
        clip_dir = self.workdir / "lines"
        clip_dir.mkdir(exist_ok=True)

        for step, i in enumerate(order):
            line = lines[i]
            if progress:
                progress(step, len(lines), line)
            profile = profiles[line.speaker]
            mode = profile.voice_mode
            reference = None
            anchor = anchors.get(line.speaker)
            if mode == "clone_first" and anchor is not line and speaker_refs.get(line.speaker):
                reference = speaker_refs[line.speaker]  # the character's generated first line
                mode = f"clone→#{anchor.index}" if anchor else "clone_speaker"
            elif mode == "clone_first" and anchor is line:
                # The first line copies the voice from several of the speaker's original lines together: one
                # short line (often ~2 s of separated vocals) is too little audio to capture a voice, and
                # every other line of the speaker copies this one.
                reference = self.build_speaker_reference(profile, lines, audio16)
                if reference is None:
                    seg = _segment(audio16, ANALYSIS_SR, line.start, line.end, pad=0.15)
                    reference = self._write_reference(f"line_{line.index:04d}", seg) or profile.reference_wav
            elif mode == "clone_first" or mode == "clone_line":
                seg = _segment(audio16, ANALYSIS_SR, line.start, line.end, pad=0.15)
                reference = self._write_reference(f"line_{line.index:04d}", seg)
                if reference is None:
                    reference = profile.reference_wav
            elif mode == "clone_speaker":
                reference = speaker_refs.get(line.speaker)
            elif mode == "design" and speaker_refs.get(line.speaker):
                # Every line of a designed voice clones its neutral designed reference, so the voice stays the same.
                reference = speaker_refs[line.speaker]
                mode = "design→clone"
            if mode in ("clone_first", "clone_line", "clone_speaker") and reference is None:
                mode = "design"  # too little original audio to clone from

            target = None
            if options.verify_voice and reference:
                try:
                    target = self._embedding_of_file(reference)
                except Exception as exc:
                    logger.warning("Voice check disabled: %s", exc)
                    options.verify_voice = False
            others = []
            if target is not None:
                for other_name, other_ref in speaker_refs.items():
                    if other_name != line.speaker and other_ref:
                        other_emb = self._embedding_of_file(other_ref)
                        if other_emb is not None:
                            others.append(other_emb)
            own_pitch = None if mode == "clone_line" else speaker_pitches.get(line.speaker)
            target_pitch = self._target_pitch(line, reference, audio16, own_pitch)
            is_first = anchor is line and profile.voice_mode == "clone_first"
            copies_generated = mode.startswith("clone→") or mode == "design→clone"
            own_lines = accepted.get(line.speaker)
            best = self._generate_best(
                line,
                profile,
                mode,
                reference,
                target,
                options,
                others=others,
                target_pitch=target_pitch,
                attempts=options.first_line_attempts if is_first else options.max_attempts,
                min_similarity=max(options.min_voice_similarity, options.min_consistency) if copies_generated else None,
                peers=_centroid(own_lines) if copies_generated and own_lines else None,
            )
            if best["voice_ok"] and best.get("embedding") is not None:
                accepted.setdefault(line.speaker, []).append(best["embedding"])
            elif not best["voice_ok"]:
                inconsistent.append(f"#{line.index} ({line.speaker})")
            wav = best["wav"]
            if is_first:
                # Later lines clone this generated line (full speed, before fitting), or, if it is too short,
                # the original clip it was copied from.
                speaker_refs[line.speaker] = self._write_reference(f"first_{line.speaker}", wav, sr) or reference
                if not best["voice_ok"]:
                    message = (
                        f"{line.speaker}'s first line (#{line.index}) did not fully match the original voice after "
                        f"{best['tries']} tries; all their lines copy it. Listen to lines/{line.index:04d}_*.wav "
                        "and, if needed, change the seed or give this character a reference clip."
                    )
                    logger.warning(message)
                    self.warnings.append(message)
            raw_duration = len(wav) / sr
            next_start = lines[i + 1].start if i + 1 < len(lines) else total
            fitted, rate = fit_to_slot(
                wav, sr, slot=line.duration, available=next_start - line.start, max_speedup=options.max_speedup
            )
            sf.write(str(clip_dir / f"{line.index:04d}_{re.sub(r'[^\w-]+', '_', line.speaker)[:24]}.wav"), fitted, sr)
            clips.append((line.start, fitted))
            report.append(
                {
                    "index": line.index,
                    "start": round(line.start, 3),
                    "end": round(line.end, 3),
                    "speaker": line.speaker,
                    "gender": profile.gender,
                    "mode": mode,
                    "emotion": line.emotion,
                    "tone": line_style(line),
                    "voice_match": None if best["similarity"] is None else round(best["similarity"], 2),
                    "consistency": None if best.get("consistency") is None else round(best["consistency"], 2),
                    "attempts": best["tries"],
                    "voice_ok": best["voice_ok"],
                    "pitch_hz": None if best.get("pitch") is None else round(best["pitch"]),
                    "target_pitch_hz": None if target_pitch is None else round(target_pitch),
                    "pitch_shift_st": round(best.get("pitch_shift", 0.0), 1),
                    "generated_s": round(raw_duration, 2),
                    "placed_s": round(len(fitted) / sr, 2),
                    "speedup": round(rate, 2),
                    "text": line.text,
                }
            )
        if progress:
            progress(len(lines), len(lines), lines[-1])
        report.sort(key=lambda r: r["index"])  # first lines were generated ahead of the others
        if inconsistent:
            message = (
                f"{len(inconsistent)} line(s) were still not consistent with their speaker after "
                f"{options.max_attempts} tries and are marked ⚠ in the report: {', '.join(inconsistent)}. "
                "Raise the number of tries, or change those lines' tone / the speaker's voice prompt, and generate again."
            )
            logger.warning(message)
            self.warnings.append(message)

        background = self._background(media_path, options.background, sr)
        mixed = mix_timeline(
            clips,
            total,
            sr,
            background=background,
            background_mode=options.background,
            duck_db=options.duck_db,
            speech_regions=[(s, s + len(w) / sr) for s, w in clips],
        )

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        stem = out.with_suffix("")
        audio_path = str(stem) + "_dub.wav"
        srt_path = str(stem) + ".srt"
        speaker_srt_path = str(stem) + "_speakers.srt"
        sf.write(audio_path, mixed, sr)
        Path(srt_path).write_text(format_srt(lines), encoding="utf-8")
        Path(speaker_srt_path).write_text(format_srt(lines, with_speaker=True), encoding="utf-8")
        (self.workdir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        instrumental_path = None
        if self.instrumental_44k.exists():
            instrumental_path = str(stem) + "_instrumental.wav"
            shutil.copyfile(self.instrumental_44k, instrumental_path)

        video_path = None
        if has_video_stream(media_path):
            video_path = mux_video(media_path, audio_path, str(out), srt_path if options.embed_subtitles else None)
        return DubResult(video_path, audio_path, srt_path, speaker_srt_path, report, instrumental_path)


def profiles_to_json(profiles: dict[str, SpeakerProfile]) -> str:
    return json.dumps({k: asdict(v) for k, v in profiles.items()}, ensure_ascii=False, indent=2)
