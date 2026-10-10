"""Place the voiced lines of an SRT → Speech project on a video and edit them like clips on a timeline.

The editor state of a project lives in ``<workdir>/editor.json``::

    {
      "video": {"source": ..., "preview": ..., "duration": 63.2, "regions": [[1.2, 3.4], ...], "peaks": [...]},
      "clips": {"12": {"start": 41.3, "gain_db": 0.0, "muted": false, "trim_in": 0.0, "trim_out": 0.0}},
      "mix": {"original": "replace", "original_gain_db": 0.0, "vocals_gain_db": 0.0}
    }

``regions`` are the stretches of the video's own soundtrack where someone speaks (see :func:`detect_speech`),
so every line can be lined up with the speech it replaces (:func:`align_to_speech`). :func:`render_mix` and
:func:`export_video` build the final soundtrack from the clips as they are placed, trimmed and mixed.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .dubbing import (
    ANALYSIS_SR, _require_binary, _run, extract_audio, has_video_stream, mix_timeline, probe_duration,
)

logger = logging.getLogger(__name__)

# Containers the browser plays as they are; anything else gets an .mp4 preview copy.
BROWSER_VIDEO = (".mp4", ".m4v", ".webm", ".mov", ".ogv")
# Audio files work too (a soundtrack with no picture); these play in the browser as they are.
BROWSER_AUDIO = (".wav", ".mp3", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".flac")
# music = the soundtrack with the original voices removed; replace = the original silenced where new voices replace it
ORIGINAL_MODES = ("keep", "duck", "replace", "music", "mute")
SILENT_DB = -90.0


def replaced_spans(spans: list[tuple[float, float]], regions: Optional[list] = None) -> list[tuple[float, float]]:
    """Where the original soundtrack is silenced in ``replace`` mode: wherever a new voice plays, and the whole of
    each stretch of original speech a new voice overlaps (so the end of the original line does not leak through).
    Original speech no new voice covers stays."""
    out = list(spans)
    for a, b in regions or []:
        if any(s < b and e > a for s, e in spans):
            out.append((float(a), float(b)))
    out.sort()
    merged: list[list[float]] = []
    for a, b in out:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


SPEED_RANGE = (0.5, 2.0)  # how much a clip can be slowed down / sped up (the pitch is kept)


def default_clip(start: float) -> dict:
    return {
        "start": round(float(start), 3), "gain_db": 0.0, "muted": False, "trim_in": 0.0, "trim_out": 0.0, "speed": 1.0,
    }


def clip_speed(clip: dict) -> float:
    return float(np.clip(float(clip.get("speed") or 1.0), *SPEED_RANGE))


def change_speed(wav: np.ndarray, sr: int, speed: float) -> np.ndarray:
    """``wav`` played ``speed`` times as fast (2 = twice as fast, 0.5 = half), keeping its pitch."""
    if abs(speed - 1.0) < 1e-3 or len(wav) == 0:
        return np.asarray(wav, dtype=np.float32)
    from .dubbing import time_stretch

    return time_stretch(np.asarray(wav, dtype=np.float32), sr, float(np.clip(speed, *SPEED_RANGE)))


def stretched_file(raw: str | Path, speed: float, cache_dir: str | Path) -> Path:
    """A copy of a line's audio at another speed (cached; made again when the line is regenerated)."""
    import soundfile as sf

    raw = Path(raw)
    speed = round(float(np.clip(speed, *SPEED_RANGE)), 2)
    out = Path(cache_dir) / f"{raw.stem}_{int(raw.stat().st_mtime)}_x{speed:.2f}.wav"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        for old in out.parent.glob(f"{raw.stem}_*_x{speed:.2f}.wav"):
            old.unlink()  # made from audio the line no longer has
        wav, sr = sf.read(str(raw), dtype="float32")
        sf.write(str(out), change_speed(wav, sr, speed), sr)
    return out


def clip_line(key: str, clip: dict) -> int:
    """The line a clip plays: clip ``"12"`` is line 12, and the pieces it was cut into (``"12~1"``, …) say so."""
    return int(clip["line"]) if clip.get("line") is not None else int(str(key).split("~")[0])


def collapse_pieces(clips: dict, lines: Iterable[int]) -> dict:
    """Make each of ``lines`` one untrimmed clip again (where its earliest piece was), e.g. before it is
    regenerated: its new audio has a different length, so the old cuts no longer fit."""
    lines = {int(i) for i in lines}
    out, kept = {}, {}
    for key, clip in sorted(clips.items(), key=lambda kv: float(kv[1].get("start", 0))):
        line = clip_line(key, clip)
        if line not in lines:
            out[key] = clip
        elif line not in kept:
            kept[line] = key
            out[key] = dict(clip, line=line, trim_in=0.0, trim_out=0.0)
    return out


def load_state(workdir: str | Path) -> dict:
    path = Path(workdir, "editor.json")
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}
    state.setdefault("video", None)
    state.setdefault("clips", {})
    state.setdefault("mix", {"original": "replace", "original_gain_db": 0.0, "vocals_gain_db": 0.0})
    return state


def save_state(workdir: str | Path, state: dict) -> None:
    Path(workdir, "editor.json").write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")


# -----------------------------
# Speech in the video
# -----------------------------


def _speech_band(wav: np.ndarray, sr: int) -> np.ndarray:
    """Keep roughly the band of the human voice (cuts rumble, bass and hiss that are not speech)."""
    try:
        from scipy.signal import butter, sosfiltfilt

        sos = butter(4, [120.0, min(3800.0, sr / 2 - 100)], btype="bandpass", fs=sr, output="sos")
        return sosfiltfilt(sos, wav).astype(np.float32)
    except Exception:  # scipy missing or a very short clip
        return wav


def detect_speech(
    wav: np.ndarray,
    sr: int,
    frame: float = 0.03,
    min_speech: float = 0.25,
    min_gap: float = 0.35,
    pad: float = 0.05,
    sensitivity: float = 0.5,
) -> list[tuple[float, float]]:
    """Stretches ``(start, end)`` in seconds where the soundtrack has voice-band energy well above its floor.

    ``sensitivity`` 0..1: higher finds quieter speech (and more noise). For videos with loud music, run it on
    the isolated voices (:func:`voxcpm.dubbing.separate_vocals`) instead of the full soundtrack.
    """
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    hop = max(int(sr * frame), 1)
    n = len(wav) // hop
    if n < 3:
        return []
    band = _speech_band(wav, sr)[: n * hop].reshape(n, hop)
    db = 20 * np.log10(np.sqrt(np.mean(band**2, axis=1)) + 1e-9)
    floor, top = np.percentile(db, 15), np.percentile(db, 98)
    if top - floor < 6:  # silence, or a flat noise bed: nothing stands out
        return []
    threshold = floor + max(6.0, (top - floor) * (0.55 - 0.4 * float(np.clip(sensitivity, 0, 1))))
    active = db > threshold
    # Close one-frame holes and drop one-frame blips.
    smoothed = active.copy()
    smoothed[1:-1] = (active[:-2].astype(int) + active[1:-1] + active[2:]) >= 2
    regions: list[list[float]] = []
    start = None
    for i, on in enumerate(list(smoothed) + [False]):
        if on and start is None:
            start = i
        elif not on and start is not None:
            a, b = start * frame, i * frame
            if regions and a - regions[-1][1] < min_gap:
                regions[-1][1] = b
            else:
                regions.append([a, b])
            start = None
    total = len(wav) / sr
    return [
        (round(max(a - pad, 0.0), 3), round(min(b + pad, total), 3)) for a, b in regions if b - a >= min_speech
    ]


def waveform_peaks(wav: np.ndarray, sr: int, per_second: int = 40) -> list[float]:
    """Peak level of every ``1/per_second`` s, scaled to 0..1, for drawing the waveform."""
    wav = np.abs(np.asarray(wav, dtype=np.float32))
    hop = max(sr // per_second, 1)
    n = int(np.ceil(len(wav) / hop))
    if n == 0:
        return []
    padded = np.zeros(n * hop, dtype=np.float32)
    padded[: len(wav)] = wav
    peaks = padded.reshape(n, hop).max(axis=1)
    top = float(np.percentile(peaks, 99.5)) or 1.0
    return [round(float(p), 3) for p in np.clip(peaks / top, 0, 1)]


def align_to_speech(
    lines: Iterable[tuple[int, float, float]], regions: list[tuple[float, float]], reach: float = 0.8
) -> dict[int, float]:
    """For each ``(index, start, end)`` line, where it should start to line up with the speech in the video.

    A line is matched to the region that overlaps its subtitle slot the most (looking ``reach`` seconds around
    it) and starts where that speech starts. Lines spoken without a clear pause share one region: the first of
    them takes its start, the next ones move by the same amount (keeping their spacing). Lines with no speech
    near them are left out.
    """
    starts: dict[int, float] = {}
    shift: dict[int, float] = {}  # region -> how far its first line moved
    for index, start, end in sorted(lines, key=lambda l: l[1]):
        best, best_score = None, 0.0
        for k, (a, b) in enumerate(regions):
            if b < start - reach or a > end + reach:
                continue
            overlap = max(0.0, min(b, end) - max(a, start))
            if overlap <= 0 and k in shift:
                continue  # speech near (not under) this line that another line already took
            score = overlap if overlap > 0 else 1e-3 / (1 + abs(a - start))
            if score > best_score:
                best, best_score = k, score
        if best is None:
            continue
        if best in shift:
            starts[index] = round(max(start + shift[best], regions[best][0]), 3)
        else:
            starts[index] = regions[best][0]
            shift[best] = regions[best][0] - start
    return starts


# -----------------------------
# The video
# -----------------------------


def prepare_video(source: str, video_dir: str | Path, isolate_voices: bool = False) -> dict:
    """Copy a video (or an audio file such as a .wav) into the project, make a browser-playable preview if
    needed and find the speech in it."""
    import soundfile as sf

    video_dir = Path(video_dir)
    if video_dir.exists():
        shutil.rmtree(video_dir)
    video_dir.mkdir(parents=True)
    src = Path(source)
    stored = video_dir / ("source" + src.suffix.lower())
    shutil.copy2(src, stored)
    preview = stored
    is_video = has_video_stream(str(stored))
    if not is_video and stored.suffix not in BROWSER_AUDIO:
        preview = video_dir / "preview.wav"
        _run([_require_binary("ffmpeg"), "-y", "-v", "error", "-i", str(stored), "-vn", str(preview)])
    elif is_video and stored.suffix not in BROWSER_VIDEO:
        preview = video_dir / "preview.mp4"
        _run([
            _require_binary("ffmpeg"), "-y", "-v", "error", "-i", str(stored), "-c:v", "libx264", "-preset",
            "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", str(preview),
        ])
    duration = probe_duration(str(stored))
    info = {
        "source": str(stored), "preview": str(preview), "name": src.name, "duration": round(duration, 3),
        "regions": [], "peaks": [], "has_audio": True, "isolated": False, "has_video": is_video,
    }
    analysis = video_dir / "audio16k.wav"
    try:
        extract_audio(str(stored), str(analysis), ANALYSIS_SR)
    except RuntimeError:
        info["has_audio"] = False
        return info
    wav, sr = sf.read(str(analysis), dtype="float32")
    info["peaks"] = waveform_peaks(wav, sr)
    info["regions"] = [list(r) for r in detect_speech(wav, sr)]
    ensure_subtitle_box(info)
    if isolate_voices:
        info.update(isolate_and_detect(info, video_dir))
    return info


def isolate_and_detect(info: dict, video_dir: str | Path, sensitivity: float = 0.5) -> dict:
    """Split the soundtrack into voices and music (Demucs) and find the speech in the voices only."""
    import soundfile as sf

    from .dubbing import separate_vocals

    video_dir = Path(video_dir)
    vocals, music = video_dir / "voices.wav", video_dir / "music.wav"
    if not (vocals.exists() and music.exists()):
        separate_vocals(info["source"], str(vocals), str(music))
    wav, sr = sf.read(str(vocals), dtype="float32")
    return {"regions": [list(r) for r in detect_speech(wav, sr, sensitivity=sensitivity)], "isolated": True,
            "music": str(music)}


def redetect(info: dict, video_dir: str | Path, sensitivity: float) -> list[list[float]]:
    import soundfile as sf

    video_dir = Path(video_dir)
    path = video_dir / ("voices.wav" if info.get("isolated") else "audio16k.wav")
    wav, sr = sf.read(str(path), dtype="float32")
    return [list(r) for r in detect_speech(wav, sr, sensitivity=sensitivity)]


# -----------------------------
# Subtitles burned into the picture
# -----------------------------

SUBTITLE_MODES = ("blur", "fill", "off")  # blur the area / fill it in from around it (delogo) / leave it


def probe_video(path: str | Path) -> dict:
    """Width, height, codec, picture bitrate and frame rate of a video's first picture stream."""
    proc = subprocess.run(
        [_require_binary("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,width,height,bit_rate,r_frame_rate,pix_fmt", "-of", "json", str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    stream = (json.loads(proc.stdout or "{}").get("streams") or [{}])[0]
    num, _, den = str(stream.get("r_frame_rate") or "30/1").partition("/")
    return {
        "codec": stream.get("codec_name"), "width": int(stream.get("width") or 0), "height": int(stream.get("height") or 0),
        "bit_rate": int(stream.get("bit_rate") or 0), "fps": float(num) / float(den or 1) if float(den or 1) else 30.0,
        "pix_fmt": stream.get("pix_fmt"),
    }


def _gray_frame(path: str, t: float, width: int, height: int) -> Optional[np.ndarray]:
    raw = subprocess.run(
        [_require_binary("ffmpeg"), "-v", "error", "-ss", f"{t:.2f}", "-i", path, "-frames:v", "1",
         "-vf", f"scale={width}:{height}", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout
    return np.frombuffer(raw, np.uint8).reshape(height, width).astype(np.float32) if len(raw) == width * height else None


def find_subtitle_box(frames: np.ndarray) -> Optional[list[float]]:
    """Where subtitles are burned into the picture, from grey frames taken while someone speaks: rows in the lower
    half with bright, sharp-edged strokes in most frames. ``[x, y, w, h]`` as fractions of the picture (centred
    and wide enough for longer lines), or ``None`` if there is no such band."""
    n, height, width = frames.shape
    if n < 3:
        return None
    gx = np.abs(np.diff(frames, axis=2))
    bright = frames[:, :, 1:] > 200
    text = gx * (bright | np.roll(bright, 1, axis=2))
    rows = np.percentile(text.mean(axis=2), 50, axis=0)  # text at these rows in at least half the frames
    low = int(height * 0.45)
    profile = rows.copy()
    profile[:low] = 0
    if profile.max() < 2.0:
        return None
    threshold = max(float(np.median(rows)) * 4, float(profile.max()) * 0.3)
    peak = int(profile.argmax())
    y0 = y1 = peak
    while y0 > low and profile[y0 - 1] > threshold:
        y0 -= 1
    while y1 < height - 1 and profile[y1 + 1] > threshold:
        y1 += 1
    cols = np.percentile(text[:, y0:y1 + 1, :].mean(axis=1), 90, axis=0)
    xs = np.where(cols > max(float(cols.max()) * 0.15, 1.0))[0]
    if not len(xs):
        return None
    band = (y1 - y0 + 1) / height
    pad_y = max(band * 0.6, 0.01)
    half = max(abs(xs.min() / width - 0.5), abs((xs.max() + 1) / width - 0.5)) * 1.12 + 0.03
    half = min(max(half, 0.3), 0.49)
    top = max(y0 / height - pad_y, 0.0)
    bottom = min((y1 + 1) / height + pad_y, 1.0)
    return [round(float(v), 4) for v in (0.5 - half, top, 2 * half, bottom - top)]


def ensure_subtitle_box(info: dict) -> bool:
    """Look for subtitles burned into a video's picture (once); True if ``info`` changed."""
    if "subtitle_box" in info or not info.get("has_video", True) or not info.get("source"):
        return False
    try:
        info["subtitle_box"] = detect_subtitle_box(info["source"], info.get("regions") or [])
    except Exception as exc:
        logger.warning("Looking for burned-in subtitles failed: %s", exc)
        info["subtitle_box"] = None
    return True


def detect_subtitle_box(video_path: str | Path, regions: list, samples: int = 30) -> Optional[list[float]]:
    """:func:`find_subtitle_box` on frames from the middle of the speech found in the video."""
    info = probe_video(video_path)
    if not info["width"] or not info["height"]:
        return None
    width = 270
    height = int(round(info["height"] * width / info["width"] / 2) * 2)
    times = [(a + b) / 2 for a, b in regions if b - a > 0.6]
    if len(times) > samples:
        times = [times[int(i * len(times) / samples)] for i in range(samples)]
    frames = [f for f in (_gray_frame(str(video_path), t, width, height) for t in times) if f is not None]
    return find_subtitle_box(np.stack(frames)) if len(frames) >= 3 else None


def _even(v: float) -> int:
    return max(int(round(v / 2)) * 2, 2)


def subtitle_filter(mode: str, box: list[float], width: int, height: int) -> Optional[str]:
    """The ffmpeg filter that hides ``box`` (fractions of the picture): ``blur`` softens it until the text is
    unreadable, ``fill`` paints it over from the pixels around it (delogo)."""
    if mode not in ("blur", "fill") or not box:
        return None
    x, y, w, h = box
    px, py = _even(x * width), _even(y * height)
    pw, ph = _even(w * width), _even(h * height)
    px, py = min(max(px, 2), width - 4), min(max(py, 2), height - 4)
    pw, ph = min(pw, width - px - 2), min(ph, height - py - 2)
    if mode == "fill":
        return f"[0:v]delogo=x={px}:y={py}:w={pw}:h={ph}[v]"
    sigma = max(8, ph // 3)
    return (f"[0:v]split=2[base][cut];[cut]crop={pw}:{ph}:{px}:{py},gblur=sigma={sigma}:steps=3[soft];"
            f"[base][soft]overlay={px}:{py}[v]")


def _encoders(codec: Optional[str]) -> list[list[str]]:
    """Ways to encode the cleaned picture in the source's codec, best first. The software encoders keep the
    picture practically identical at about the original size (x265 CRF 28: SSIM 0.99 on a 450 kb/s HEVC
    drama, 1.6x the file); the Mac's hardware encoder is faster but softer, so it is only the fallback."""
    if codec == "hevc":
        return [
            ["-c:v", "libx265", "-crf", "28", "-preset", "fast", "-x265-params", "log-level=error", "-tag:v", "hvc1"],
            ["-c:v", "hevc_videotoolbox", "-tag:v", "hvc1"],
        ]
    return [["-c:v", "libx264", "-crf", "20", "-preset", "fast"], ["-c:v", "h264_videotoolbox"]]


# -----------------------------
# Mix and export
# -----------------------------


LEVEL_TARGET_DB = -20.0  # where "Level voices" puts every clip (speech level, see speech_level)
LEVEL_TARGET_RANGE = (-30.0, -10.0)  # the loudness the editor lets you choose
LEVEL_RANGE_DB = 24.0  # it never turns a clip up or down by more than this
# How strongly a line is evened out from the inside: (ratio, most dB a word is turned up or down).
LEVEL_STRENGTHS = {"off": (1.0, 0.0), "light": (2.0, 6.0), "normal": (3.0, 9.0), "strong": (6.0, 12.0)}


def level_options(level) -> Optional[dict]:
    """``{"target": dB, "strength": name}`` for "Level voices", or ``None`` when it is off. ``level`` is ``True`` /
    ``False`` or such a dict (missing keys take the defaults; the editor saves ``mix.level_target`` and
    ``mix.level_strength``)."""
    if not level and not isinstance(level, dict):  # {} means "on, with the defaults"
        return None
    opts = level if isinstance(level, dict) else {}
    try:
        target = float(opts.get("target", LEVEL_TARGET_DB))
    except (TypeError, ValueError):
        target = LEVEL_TARGET_DB
    strength = opts.get("strength") if opts.get("strength") in LEVEL_STRENGTHS else "normal"
    return {"target": float(np.clip(target, *LEVEL_TARGET_RANGE)), "strength": strength}


def _biquad(b, a, x: np.ndarray) -> np.ndarray:
    from scipy.signal import lfilter

    return lfilter(np.asarray(b) / a[0], np.asarray(a) / a[0], x)


def k_weighted(wav: np.ndarray, sr: int) -> np.ndarray:
    """``wav`` through the ITU-R BS.1770 "K" filter (a +4 dB lift above ~1.7 kHz and a cut below ~38 Hz), so its
    power follows how loud it sounds: a deep voice is not counted louder than it sounds, nor a bright shout quieter.
    (The editor filters the same way.)"""
    x = np.asarray(wav, dtype=np.float64)
    if not len(x):
        return x
    A = 10 ** (3.99984385397 / 40)
    w0 = 2 * np.pi * 1681.9744509555319 / sr
    cos, alpha = np.cos(w0), np.sin(w0) / (2 * 0.7071752369554193)
    root = 2 * np.sqrt(A) * alpha
    x = _biquad(
        [A * ((A + 1) + (A - 1) * cos + root), -2 * A * ((A - 1) + (A + 1) * cos), A * ((A + 1) + (A - 1) * cos - root)],
        [(A + 1) - (A - 1) * cos + root, 2 * ((A - 1) - (A + 1) * cos), (A + 1) - (A - 1) * cos - root],
        x,
    )
    w0 = 2 * np.pi * 38.13547087613982 / sr
    cos, alpha = np.cos(w0), np.sin(w0) / (2 * 0.5003270373253953)
    return _biquad([(1 + cos) / 2, -(1 + cos), (1 + cos) / 2], [1 + alpha, -2 * cos, 1 - alpha], x)


def speech_level(wav: np.ndarray, sr: int, frame: float = 0.05, weighted: bool = True) -> Optional[float]:
    """How loud the speech in ``wav`` sounds, in dB: the mean power (after :func:`k_weighted`, unless
    ``weighted=False``) of its 50 ms frames that are within 30 dB of its loudest one, so pauses and quiet tails do
    not count. ``None`` for silence. (The editor measures the same way.)"""
    wav = k_weighted(wav, sr) if weighted else np.asarray(wav, dtype=np.float64)
    n = max(int(sr * frame), 1)
    count = len(wav) // n
    if count == 0:
        power = np.array([np.mean(wav**2)]) if len(wav) else np.array([0.0])
    else:
        power = np.mean(wav[: count * n].reshape(count, n) ** 2, axis=1)
    db = 10 * np.log10(power + 1e-12)
    voiced = db > max(float(db.max()) - 30.0, -60.0)
    if not voiced.any():
        return None
    return float(10 * np.log10(np.mean(power[voiced])))


EVEN_FRAME = 0.05  # seconds between points of the evening-out curve
EVEN_WINDOW = 0.3  # each point looks at this much audio around it
EVEN_RATIO = 3.0  # "normal": words 9 dB louder than the line end up 3 dB louder
EVEN_MAX_DB = 9.0


def evening_curve(wav: np.ndarray, sr: int, strength: str = "normal") -> np.ndarray:
    """Gain in dB every :data:`EVEN_FRAME` s that evens out a line from the inside: words that sound louder than the
    line's speech level are turned down and quieter ones up (ratio and limit from :data:`LEVEL_STRENGTHS`);
    silence and breaths are left alone. Point ``i`` is at ``(i + 0.5) * EVEN_FRAME`` s. (The editor computes the
    same curve, so the preview sounds like the export.)"""
    ratio, most = LEVEL_STRENGTHS.get(strength, LEVEL_STRENGTHS["normal"])
    weighted = k_weighted(wav, sr)
    hop, half = max(int(round(sr * EVEN_FRAME)), 1), max(int(round(sr * EVEN_WINDOW / 2)), 1)
    count = int(np.ceil(len(weighted) / hop))
    line = speech_level(weighted, sr, weighted=False)
    if not count or line is None or most <= 0:
        return np.zeros(max(count, 1))
    squares = np.concatenate([[0.0], np.cumsum(weighted * weighted)])
    gains = np.zeros(count)
    for i in range(count):
        c = int((i + 0.5) * hop)
        a, b = max(c - half, 0), min(c + half, len(weighted))
        power = (squares[b] - squares[a]) / max(b - a, 1)
        db = 10 * np.log10(power + 1e-12)
        if db > max(line - 20.0, -60.0):
            gains[i] = np.clip((line - db) * (1 - 1 / ratio), -most, most)
    k = 5  # smooth over 0.25 s so the volume glides
    padded = np.concatenate([np.full(k // 2, gains[0]), gains, np.full(k // 2, gains[-1])])
    return np.convolve(padded, np.ones(k) / k, mode="valid")


def even_out(wav: np.ndarray, sr: int, strength: str = "normal") -> np.ndarray:
    """``wav`` with :func:`evening_curve` applied."""
    curve = evening_curve(wav, sr, strength)
    times = (np.arange(len(curve)) + 0.5) * EVEN_FRAME
    gain_db = np.interp(np.arange(len(wav)) / sr, times, curve)
    return (np.asarray(wav, dtype=np.float32) * (10 ** (gain_db / 20)).astype(np.float32)).astype(np.float32)


def level_gain_db(wav: np.ndarray, sr: int, target: float = LEVEL_TARGET_DB) -> float:
    level = speech_level(wav, sr)
    return 0.0 if level is None else float(np.clip(target - level, -LEVEL_RANGE_DB, LEVEL_RANGE_DB))


def soft_limit(wav: np.ndarray, knee: float = 0.8) -> np.ndarray:
    """Leaves everything below ``knee`` alone and bends louder peaks smoothly under 1.0 (no clipping, and no
    turning the whole clip down, which would undo the levelling)."""
    wav = np.asarray(wav, dtype=np.float32)
    over = np.abs(wav) > knee
    if not over.any():
        return wav
    out = wav.copy()
    mag = np.abs(wav[over])
    out[over] = np.sign(wav[over]) * (knee + (1 - knee) * np.tanh((mag - knee) / (1 - knee)))
    return out


def clip_audio(wav: np.ndarray, sr: int, clip: dict, level=False) -> np.ndarray:
    """The part of a line's audio the clip keeps (after trimming), at the clip's speed and volume.

    ``trim_in`` / ``trim_out`` are seconds of the line's own audio; at speed 2 the clip lasts half as long. With
    ``level`` ("Level voices": ``True`` or :func:`level_options` settings), the line is first evened out from the
    inside (:func:`even_out`) and the part is brought to the target loudness, so every clip sounds as loud as the
    others, word for word; the clip's own volume is then a change on top of that."""
    opts = level_options(level)
    a = int(round(max(float(clip.get("trim_in") or 0), 0) * sr))
    b = len(wav) - int(round(max(float(clip.get("trim_out") or 0), 0) * sr))
    source = even_out(wav, sr, opts["strength"]) if opts else np.asarray(wav, dtype=np.float32)
    part = change_speed(source[a:max(b, a)], sr, clip_speed(clip))
    gain = float(clip.get("gain_db") or 0) + (level_gain_db(part, sr, opts["target"]) if opts else 0.0)
    part = part * np.float32(10 ** (gain / 20))
    return soft_limit(part) if opts else part


SECTION_SOUNDS = ("auto", "full", "music", "mute")  # as set for the whole video / full original / voices removed / silent


def _fit(wav: Optional[np.ndarray], n: int) -> Optional[np.ndarray]:
    if wav is None:
        return None
    out = np.zeros(n, dtype=np.float32)
    m = min(n, len(wav))
    out[:m] = np.asarray(wav[:m], dtype=np.float32)
    return out


def render_mix(
    clips: list[tuple[dict, np.ndarray]],
    total_seconds: float,
    sr: int,
    original: Optional[np.ndarray] = None,
    mode: str = "duck",
    original_gain_db: float = 0.0,
    vocals_gain_db: float = 0.0,
    duck_db: float = -15.0,
    regions: Optional[list] = None,
    level=False,
    sections: Optional[list[dict]] = None,
    full: Optional[np.ndarray] = None,
    music: Optional[np.ndarray] = None,
    fade: float = 0.02,
) -> np.ndarray:
    """One soundtrack: every unmuted clip at its place, over the original soundtrack (kept, ducked under the
    clips, silenced where they replace it, or left out). For ``mode="music"`` pass the soundtrack without voices
    as ``original``; for ``mode="replace"`` pass the speech found in the video as ``regions``.

    ``sections`` (``[{"start", "end", "gain_db", "sound"}]``, the original sound cut into parts in the editor) give
    each part of the original its own volume and sound: ``auto`` (as above), ``full`` (the whole original soundtrack,
    voices too - pass it as ``full``), ``music`` (voices removed - pass ``music``) or ``mute``. Parts meet with
    short ``fade`` s crossfades."""
    vocals_gain = np.float32(10 ** (vocals_gain_db / 20))
    placed, spans = [], []
    for clip, wav in clips:
        if clip.get("muted"):
            continue
        part = clip_audio(wav, sr, clip, level=level) * vocals_gain
        if len(part):
            placed.append((float(clip["start"]), part))
            spans.append((float(clip["start"]), float(clip["start"]) + len(part) / sr))
    background = None
    if original is not None and mode != "mute":
        background = np.asarray(original, dtype=np.float32) * np.float32(10 ** (original_gain_db / 20))
    duck_spans, bg_mode = spans, "duck" if mode in ("duck", "replace") else "mute" if mode == "mute" else "keep"
    if mode == "replace":
        duck_spans, duck_db = replaced_spans(spans, regions), SILENT_DB
    if not sections:
        return mix_timeline(placed, total_seconds, sr, background=background, background_mode=bg_mode,
                            duck_db=duck_db, speech_regions=duck_spans)
    n = int(round(total_seconds * sr))
    gain = np.float32(10 ** (original_gain_db / 20))
    sources = {
        "auto": mix_timeline([], total_seconds, sr, background=background, background_mode=bg_mode, duck_db=duck_db,
                             speech_regions=duck_spans) if background is not None else None,
        "full": None if full is None else _fit(full, n) * gain,
        "music": None if music is None else _fit(music, n) * gain,
    }
    weights = {k: np.zeros(n, dtype=np.float32) for k in sources}
    for sec in sections:
        sound = sec.get("sound") or "auto"
        if sound not in weights:
            continue  # "mute": nothing
        a, b = max(int(float(sec["start"]) * sr), 0), min(int(float(sec["end"]) * sr), n)
        if b > a:
            weights[sound][a:b] = 10 ** (float(sec.get("gain_db") or 0) / 20)
    k = max(int(fade * sr), 1)
    bg = np.zeros(n, dtype=np.float32)
    for name, src in sources.items():
        if src is not None and weights[name].any():
            w = np.convolve(weights[name], np.ones(k, dtype=np.float32) / k, mode="same")
            bg += _fit(src, n) * w
    return mix_timeline(placed, total_seconds, sr, background=bg, background_mode="keep")


def read_soundtrack(media_path: str, sr: int, out_wav: str | Path) -> np.ndarray:
    import soundfile as sf

    extract_audio(media_path, str(out_wav), sr)
    wav, _ = sf.read(str(out_wav), dtype="float32")
    return wav


def subtitle_overlays(source: str, overlays: list[tuple[float, float, int, int, str]]) -> tuple[list[str], str, str]:
    """ffmpeg inputs and filters that draw subtitle images over the picture labelled ``source``: each
    ``(start, end, x, y, png)`` is shown at ``x, y`` from ``start`` to ``end`` seconds. The images are made by the
    editor in the browser (it lays out Khmer and other complex scripts correctly), so the export looks exactly like
    the preview. Inputs start at index 2 (0 = video, 1 = audio). Returns (inputs, filters, last label)."""
    inputs, filters, label = [], [], source
    for i, (start, end, x, y, png) in enumerate(overlays):
        inputs += ["-i", str(png)]
        out = f"s{i}"
        filters.append(
            f"[{label}][{i + 2}:v]overlay={int(x)}:{int(y)}:eof_action=repeat:"
            f"enable='between(t,{float(start):.3f},{float(end):.3f})'[{out}]"
        )
        label = out
    return inputs, ";".join(filters), label


def export_video(
    video_path: str,
    audio_path: str,
    out_path: str | Path,
    subtitles: Optional[dict] = None,
    overlays: Optional[list[tuple[float, float, int, int, str]]] = None,
) -> str:
    """The video with ``audio_path`` as its soundtrack. The picture is copied as it is, unless ``subtitles``
    (``{"mode": "blur" | "fill", "box": [x, y, w, h]}``) asks to hide the subtitles burned into it, or ``overlays``
    (see :func:`subtitle_overlays`) draws new ones: then it is re-encoded at the original size, codec and frame
    rate, at a quality that keeps it practically identical."""
    out_path = str(out_path)
    info = probe_video(video_path)
    graph, label = [], "0:v"
    if subtitles and subtitles.get("mode") in ("blur", "fill") and subtitles.get("box"):
        cleanup = subtitle_filter(subtitles["mode"], subtitles["box"], info["width"], info["height"])
        if cleanup:
            graph.append(cleanup.replace("[v]", "[clean]"))
            label = "clean"
    inputs = []
    if overlays:
        inputs, drawn, label = subtitle_overlays(label, overlays)
        graph.append(drawn)
    if graph:
        rate = max(info["bit_rate"] * 2.5, 1_500_000) if info["bit_rate"] else 6_000_000
        base = [_require_binary("ffmpeg"), "-y", "-v", "error", "-i", video_path, "-i", audio_path, *inputs,
                "-filter_complex", ";".join(graph), "-map", f"[{label}]", "-map", "1:a:0", "-shortest",
                "-map_metadata", "0", "-pix_fmt", "yuv420p", "-fps_mode", "passthrough", "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart"]
        last = None
        for encoder in _encoders(info["codec"]):
            extra = ["-b:v", str(int(rate))] if "-crf" not in encoder else []
            try:
                _run(base + encoder + extra + [out_path])
                return out_path
            except RuntimeError as exc:
                last = exc
                logger.info("Encoder %s failed, trying the next one", encoder[1])
        raise last
    base = [_require_binary("ffmpeg"), "-y", "-v", "error", "-i", video_path, "-i", audio_path, "-map", "0:v:0",
            "-map", "1:a:0", "-shortest"]
    try:
        _run(base + ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k", out_path])
    except RuntimeError:
        logger.info("Copying the picture failed; re-encoding it")
        _run(base + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac",
                     "-b:a", "192k", out_path])
    return out_path


def export_extension(video_path: str) -> str:
    return ".mp4" if Path(video_path).suffix.lower() in (".mp4", ".m4v", ".mov") else ".mkv"


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


__all__ = [
    "ORIGINAL_MODES",
    "SECTION_SOUNDS",
    "replaced_spans",
    "align_to_speech",
    "clip_audio",
    "even_out",
    "evening_curve",
    "level_gain_db",
    "level_options",
    "k_weighted",
    "speech_level",
    "change_speed",
    "clip_line",
    "clip_speed",
    "collapse_pieces",
    "default_clip",
    "detect_speech",
    "detect_subtitle_box",
    "ensure_subtitle_box",
    "find_subtitle_box",
    "subtitle_filter",
    "subtitle_overlays",
    "export_video",
    "load_state",
    "prepare_video",
    "redetect",
    "render_mix",
    "save_state",
    "waveform_peaks",
]
