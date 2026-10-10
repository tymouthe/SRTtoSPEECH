"""Voice a whole SRT script with VoxCPM, without any video or original audio.

Speakers, gender, age and emotion come from tags in the subtitles::

    [Dara|male|kid|sad] ម៉ាក់ កូននឹកម៉ាក់ណាស់
    [Srey|female|adult] រៀបការអស់រយៈពេល ៧ ឆ្នាំ
    [Dara] ...                  <- gender/age only need to be given once per speaker

Every part after the name is optional and may come in any order: ``male``/``female`` (also ``man``,
``woman``, ``boy``, ``girl``), ``adult``/``kid`` (also ``child``) and an emotion (``happy``, ``sad``,
``angry``, ``fearful``, ``disgusted``, ``surprised``, ``neutral`` or any other word, used as written).
A missing emotion is guessed from the punctuation and (English) keywords, and :func:`format_tagged_srt`
writes the result back as a tagged SRT that can be checked, edited and used again.

Each speaker's voice is designed once from their gender and age, as a calm, neutral reading of their
first line(s), retried until its pitch is in the range for that gender and age; every line of the speaker
clones that reading, adds its own emotion, and is retried until it is consistent with it.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

from .dubbing import (
    DubOptions,
    SpeakerProfile,
    SubtitleLine,
    VideoDubber,
    _centroid,
    fit_to_slot,
    loop_to_length,
    remove_silences,
    format_srt,
    line_style,
    load_srt,
    mix_timeline,
    parse_srt,
    seconds_to_srt_time,
)

logger = logging.getLogger(__name__)

AGES = ("adult", "kid")
DEFAULT_SPEAKER = "Narrator"

_GENDER_WORDS = {"male": "male", "m": "male", "man": "male", "female": "female", "f": "female", "woman": "female"}
_AGE_WORDS = {"adult": "adult", "kid": "kid", "child": "kid", "children": "kid"}
_KID_WORDS = {"boy": "male", "girl": "female"}

VOICE_DESCRIPTIONS = {
    ("male", "adult"): "Adult man's voice, clearly masculine, deep and resonant, low pitch",
    ("female", "adult"): "Adult woman's voice, clearly feminine, warm and clear, medium-high pitch",
    ("unknown", "adult"): "Natural adult voice",
    ("male", "kid"): "Young boy of about ten, clearly a boy, boyish and energetic, slightly husky, not girlish",
    ("female", "kid"): "Young girl's voice, a child of about eight, high and bright",
    ("unknown", "kid"): "Young child's voice, a child of about eight, high and bright",
}
# Median F0 (Hz) a speaker's voice must land in to sound clearly like that gender and age: kept well away
# from the ~165 Hz point where adult male and female voices meet.
PITCH_RANGES = {
    ("male", "adult"): (90.0, 140.0),
    ("female", "adult"): (185.0, 255.0),
    ("unknown", "adult"): (85.0, 255.0),
    ("male", "kid"): (195.0, 280.0),  # a boy at the high end sounds like a girl: kept lower
    ("female", "kid"): (220.0, 400.0),
    ("unknown", "kid"): (200.0, 400.0),
}


# Hard bounds every line must stay within (emotion may move the pitch, but never across to the other gender).
PITCH_LIMITS = {
    ("male", "adult"): (70.0, 160.0),
    ("female", "adult"): (175.0, 330.0),
    ("male", "kid"): (200.0, 460.0),
    ("female", "kid"): (210.0, 460.0),
    ("unknown", "kid"): (200.0, 460.0),
}


def pitch_limits(gender: str, age: str) -> Optional[tuple[float, float]]:
    return PITCH_LIMITS.get((gender, age))


def voice_description(gender: str, age: str) -> str:
    return VOICE_DESCRIPTIONS.get((gender, age), VOICE_DESCRIPTIONS[("unknown", "adult")])


def pitch_range(gender: str, age: str) -> tuple[float, float]:
    return PITCH_RANGES.get((gender, age), PITCH_RANGES[("unknown", "adult")])


# -----------------------------
# Tags
# -----------------------------


@dataclass
class VoiceTag:
    name: Optional[str]
    gender: Optional[str] = None
    age: Optional[str] = None
    emotion: Optional[str] = None


def parse_voice_tag(tag: Optional[str]) -> VoiceTag:
    """``"Dara|male|kid|sad"`` -> ``VoiceTag("Dara", "male", "kid", "sad")``; parts after the name are optional."""
    if not tag:
        return VoiceTag(None)
    parts = [p.strip() for p in re.split(r"[|/]", tag) if p.strip()]
    if not parts:
        return VoiceTag(None)
    result = VoiceTag(parts[0])
    for part in parts[1:]:
        word = part.lower()
        if word in _GENDER_WORDS:
            result.gender = _GENDER_WORDS[word]
        elif word in _AGE_WORDS:
            result.age = _AGE_WORDS[word]
        elif word in _KID_WORDS:
            result.gender, result.age = _KID_WORDS[word], "kid"
        else:
            result.emotion = word
    return result


_EMOTION_KEYWORDS = {
    "happy": ("haha", "yay", "great", "wonderful", "love", "glad", "happy", "thank", "awesome", "congrat"),
    "sad": ("sorry", "miss you", "cry", "sad", "alone", "lost", "goodbye", "died", "dead", "tears"),
    "angry": ("shut up", "how dare", "damn", "hate", "get out", "idiot", "stupid", "enough"),
    "fearful": ("help", "scared", "afraid", "please don't", "run", "ghost", "no no"),
    "surprised": ("what?!", "really?", "oh my", "wow", "no way", "seriously"),
}


def guess_emotion(text: str) -> str:
    """A rough emotion guess from the text alone: English keywords, then punctuation (any language)."""
    lowered = text.lower()
    for emotion, words in _EMOTION_KEYWORDS.items():
        if any(w in lowered for w in words):
            return emotion
    stripped = text.strip()
    if re.search(r"[?？][!！]|[!！][?？]", stripped):
        return "surprised"
    if stripped.endswith(("!", "！")):
        return "excited"
    if stripped.endswith(("...", "…", "។។")):
        return "hesitant"
    return "neutral"


def read_script(lines: list[SubtitleLine], guess_missing_emotion: bool = True) -> tuple[list[SubtitleLine], dict]:
    """Apply the voice tags of parsed SRT lines (in place) and build one profile per speaker.

    Untagged lines belong to :data:`DEFAULT_SPEAKER`. A speaker's gender and age come from the first line
    that gives them; emotion is per line (guessed when missing, marked in ``features["emotion_guessed"]``).
    """
    known: dict[str, dict] = {}
    for line in lines:
        tag = parse_voice_tag(line.speaker)
        line.speaker = tag.name or DEFAULT_SPEAKER
        info = known.setdefault(line.speaker, {"gender": None, "age": None})
        info["gender"] = info["gender"] or tag.gender
        info["age"] = info["age"] or tag.age
        if tag.emotion:
            line.emotion = tag.emotion
            line.features["emotion_tagged"] = True
        elif guess_missing_emotion:
            line.emotion = guess_emotion(line.text)
            line.features["emotion_guessed"] = True
        else:
            line.emotion = "neutral"
    profiles = {}
    for name, info in known.items():
        gender, age = info["gender"] or "unknown", info["age"] or "adult"
        for line in lines:
            if line.speaker == name:
                line.gender = gender
        profiles[name] = SpeakerProfile(
            name=name, gender=gender, age=age, voice_mode="clone_first", description=voice_description(gender, age)
        )
    return lines, profiles


def load_script(path: str | os.PathLike, guess_missing_emotion: bool = True) -> tuple[list[SubtitleLine], dict]:
    return read_script(load_srt(path), guess_missing_emotion)


_ARROW_TIME_RE = re.compile(r"^\s*\d+:\d{1,2}:\d{1,2}[,.]\d{1,3}\s*-->\s*\d+:\d{1,2}:\d{1,2}[,.]\d{1,3}\s*$")


def find_srt_problems(path: str | os.PathLike) -> list[str]:
    """Blocks of an SRT that would be skipped (no readable ``start --> end`` timing line), and voice tags that are
    not read as meant (not recognised at all, or with more than one word that is not a gender or age), as messages."""
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "utf-16", "gb18030", "latin-1"):
        try:
            content = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    problems = []
    for block in re.split(r"\n\s*\n", content.strip()):
        rows = [r for r in block.split("\n") if r.strip()]
        if rows and any(_ARROW_TIME_RE.match(r) for r in rows):
            problems.extend(_tag_problems(rows))
            continue
        if not rows:
            continue
        number = rows[0].strip() if rows[0].strip().isdigit() else "?"
        timing = next((r.strip() for r in rows if ":" in r and "," in r and not r.strip().startswith("[")), None)
        text = next((r.strip() for r in reversed(rows) if r.strip() != number and r.strip() != timing), "")
        problems.append(
            f"Block {number} was skipped: its timing line "
            + (f"'{timing}' " if timing else "")
            + "is not 'HH:MM:SS,mmm --> HH:MM:SS,mmm'"
            + (f" (text: {text[:40]}…)" if text else "")
        )
    return problems


def _tag_problems(rows: list[str]) -> list[str]:
    from .dubbing import split_speaker_tag

    number = rows[0].strip() if rows[0].strip().isdigit() else "?"
    text = " ".join(r.strip() for r in rows if r.strip() != number and not _ARROW_TIME_RE.match(r))
    if not text.startswith(("[", "【")):
        return []
    name, _ = split_speaker_tag(text)
    if name is None:
        tag = text[: text.find("]") + 1] if "]" in text else text[:40]
        return [f"Line {number}: its tag {tag} could not be read, so the line is read by {DEFAULT_SPEAKER}."]
    parts = [p.strip() for p in re.split(r"[|/]", name) if p.strip()][1:]
    known = set(_GENDER_WORDS) | set(_AGE_WORDS) | set(_KID_WORDS)
    others = [p for p in parts if p.lower() not in known]
    if len(others) > 1:
        return [
            f"Line {number}: '{'|'.join(others)}' — after the name only male/female, adult/kid and one emotion are "
            f"understood, so '{others[-1]}' is used as the emotion and the rest is ignored."
        ]
    return []


def format_tagged_srt(lines: list[SubtitleLine], profiles: dict[str, SpeakerProfile]) -> str:
    """SRT with a full ``[Name|gender|age|emotion]`` tag on every line, ready to edit and voice again."""
    out = []
    for i, line in enumerate(lines, 1):
        profile = profiles.get(line.speaker)
        parts = [line.speaker or DEFAULT_SPEAKER]
        if profile is not None and profile.gender in ("male", "female"):
            parts.append(profile.gender)
        if profile is not None:
            parts.append(profile.age or "adult")
        parts.append(line.emotion or "neutral")
        out.append(
            f"{i}\n{seconds_to_srt_time(line.start)} --> {seconds_to_srt_time(line.end)}\n[{'|'.join(parts)}] {line.text}\n"
        )
    return "\n".join(out)


# -----------------------------
# Tone from the original video
# -----------------------------


def tones_from_audio(
    lines: list[SubtitleLine],
    audio: np.ndarray,
    sr: int,
    clip_dir: str | os.PathLike,
    regions: Optional[list] = None,
    emotion_model=None,
    music_removed: bool = False,
    read_emotion: bool = True,
) -> dict[int, dict]:
    """Read each line's emotion and delivery from the original speech at its SRT time (``audio``: the video's
    soundtrack, or its separated voices, mono). Returns ``{index: {"emotion", "tone"}}`` for the lines that have
    speech there (with ``regions``, the speech found in the video, a line needs some of it under its time).

    The emotion comes from SenseVoice; the tone ("louder and more forceful", "speaking faster", …) compares the
    line with how that speaker usually sounds in the video (:func:`voxcpm.dubbing.describe_tone`).

    SenseVoice hears music under speech as anger, so where it reports background music the emotion is left out
    (``"music": True``) unless the voices were separated from the music first (``music_removed``). Its labels are
    coarse (any tense, raised voice is "angry"), so ``read_emotion=False`` reads only the tone."""
    import soundfile as sf

    from .dubbing import (
        _segment,
        _speech_units,
        describe_tone,
        load_emotion_model,
        measure_line,
        parse_sensevoice_emotion,
        speaker_baseline,
    )

    probes = []
    for line in lines:
        if line.duration <= 0.2:
            continue
        if regions is not None:
            heard = sum(max(0.0, min(b, line.end) - max(a, line.start)) for a, b in regions)
            if heard < max(0.15, 0.25 * line.duration):
                continue  # no speech in the video at this line's time
        seg = _segment(audio, sr, line.start, line.end)
        probe = replace(line, features=measure_line(seg, sr))
        units = _speech_units(line.text)
        probe.features["rate"] = units / line.duration if line.duration > 0.3 and units else None
        probes.append(probe)
    found = {}
    for name in dict.fromkeys(p.speaker for p in probes):
        own = [p for p in probes if p.speaker == name]
        baseline = speaker_baseline(own)
        for probe in own:
            found[probe.index] = {"emotion": "", "tone": describe_tone(probe, baseline), "music": False}

    if not read_emotion:
        return found
    clip_dir = Path(clip_dir)
    clip_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for probe in probes:
        path = clip_dir / f"line_{probe.index:04d}.wav"
        sf.write(str(path), _segment(audio, sr, probe.start, probe.end, pad=0.1), sr)
        paths.append(str(path))
    try:
        model = emotion_model or load_emotion_model()
        results = model.generate(input=paths, language="auto", use_itn=False) if paths else []
        heard = {str(r.get("key")): r.get("text", "") for r in results}
        for probe, path in zip(probes, paths):
            raw = heard.get(Path(path).stem, "")
            if "<|BGM|>" in raw and not music_removed:
                found[probe.index]["music"] = True  # its "emotion" is mostly the music's
                continue
            found[probe.index]["emotion"] = parse_sensevoice_emotion(raw)
    except Exception as exc:  # the tone still helps without the emotion
        logger.warning("Emotion detection from the video skipped: %s", exc)
    return found


def apply_tones(lines: list[SubtitleLine], tones: dict[int, dict]) -> int:
    """Use the emotion and tone read from the video (:func:`tones_from_audio`) for each line (in place).

    An emotion written in the line's tag always wins; a detected emotion replaces the guess from punctuation
    (a "neutral" reading keeps the guess). Returns how many lines got something from the video."""
    used = 0
    for line in lines:
        found = tones.get(line.index)
        if not found:
            continue
        used += 1
        line.tone = found.get("tone") or ""
        emotion = (found.get("emotion") or "").strip().lower()
        if emotion and emotion != "neutral" and not line.features.get("emotion_tagged"):
            line.emotion = emotion
            line.features.pop("emotion_guessed", None)
            line.features["emotion_from_video"] = True
    return used


# -----------------------------
# Generation
# -----------------------------


def pad_with_silence(wav: np.ndarray, sr: int, seconds: float) -> np.ndarray:
    """Lengthen ``wav`` to ``seconds`` by adding silence after the speech; longer clips are left as they are."""
    missing = int(round(seconds * sr)) - len(wav)
    if missing <= 0:
        return wav
    return np.concatenate([wav, np.zeros(missing, dtype=np.float32)]).astype(np.float32)


@dataclass
class ScriptResult:
    audio_path: str
    lines_dir: str
    tagged_srt_path: str
    report: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    voices: dict[str, str] = field(default_factory=dict)  # each speaker's voice clip (what their lines copy)


class ScriptVoicer(VideoDubber):
    """Voices SRT lines with one consistent, designed voice per speaker (no video or original audio)."""

    def speaker_voice(
        self, profile: SpeakerProfile, own: list[SubtitleLine], options: DubOptions, min_seconds: float = 4.0
    ) -> Optional[str]:
        """Design a speaker's voice once: a calm, neutral reading of their first line, joined with their next
        lines (up to 3) until it covers ``min_seconds`` - a short clip is too little to clone a voice from.
        Retried until its pitch is in the range for the speaker's gender and age."""
        picked, seconds = [], 0.0
        for line in own:
            picked.append(line)
            seconds += line.duration
            if seconds >= min_seconds or len(picked) >= 3:
                break
        sample = SubtitleLine(
            index=picked[0].index,
            start=0.0,
            end=max(seconds, options.short_line_seconds + 0.5),  # never treated as a short line
            text=" ".join(l.text for l in picked),
            speaker=profile.name,
            gender=profile.gender,
            emotion="calm and neutral",
        )
        low, high = pitch_range(profile.gender, profile.age)
        best = self._generate_best(
            sample, profile, "design", None, None, options,
            attempts=options.first_line_attempts, pitch_range=(low, high),
        )
        path = self._write_reference(f"voice_{profile.name}", best["wav"], self.sample_rate)
        if path is None or not best["voice_ok"]:
            message = (
                f"{profile.name}'s voice (a neutral reading of line #{sample.index}) "
                + ("came out too short to clone; their lines are designed one by one and may vary."
                   if path is None else
                   f"is not in the {profile.gender} {profile.age} pitch range ({low:.0f}-{high:.0f} Hz) after "
                   f"{best['tries']} tries; all their lines copy it. Change the voice description or the seed, "
                   "or give this speaker a reference clip.")
            )
            logger.warning(message)
            self.warnings.append(message)
        return path

    def voice(
        self,
        lines: list[SubtitleLine],
        profiles: dict[str, SpeakerProfile],
        output_path: str,
        options: Optional[DubOptions] = None,
        progress: Optional[Callable[[int, int, SubtitleLine], None]] = None,
        only: Optional[Iterable[int]] = None,
        new_voices: Iterable[str] = (),
    ) -> ScriptResult:
        """Voice every line, or with ``only`` (line numbers) regenerate just those lines of an earlier run in the
        same ``workdir``: same speaker voices, new seeds, and the combined track is rebuilt from all lines.

        ``new_voices`` (speaker names) designs a new voice for those speakers - from their current gender, age
        and description, with a new seed - and regenerates all of their lines with it."""
        import soundfile as sf

        options = options or DubOptions()
        lines = sorted(lines, key=lambda l: l.start)
        if not lines:
            raise ValueError("The script has no lines.")
        for line in lines:
            if line.speaker not in profiles:
                profiles[line.speaker] = SpeakerProfile(
                    name=line.speaker, gender=line.gender, voice_mode="clone_first",
                    description=voice_description(line.gender, "adult"),
                )
        sr = self.sample_rate
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        stem = out.with_suffix("")
        lines_dir = Path(str(stem) + "_lines")
        lines_dir.mkdir(exist_ok=True)
        raw_dir = self.workdir / "raw"  # every line at full speed, as generated (for rebuilding the track)
        raw_dir.mkdir(exist_ok=True)
        state_path = self.workdir / "script_state.json"

        new_voices = {n for n in new_voices if n}
        unknown = new_voices - {l.speaker for l in lines}
        if unknown:
            raise ValueError(f"No lines for speaker(s) {', '.join(sorted(unknown))} in the script.")
        if new_voices:
            only = set(only or ()) | {l.index for l in lines if l.speaker in new_voices}
        only = None if only is None else {int(i) for i in only}
        state = {"refs": {}, "entries": {}, "rounds": {}, "voice_rounds": {}}
        if only is not None:
            # Some lines only (one speaker's, failed ones, …): on top of what was made before, if anything.
            if state_path.exists():
                state = json.loads(state_path.read_text(encoding="utf-8"))
            state.setdefault("voice_rounds", {})
            missing = only - {l.index for l in lines}
            if missing:
                raise ValueError(f"No line numbered {', '.join(map(str, sorted(missing)))} in the script.")
        firsts: dict[str, SubtitleLine] = {}
        for line in lines:
            firsts.setdefault(line.speaker, line)

        # One reference voice per speaker, made before any line: a calm, neutral reading of their first
        # line(s). Every line of the speaker - the first one too - clones it and adds its own emotion, so no
        # line's emotion (a happy first line, say) is copied into all the others. A regeneration reuses them.
        refs: dict[str, str] = {
            n: path for n, path in state["refs"].items()
            if n in profiles and n not in new_voices and Path(path).exists()
        }
        for name, profile in profiles.items():
            if profile.reference_wav:
                # A voice template (or your own clip): every line copies it; "new voice" redoes their lines with it.
                refs[name] = profile.reference_wav
            elif name not in refs and name in firsts and (only is None or any(
                l.speaker == name and l.index in only for l in lines
            )):
                if progress:
                    progress(0, len(lines), firsts[name])
                # A new voice for a speaker is designed with a new seed (round 0 is the first voice).
                voice_round = int(state["voice_rounds"].get(name, -1)) + 1 if name in new_voices else int(
                    state["voice_rounds"].get(name, 0)
                )
                state["voice_rounds"][name] = voice_round
                voice_options = replace(options, seed=(options.seed or 0) + voice_round * 7_368_787)
                ref = self.speaker_voice(profile, [l for l in lines if l.speaker == name], voice_options)
                if ref:
                    refs[name] = ref

        # Voices of each speaker's consistent lines so far (a regeneration starts from the lines it keeps).
        accepted: dict[str, list[np.ndarray]] = {}
        if only is not None and options.verify_voice:
            for line in lines:
                if line.speaker in new_voices:
                    continue  # their old lines are in the old voice
                entry = state["entries"].get(str(line.index))
                raw = raw_dir / f"{line.index:04d}.wav"
                if line.index in only or not entry or not entry.get("voice_ok") or not raw.exists():
                    continue
                try:
                    wav, rate = sf.read(str(raw), dtype="float32")
                    emb = self._embed(loop_to_length(wav, rate, 1.5), rate)
                except Exception as exc:
                    logger.warning("Voice check of kept lines skipped: %s", exc)
                    break
                if emb is not None:
                    accepted.setdefault(line.speaker, []).append(emb)

        # Speaker by speaker (the one with the most lines first), each in time order: one voice at a time.
        counts = Counter(l.speaker for l in lines)
        rank = {name: i for i, name in enumerate(sorted(counts, key=lambda n: (-counts[n], firsts[n].start)))}
        todo = sorted((l for l in lines if only is None or l.index in only), key=lambda l: (rank[l.speaker], l.start))
        for step, line in enumerate(todo):
            if progress:
                progress(step, len(todo), line)
            profile = profiles[line.speaker]
            reference = refs.get(line.speaker)
            if reference is None:
                mode = "design"
            elif profile.reference_wav:
                mode = "clone (voice template)"
            else:
                mode = f"clone→#{firsts[line.speaker].index} (neutral)"

            target = None
            if reference and options.verify_voice:
                try:
                    target = self._embedding_of_file(reference)
                except Exception as exc:
                    logger.warning("Voice check disabled: %s", exc)
                    self.warnings.append(f"Voice check disabled (voice model unavailable): {exc}")
                    options.verify_voice = False
                    target = None
            others = []
            if target is not None:
                for name, ref in refs.items():
                    if name != line.speaker:
                        emb = self._embedding_of_file(ref)
                        if emb is not None:
                            others.append(emb)
            # Each regeneration of a line uses new seeds (round 0 is the first generation).
            round_ = int(state["rounds"].get(str(line.index), -1)) + 1
            line_options = replace(options, seed=(options.seed or 0) + round_ * 104_729)
            own = accepted.get(line.speaker)
            best = self._generate_best(
                line,
                profile,
                mode,
                reference,
                target,
                line_options,
                others=others,
                target_pitch=self._target_pitch(line, reference, None) if reference else None,
                attempts=options.max_attempts,
                min_similarity=max(options.min_voice_similarity, options.min_consistency) if reference else None,
                peers=_centroid(own) if own else None,
                pitch_range=None if reference else pitch_range(profile.gender, profile.age),
                pitch_limits=pitch_limits(profile.gender, profile.age),
            )
            if best["voice_ok"] and best.get("embedding") is not None:
                accepted.setdefault(line.speaker, []).append(best["embedding"])
            if options.remove_silence:
                best["wav"] = remove_silences(best["wav"], sr, max_pause=options.max_pause)
            sf.write(str(raw_dir / f"{line.index:04d}.wav"), best["wav"], sr)
            for old in lines_dir.glob(f"{line.index:04d}_*.wav"):
                old.unlink()  # the speaker of a regenerated line may have changed
            line_file = pad_with_silence(best["wav"], sr, line.duration) if options.pad_to_slot else best["wav"]
            sf.write(str(lines_dir / f"{line.index:04d}_{re.sub(r'[^\w-]+', '_', line.speaker)[:24]}.wav"), line_file, sr)
            state["rounds"][str(line.index)] = round_
            state["entries"][str(line.index)] = {
                "mode": mode,
                "voice_match": None if best["similarity"] is None else round(best["similarity"], 2),
                "consistency": None if best.get("consistency") is None else round(best["consistency"], 2),
                "voice_ok": best["voice_ok"],
                "gender_ok": best["gender_ok"],
                "pitch_hz": None if best.get("pitch") is None else round(best["pitch"]),
                "attempts": best["tries"],
                "generated_s": round(len(best["wav"]) / sr, 2),
                "trimmed": bool(best.get("trimmed")),
                "issues": best.get("issues", []),
                "regenerated": round_,
            }
            # Saved after every line, so an interrupted run keeps (and can show) the lines already made.
            state["refs"] = refs
            state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        if progress:
            progress(len(todo), len(todo), todo[-1] if todo else lines[-1])
        state["refs"] = refs
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

        # Combined track: lines keep their natural pace (sped up by at most ``max_speedup``, default 1.1 here,
        # which is inaudible); a line that is still too long pushes the following lines a little later
        # instead of being squeezed, since squeezing changes how the voice sounds.
        clips, report, waiting = [], [], []
        cursor = 0.0
        for line in lines:
            raw = raw_dir / f"{line.index:04d}.wav"
            entry = state["entries"].get(str(line.index))
            if entry is None or not raw.exists():
                waiting.append(line)  # not made yet (e.g. only one speaker was generated): left out of the track
                continue
            wav, _ = sf.read(str(raw), dtype="float32")
            fitted, rate = fit_to_slot(wav, sr, slot=line.duration, available=1e9, max_speedup=options.max_speedup)
            start = max(line.start, cursor)
            clips.append((start, fitted))
            cursor = start + len(fitted) / sr + 0.15
            profile = profiles[line.speaker]
            report.append(
                {
                    "index": line.index,
                    "start": round(line.start, 3),
                    "end": round(line.end, 3),
                    "speaker": line.speaker,
                    "gender": profile.gender,
                    "age": profile.age,
                    "emotion": line.emotion,
                    "emotion_guessed": bool(line.features.get("emotion_guessed")),
                    **entry,
                    "placed_s": round(len(fitted) / sr, 2),
                    "speedup": round(rate, 2),
                    "shift_s": round(start - line.start, 2),
                    "style": line_style(line),
                    "text": line.text,
                }
            )
        if not report:
            raise ValueError("No line has been generated yet.")
        if waiting:
            names = sorted({l.speaker for l in waiting})
            self.warnings.append(
                f"{len(waiting)} line(s) are not generated yet ({', '.join(names)}), so the combined track has only "
                "the lines made so far. Generate them to complete it."
            )
        total = max(cursor, max(l.end for l in lines)) + 0.5
        shifted = [r for r in report if r["shift_s"] > 0.05]
        if shifted:
            self.warnings.append(
                f"{len(shifted)} line(s) were longer than their subtitle slot, so the combined track moves later lines "
                f"by up to {max(r['shift_s'] for r in shifted):.1f} s (the per-line files are unchanged)."
            )
        inconsistent = [f"#{r['index']} ({r['speaker']})" for r in report if not r["voice_ok"]]
        if inconsistent:
            message = (
                f"{len(inconsistent)} line(s) are still not consistent with their speaker and are marked ⚠ in the "
                f"report: {', '.join(inconsistent)}. Select them and click Regenerate to try again."
            )
            logger.warning(message)
            self.warnings.append(message)
        unclear = [f"#{r['index']} ({r['speaker']}, {r['pitch_hz']} Hz)" for r in report if not r.get("gender_ok", True)]
        if unclear:
            message = (
                f"{len(unclear)} line(s) may not sound clearly like their speaker's gender/age (pitch outside the "
                f"clear range): {', '.join(unclear)}. Select them and click Regenerate to try again."
            )
            logger.warning(message)
            self.warnings.append(message)

        mixed = mix_timeline(clips, total, sr, background_mode="mute")
        audio_path = str(stem) + ".wav"
        sf.write(audio_path, mixed, sr)
        tagged_srt_path = str(stem) + "_tagged.srt"
        Path(tagged_srt_path).write_text(format_tagged_srt(lines, profiles), encoding="utf-8")
        Path(str(stem) + "_plain.srt").write_text(format_srt(lines), encoding="utf-8")
        (self.workdir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return ScriptResult(
            audio_path, str(lines_dir), tagged_srt_path, report, list(self.warnings), voices=dict(refs)
        )

    def regenerate(
        self,
        lines: list[SubtitleLine],
        profiles: dict[str, SpeakerProfile],
        output_path: str,
        indices: Iterable[int],
        options: Optional[DubOptions] = None,
        progress: Optional[Callable[[int, int, SubtitleLine], None]] = None,
    ) -> ScriptResult:
        """Regenerate only the lines numbered ``indices`` (e.g. the failed ones) of an earlier :meth:`voice` run."""
        return self.voice(lines, profiles, output_path, options, progress, only=indices)


def revoice_speakers(
    voicer: "ScriptVoicer",
    lines: list[SubtitleLine],
    profiles: dict[str, SpeakerProfile],
    output_path: str,
    speakers: Iterable[str],
    options: Optional[DubOptions] = None,
    progress: Optional[Callable[[int, int, SubtitleLine], None]] = None,
) -> ScriptResult:
    """Give ``speakers`` a new voice and regenerate all of their lines with it (other speakers are kept)."""
    return voicer.voice(lines, profiles, output_path, options, progress, only=(), new_voices=speakers)


def load_results(workdir: str | os.PathLike, output_path: str | os.PathLike) -> Optional[ScriptResult]:
    """The results of the last finished :meth:`ScriptVoicer.voice` run in ``workdir`` (``None`` if there is none)."""
    workdir = Path(workdir)
    stem = Path(output_path).with_suffix("")
    report_path, audio_path = workdir / "report.json", Path(str(stem) + ".wav")
    if not report_path.exists() or not audio_path.exists():
        return None
    state_path = workdir / "script_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    return ScriptResult(
        str(audio_path),
        str(stem) + "_lines",
        str(stem) + "_tagged.srt",
        json.loads(report_path.read_text(encoding="utf-8")),
        [],
        voices={n: p for n, p in state.get("refs", {}).items() if Path(p).exists()},
    )


def generated_lines(workdir: str | os.PathLike) -> set[int]:
    """Line numbers already generated in ``workdir`` (also while a run is still going or after it stopped)."""
    state_path = Path(workdir) / "script_state.json"
    if not state_path.exists():
        return set()
    entries = json.loads(state_path.read_text(encoding="utf-8")).get("entries", {})
    return {int(i) for i in entries if (Path(workdir) / "raw" / f"{int(i):04d}.wav").exists()}


def failed_lines(report: list[dict]) -> list[int]:
    """Line numbers that are not consistent with their speaker or not clearly their gender."""
    return [r["index"] for r in report if not r.get("voice_ok", True) or not r.get("gender_ok", True)]


# -----------------------------
# Voice templates
# -----------------------------

AUTO_VOICE = "auto"  # no template: the voice is designed from the speaker's first line(s)
VOICE_LIBRARY_DIR = Path(os.environ.get("VOXCPM_VOICES_DIR", Path.home() / ".voxcpm" / "voices"))
# What a template says (calm and neutral, about 6 s): "Hello, I am glad to meet you today. Let's start our story
# together." Every line of a speaker copies the voice of this clip and adds its own emotion.
TEMPLATE_TEXT = "សួស្តី ខ្ញុំរីករាយណាស់ដែលបានជួបអ្នកនៅថ្ងៃនេះ។ តោះយើងចាប់ផ្តើមរឿងរបស់យើងទាំងអស់គ្នា។"
# Written as casting notes for voice actors, so the templates sound like film dubbing, not like ordinary speech.
STARTER_VOICES = [
    {"name": "Man", "gender": "male", "age": "adult",
     "description": "Professional male voice actor for film dubbing, adult man, deep, rich and resonant, clear diction, "
                    "natural and expressive"},
    {"name": "Woman", "gender": "female", "age": "adult",
     "description": "Professional female voice actor for film dubbing, adult woman, warm, smooth and expressive, clear "
                    "diction, clearly feminine"},
    {"name": "Young man", "gender": "male", "age": "adult",
     "description": "Young male voice actor in his twenties for film dubbing, clear and bright, confident, clearly "
                    "masculine, natural and expressive"},
    {"name": "Young woman", "gender": "female", "age": "adult",
     "description": "Young female voice actor in her twenties for film dubbing, bright, sweet and lively, clear diction, "
                    "clearly feminine, medium-high pitch"},
    {"name": "Old man", "gender": "male", "age": "adult",
     "description": "Veteran male voice actor playing an elderly man of about seventy, deep and low, gravelly, slow "
                    "and weighty, clearly masculine, not high-pitched"},
    {"name": "Old woman", "gender": "female", "age": "adult",
     "description": "Veteran female voice actor playing an elderly woman of about seventy, soft, slightly raspy, gentle "
                    "and slow, clearly feminine"},
    {"name": "Boy", "gender": "male", "age": "kid",
     "description": "Child voice actor, a boy of about ten, clearly a boy, boyish and energetic, slightly husky, not "
                    "girlish, clear diction"},
    {"name": "Girl", "gender": "female", "age": "kid",
     "description": "Child voice actress, a girl of about ten, sweet, bright and lively, clearly a girl, clear diction"},
    {"name": "Kid", "gender": "unknown", "age": "kid",
     "description": "Child voice actor of about eight, bright, playful and clear"},
    {"name": "Narrator", "gender": "unknown", "age": "adult",
     "description": "Professional narrator for film trailers and documentaries, warm, rich, calm and authoritative, "
                    "clear diction"},
]


def _library(directory=None) -> Path:
    return Path(directory) if directory else VOICE_LIBRARY_DIR


def load_voice_library(directory=None) -> dict[str, dict]:
    """The voice templates: ``{name: {"path", "gender", "age", "description", "starter"}}`` (missing clips left out)."""
    index = _library(directory) / "voices.json"
    try:
        data = json.loads(index.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {n: dict(v, path=str(_library(directory) / v["file"])) for n, v in data.items()
            if (_library(directory) / v.get("file", "")).exists()}


def save_voice_template(
    name: str, wav_path: str | os.PathLike, gender: str = "unknown", age: str = "adult", description: str = "",
    starter: bool = False, directory=None,
) -> dict:
    """Keep a clip in the voice library as the template ``name`` (replacing one of that name)."""
    import shutil

    name = str(name).strip()
    if not name or name.lower() == AUTO_VOICE:
        raise ValueError("Give the voice a name.")
    folder = _library(directory)
    folder.mkdir(parents=True, exist_ok=True)
    index = folder / "voices.json"
    try:
        data = json.loads(index.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    slug = re.sub(r"[^\w-]+", "_", name)[:40] or "voice"
    file = f"{slug}.wav"
    if name not in data and any(v.get("file") == file for v in data.values()):
        file = f"{slug}_{len(data)}.wav"
    shutil.copyfile(wav_path, folder / file)
    data[name] = {"file": file, "gender": gender, "age": age, "description": description, "starter": bool(starter)}
    pitch = template_pitch({"path": str(folder / file)})  # kept, so matching speakers to templates is quick
    if pitch:
        data[name]["pitch_hz"] = round(pitch, 1)
    index.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return dict(data[name], path=str(folder / file))


def delete_voice_template(name: str, directory=None) -> None:
    folder = _library(directory)
    index = folder / "voices.json"
    data = json.loads(index.read_text(encoding="utf-8")) if index.exists() else {}
    entry = data.pop(name, None)
    if entry:
        (folder / entry["file"]).unlink(missing_ok=True)
        index.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def template_voice(name: str, directory=None) -> Optional[dict]:
    """The template called ``name`` (``None`` for "auto" or an empty name); an unknown name is an error."""
    name = str(name or "").strip()
    if not name or name.lower() == AUTO_VOICE:
        return None
    library = load_voice_library(directory)
    if name not in library:
        raise ValueError(f"There is no voice template called {name!r} (make it under 🎙 Voice templates).")
    return library[name]


def gender_lean(embedding: np.ndarray, males: list[np.ndarray], females: list[np.ndarray]) -> float:
    """How much more a voice sounds like the men than like the women (speaker-embedding similarity, -1..1)."""
    unit = embedding / (np.linalg.norm(embedding) + 1e-9)
    best = lambda refs: max(float(unit @ (r / (np.linalg.norm(r) + 1e-9))) for r in refs)
    return best(males) - best(females)


def make_voice_template(
    voicer: "ScriptVoicer", name: str, gender: str, age: str, description: str, seed: int = 0,
    options: Optional[DubOptions] = None, starter: bool = False, directory=None, text: str = TEMPLATE_TEXT,
    candidates: int = 1, gender_refs: Optional[dict[str, list[np.ndarray]]] = None, embed=None,
) -> dict:
    """Design a voice from its description (a calm reading of :data:`TEMPLATE_TEXT`, retried until its pitch fits
    the gender and age) and keep it in the library. The same name, description and seed give the same voice.

    With ``candidates`` > 1, ``gender_refs`` (``{"male": [...], "female": [...]}`` speaker embeddings, e.g. of the
    adult templates) and ``embed`` (path -> embedding), several versions are made and the one that sounds most like
    its gender is kept - a boy's voice is as high as a girl's, so pitch alone can not tell them apart."""
    import shutil
    import tempfile

    profile = SpeakerProfile(name=name, gender=gender, age=age, voice_mode="clone_first",
                             description=description or voice_description(gender, age))
    line = SubtitleLine(index=1, start=0.0, end=6.0, text=text, speaker=name, gender=gender, emotion="calm and neutral")
    judge = gender in ("male", "female") and candidates > 1 and embed and gender_refs \
        and gender_refs.get("male") and gender_refs.get("female")
    best_path, best_score, scores = None, None, []
    with tempfile.TemporaryDirectory() as tmp:
        for k in range(max(candidates if judge else 1, 1)):
            # Far apart: each version's own retries step by 7919, and must not run into the next version's seeds.
            path = voicer.speaker_voice(profile, [line], replace(options or DubOptions(), seed=seed + 1_000_003 * k))
            if not path:
                continue
            kept = str(Path(tmp) / f"candidate_{k}.wav")
            shutil.copyfile(path, kept)
            score = 0.0
            if judge:
                emb = embed(kept)
                if emb is not None:
                    lean = gender_lean(emb, gender_refs["male"], gender_refs["female"])
                    score = lean if gender == "male" else -lean
            scores.append(round(score, 3))
            if best_score is None or score > best_score:
                best_path, best_score = kept, score
        if not best_path:
            raise RuntimeError(f"The voice {name!r} came out too short; try again or change its description.")
        made = save_voice_template(name, best_path, gender, age, profile.description, starter=starter, directory=directory)
    made["candidate_scores"] = scores
    return made


# The usual template for a gender and age (by starter name), when nothing else tells the speakers apart.
_DEFAULT_TEMPLATE = {
    ("male", "adult"): "Man", ("female", "adult"): "Woman", ("male", "kid"): "Boy", ("female", "kid"): "Girl",
    ("unknown", "kid"): "Kid", ("unknown", "adult"): "Narrator",
}
_OLD_WORDS = ("old", "elder", "grand", "aged", "senior", "chief", "ចាស់", "តា", "យាយ")
_YOUNG_WORDS = ("young", "teen", "youth", "boyish", "girlish", "ក្មេង")


def template_pitch(template: dict) -> Optional[float]:
    """Median pitch (Hz) of a template's clip (kept in the template once measured)."""
    if template.get("pitch_hz"):
        return float(template["pitch_hz"])
    import soundfile as sf

    from .dubbing import median_pitch

    try:
        wav, sr = sf.read(template["path"], dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        return median_pitch(wav, sr)
    except Exception:
        return None


def speaker_pitches(lines: list[SubtitleLine], audio: np.ndarray, sr: int, max_lines: int = 12) -> dict[str, float]:
    """Each speaker's median pitch in the original video (from up to ``max_lines`` of their longest lines)."""
    from .dubbing import _segment, median_pitch

    found: dict[str, float] = {}
    for name in dict.fromkeys(l.speaker for l in lines):
        own = sorted((l for l in lines if l.speaker == name), key=lambda l: -l.duration)[:max_lines]
        clip = np.concatenate([_segment(audio, sr, l.start, l.end) for l in own] or [np.zeros(0, np.float32)])
        f0 = median_pitch(clip.astype(np.float32), sr) if len(clip) > sr // 2 else None
        if f0:
            found[name] = f0
    return found


def match_templates(
    profiles: dict[str, SpeakerProfile],
    library: dict[str, dict],
    line_counts: Optional[dict[str, int]] = None,
    pitches: Optional[dict[str, float]] = None,
) -> dict[str, str]:
    """The template that fits each speaker best: one of their gender and age; an old / young / narrator one when
    their name or description says so; the closest in pitch to their voice in the original video (``pitches``);
    and, where there is a choice, a different one for each speaker (the speakers with the most lines choose first)."""
    if not library:
        return {}
    import math

    t_pitch = {n: template_pitch(t) for n, t in library.items()}
    used: dict[str, int] = {}
    chosen: dict[str, str] = {}
    order = sorted(profiles, key=lambda n: -(line_counts or {}).get(n, 0))
    for name in order:
        p = profiles[name]
        words = f"{name} {p.description or ''}".lower()
        old, young = any(w in words for w in _OLD_WORDS), any(w in words for w in _YOUNG_WORDS)
        narrator = "narrat" in words

        def score(t_name: str) -> float:
            t = library[t_name]
            t_words = f"{t_name} {t.get('description', '')}".lower()
            cost = 0.0
            if p.gender in ("male", "female") and t["gender"] in ("male", "female") and t["gender"] != p.gender:
                cost += 100
            if p.gender in ("male", "female") and t["gender"] == "unknown":
                cost += 8
            if (p.age or "adult") != t.get("age", "adult"):
                cost += 50
            t_old = any(w in t_words for w in ("old", "elder"))
            t_young = "young" in t_words
            cost += -10 if old and t_old else 6 if t_old else 0
            cost += -6 if young and t_young else 0
            if narrator and "narrat" in t_words:
                cost -= 20
            elif "narrat" in t_words:
                cost += 4
            f_s, f_t = (pitches or {}).get(name), t_pitch.get(t_name)
            if f_s and f_t:
                cost += 1.5 * abs(12 * math.log2(f_t / f_s))  # semitones away from how they really sound
            elif _DEFAULT_TEMPLATE.get((p.gender, p.age or "adult")) == t_name:
                cost -= 3
            return cost + 12 * used.get(t_name, 0)  # two speakers sharing a voice only when nothing else fits

        best = min(library, key=lambda t_name: (score(t_name), t_name))
        chosen[name] = best
        used[best] = used.get(best, 0) + 1
    return chosen


def speaker_summary(lines: list[SubtitleLine], profiles: dict[str, SpeakerProfile]) -> list[list]:
    """``[name, gender, age, voice description, number of lines, voice template]`` rows ("auto": designed from the
    speaker's first line)."""
    counts = Counter(l.speaker for l in lines)
    return [[p.name, p.gender, p.age, p.description, counts.get(p.name, 0), AUTO_VOICE] for p in profiles.values()]


__all__ = [
    "AUTO_VOICE",
    "DEFAULT_SPEAKER",
    "STARTER_VOICES",
    "ScriptResult",
    "ScriptVoicer",
    "VoiceTag",
    "apply_tones",
    "delete_voice_template",
    "load_voice_library",
    "gender_lean",
    "make_voice_template",
    "match_templates",
    "speaker_pitches",
    "save_voice_template",
    "template_voice",
    "failed_lines",
    "generated_lines",
    "load_results",
    "revoice_speakers",
    "format_tagged_srt",
    "guess_emotion",
    "load_script",
    "parse_srt",
    "parse_voice_tag",
    "pitch_range",
    "read_script",
    "speaker_summary",
    "tones_from_audio",
    "voice_description",
]
