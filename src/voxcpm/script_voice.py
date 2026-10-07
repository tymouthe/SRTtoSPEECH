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
    ("male", "kid"): "Young boy's voice, a child of about eight, high and bright",
    ("female", "kid"): "Young girl's voice, a child of about eight, high and bright",
    ("unknown", "kid"): "Young child's voice, a child of about eight, high and bright",
}
# Median F0 (Hz) a speaker's voice must land in to sound clearly like that gender and age: kept well away
# from the ~165 Hz point where adult male and female voices meet.
PITCH_RANGES = {
    ("male", "adult"): (90.0, 140.0),
    ("female", "adult"): (185.0, 255.0),
    ("unknown", "adult"): (85.0, 255.0),
    ("male", "kid"): (200.0, 380.0),
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
    """Blocks of an SRT that would be skipped (no readable ``start --> end`` timing line), as messages."""
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
        if not rows or any(_ARROW_TIME_RE.match(r) for r in rows):
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
            if not state_path.exists():
                raise ValueError("Generate all lines first; then failed lines can be regenerated.")
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
            if profile.reference_wav and name not in new_voices:
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

        todo = [l for l in lines if only is None or l.index in only]
        for step, line in enumerate(todo):
            if progress:
                progress(step, len(todo), line)
            profile = profiles[line.speaker]
            reference = refs.get(line.speaker)
            if reference is None:
                mode = "design"
            elif profile.reference_wav:
                mode = "clone (your reference)"
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
        clips, report = [], []
        cursor = 0.0
        for line in lines:
            raw = raw_dir / f"{line.index:04d}.wav"
            entry = state["entries"].get(str(line.index))
            if entry is None or not raw.exists():
                raise ValueError(f"Line #{line.index} has not been generated yet; generate all lines first.")
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


def speaker_summary(lines: list[SubtitleLine], profiles: dict[str, SpeakerProfile]) -> list[list]:
    """``[name, gender, age, voice description, number of lines]`` rows."""
    counts = Counter(l.speaker for l in lines)
    return [[p.name, p.gender, p.age, p.description, counts.get(p.name, 0)] for p in profiles.values()]


__all__ = [
    "DEFAULT_SPEAKER",
    "ScriptResult",
    "ScriptVoicer",
    "VoiceTag",
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
    "voice_description",
]
