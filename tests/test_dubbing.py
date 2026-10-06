import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("voxcpm")
pkg.__path__ = [str(ROOT / "src" / "voxcpm")]
sys.modules.setdefault("voxcpm", pkg)

from voxcpm import dubbing  # noqa: E402
from voxcpm.dubbing import SubtitleLine  # noqa: E402

SRT = """﻿1
00:00:00,500 --> 00:00:02,000
[Alice] Hello there!

2
00:00:02,500 --> 00:00:04,000
Bob: Hi Alice,
<i>how are you?</i>

3
00:00:04,200 --> 00:00:05,000
Untagged line: with a colon in the middle of a long sentence
"""


def test_parse_srt_extracts_speakers_times_and_text():
    lines = dubbing.parse_srt(SRT.replace("\n", "\r\n"))

    assert [(l.speaker, l.text) for l in lines] == [
        ("Alice", "Hello there!"),
        ("Bob", "Hi Alice, how are you?"),
        (None, "Untagged line: with a colon in the middle of a long sentence"),
    ]
    assert lines[1].start == 2.5 and lines[1].end == 4.0


def test_format_srt_round_trips_speakers():
    lines = dubbing.parse_srt(SRT)
    again = dubbing.parse_srt(dubbing.format_srt(lines, with_speaker=True))
    assert [(l.speaker, l.text, l.start, l.end) for l in again] == [(l.speaker, l.text, l.start, l.end) for l in lines]
    assert "[Alice]" not in dubbing.format_srt(lines)


def test_parse_timestamp_accepts_seconds_and_srt_format():
    assert dubbing.parse_timestamp(1.5) == 1.5
    assert dubbing.parse_timestamp("00:01:02,500") == pytest.approx(62.5)
    assert dubbing.parse_timestamp("3.25") == 3.25


def _tone(freq, seconds=1.0, sr=16000):
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_classify_gender_from_pitch():
    sr = 16000
    assert dubbing.classify_gender(dubbing.estimate_f0(_tone(110), sr)) == "male"
    assert dubbing.classify_gender(dubbing.estimate_f0(_tone(230), sr)) == "female"
    assert dubbing.classify_gender(np.zeros(0)) == "unknown"


def test_untagged_lines_are_grouped_by_gender_and_get_one_voice_each():
    lines = [
        SubtitleLine(1, 0, 1, "a", gender="male"),
        SubtitleLine(2, 1, 2, "b", gender="female"),
        SubtitleLine(3, 2, 3, "c", speaker="Carol", gender="female"),
    ]
    profiles = dubbing.build_speaker_profiles(lines)
    assert [l.speaker for l in lines] == ["Male", "Female", "Carol"]
    assert profiles["Carol"].voice_mode == "clone_first"
    assert profiles["Male"].voice_mode == "clone_first"
    assert profiles["Carol"].description == "Adult female voice"

    dubbing.apply_speaker_overrides(profiles, {"Carol": {"mode": "design", "voice": "old woman, raspy"}})
    assert profiles["Carol"].voice_mode == "design"
    assert profiles["Carol"].description == "old woman, raspy"
    with pytest.raises(ValueError):
        dubbing.apply_speaker_overrides(profiles, {"Carol": {"voice_mode": "nope"}})


def test_fit_to_slot_speeds_up_then_trims():
    sr = 16000
    wav = _tone(200, seconds=2.0)

    same, rate = dubbing.fit_to_slot(wav, sr, slot=3.0, available=3.0)
    assert rate == 1.0 and len(same) == len(wav)

    faster, rate = dubbing.fit_to_slot(wav, sr, slot=1.8, available=1.8, max_speedup=1.5)
    assert rate == pytest.approx(2.0 / 1.8)
    assert abs(len(faster) / sr - 1.8) < 0.05

    trimmed, rate = dubbing.fit_to_slot(wav, sr, slot=1.0, available=1.2, max_speedup=1.25)
    assert rate == pytest.approx(1.25)
    assert abs(len(trimmed) - int(1.2 * sr)) <= 1
    assert abs(trimmed[-1]) < 1e-3


def test_mix_timeline_places_clips_and_ducks_background():
    sr = 1000
    clip = np.full(500, 0.5, dtype=np.float32)
    background = np.full(3000, 0.2, dtype=np.float32)
    out = dubbing.mix_timeline(
        [(1.0, clip)], 3.0, sr, background=background, duck_db=-20, speech_regions=[(1.0, 1.5)], ramp_seconds=0.01
    )
    assert len(out) == 3000
    assert out[100] == pytest.approx(0.2)
    assert out[1250] == pytest.approx(0.5 + 0.02, abs=1e-3)

    muted = dubbing.mix_timeline([(1.0, clip)], 3.0, sr, background=background, background_mode="mute")
    assert muted[100] == 0 and muted[1250] == pytest.approx(0.5)


from voxcpm.model.voxcpm2 import VoxCPM2Model  # noqa: E402


class _StubTTS(VoxCPM2Model):
    """Passes the VoxCPM2 check without loading weights."""

    sample_rate = 24000
    _encode_sample_rate, patch_size, chunk_size = 16000, 4, 640  # 6.25 audio tokens per second, like VoxCPM2

    def __init__(self):  # skip nn.Module/model initialisation
        pass


class _StubModel:
    """Returns a 1.5 s tone per call and records the kwargs."""

    tts_model = _StubTTS()

    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return _tone(180, seconds=1.5, sr=self.tts_model.sample_rate)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_end_to_end_dub_with_stub_model(tmp_path):
    video = tmp_path / "in.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "color=c=black:s=64x64:d=4",
            "-f", "lavfi", "-i", "sine=frequency=120:duration=2",
            "-f", "lavfi", "-i", "sine=frequency=240:duration=2",
            "-filter_complex", "[1:a][2:a]concat=n=2:v=0:a=1[a]",
            "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-c:a", "aac", "-shortest", str(video),
        ],
        check=True,
    )
    srt = tmp_path / "in.srt"
    srt.write_text(
        "1\n00:00:00,200 --> 00:00:01,800\nHello from the first voice\n\n"
        "2\n00:00:02,200 --> 00:00:02,800\n[Zoe] Short slot\n\n"
        "3\n00:00:03,000 --> 00:00:04,000\n[Zoe] Same voice again\n",
        encoding="utf-8",
    )

    model = _StubModel()
    dubber = dubbing.VideoDubber(model, tmp_path / "work")
    lines, profiles = dubber.prepare(str(video), str(srt), detect_emotion=False, identify_speakers=False)
    assert [l.speaker for l in lines] == ["Male", "Zoe", "Zoe"]
    assert profiles["Zoe"].gender == "female"

    profiles["Zoe"].voice_mode = "design"
    result = dubber.run(
        str(video),
        lines,
        profiles,
        str(tmp_path / "out" / "dub.mp4"),
        dubbing.DubOptions(background="duck", verify_voice=False, max_attempts=1, first_line_attempts=1, design_attempts=1),
    )

    # Zoe's voice is designed once, up front, from a calm neutral reading without any reference...
    assert model.calls[0]["reference_wav_path"] is None
    assert model.calls[0]["text"].startswith("(Adult female voice")
    assert "calm and neutral" in model.calls[0]["text"]
    assert model.calls[1]["reference_wav_path"] is not None  # clone_line uses the original audio
    # ...and every one of her lines clones that designed voice, with the same seed.
    designed = model.calls[0]["seed"]
    assert model.calls[2]["reference_wav_path"] == model.calls[3]["reference_wav_path"] is not None
    assert result.report[1]["mode"] == result.report[2]["mode"] == "design→clone"
    assert model.calls[2]["seed"] == model.calls[3]["seed"] == designed != model.calls[1]["seed"]
    assert result.report[1]["speedup"] > 1.0  # 1.5 s of speech squeezed toward a 0.6 s slot
    assert result.report[1]["placed_s"] <= 3.0 - 2.2 + 0.01
    assert Path(result.video_path).exists()
    assert "[Zoe] Short slot" in Path(result.speaker_srt_path).read_text(encoding="utf-8")
    assert dubbing.has_video_stream(result.video_path)
    assert dubbing.probe_duration(result.video_path) == pytest.approx(4.0, abs=0.2)


def test_separate_chunks_overlap_add_reconstructs_the_mix():
    import torch

    class FakeSeparator(torch.nn.Module):
        sources = ["drums", "bass", "other", "vocals"]

        def forward(self, x):
            weights = torch.tensor([0.1, 0.2, 0.3, 0.4]).view(1, 4, 1, 1)
            return x.unsqueeze(1) * weights

    sr = 100
    mix = torch.randn(1, 2, 25 * sr + 37)
    out = dubbing._separate_chunks(FakeSeparator(), mix, sr, segment=10.0, overlap=0.1, device="cpu")
    assert out.shape == (1, 4, 2, mix.shape[-1])
    torch.testing.assert_close(out.sum(1), mix, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(out[:, 3], mix * 0.4, atol=1e-4, rtol=1e-4)


def test_parse_sensevoice_emotion():
    assert dubbing.parse_sensevoice_emotion("<|en|><|HAPPY|><|Speech|><|woitn|>great") == "happy"
    assert dubbing.parse_sensevoice_emotion("<|zh|><|EMO_UNKNOWN|><|Speech|>hi") == "neutral"
    assert dubbing.parse_sensevoice_emotion("") == "neutral"


def test_tone_is_relative_to_each_speakers_own_baseline():
    def line(i, speaker, rms_db, emotion=""):
        l = SubtitleLine(i, i, i + 1, "a few words here", speaker=speaker, emotion=emotion)
        l.features = {"rms_db": rms_db, "pitch_spread": 2.0, "f0_median": 120.0, "rate": 3.0}
        return l

    # Bob is always loud: none of his normal lines should be flagged as loud; Amy's one shout should.
    lines = [line(1, "Bob", -10), line(2, "Bob", -10), line(3, "Bob", -11), line(4, "Amy", -25), line(5, "Amy", -25),
             line(6, "Amy", -12, emotion="angry")]
    dubbing.assign_tones(lines)
    assert [l.tone for l in lines[:3]] == ["", "", ""]
    assert lines[5].tone == "louder and more forceful"
    assert dubbing.line_style(lines[5]) == "angry and intense, louder and more forceful"
    lines[5].emotion = "sarcastic"  # free-form emotions typed in the UI are passed through
    assert dubbing.line_style(lines[5]) == "sarcastic, louder and more forceful"
    assert dubbing.describe_speaker("male", lines[:3]).startswith("Adult male voice, medium pitch")


def test_speaker_seed_is_stable_per_speaker():
    assert dubbing.speaker_seed("Alice") == dubbing.speaker_seed("Alice")
    assert dubbing.speaker_seed("Alice") != dubbing.speaker_seed("Bob")
    assert dubbing.speaker_seed("Alice", 1) == dubbing.speaker_seed("Alice") + 1


def test_clean_voice_removes_noise_in_pauses_and_levels_loudness():
    sr = 24000
    rng = np.random.default_rng(0)
    speech = np.concatenate([np.zeros(sr // 2), _tone(220, 1.0, sr), np.zeros(sr // 2), _tone(330, 1.0, sr), np.zeros(sr // 2)])
    noisy = (speech + 0.01 * rng.standard_normal(len(speech))).astype(np.float32)

    cleaned = dubbing.clean_voice(noisy, sr, target_lufs=-20.0)

    def level(x, a, b):
        return 20 * np.log10(np.sqrt(np.mean(x[a:b] ** 2)) + 1e-12)

    pause = (int(1.6 * sr), int(1.9 * sr))
    voiced = (int(0.7 * sr), int(1.3 * sr))
    snr_before = level(noisy, *voiced) - level(noisy, *pause)
    snr_after = level(cleaned, *voiced) - level(cleaned, *pause)
    assert snr_after - snr_before > 15
    assert np.max(np.abs(cleaned)) <= 0.97 + 1e-6


def test_failed_optional_vocal_removal_still_analyses_original_audio(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("download.pytorch.org unreachable")

    monkeypatch.setattr(dubbing, "separate_vocals", boom)
    monkeypatch.setattr(dubbing, "extract_audio", lambda src, dst, sr: Path(dst).touch())
    dubber = dubbing.VideoDubber(None, tmp_path)
    dubber.separate = lambda *a, **k: boom()
    monkeypatch.setattr(dubber, "_load_speech_audio", lambda: np.zeros(16000 * 3, dtype=np.float32))
    srt = tmp_path / "a.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\n[Ann] Hi\n", encoding="utf-8")

    lines, profiles = dubber.prepare(
        "video.mp4", str(srt), remove_vocals=True, detect_emotion=False, identify_speakers=False
    )

    assert [l.speaker for l in lines] == ["Ann"]
    assert "Vocal removal skipped" in dubber.warnings[0]
    with pytest.raises(RuntimeError, match="mute"):
        dubber._background("video.mp4", "instrumental", 24000)


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_cluster_speakers_separates_voices_and_folds_in_short_lines():
    rng = np.random.default_rng(1)
    a, b, c = (_unit(rng.standard_normal(16)) for _ in range(3))

    def noisy(v):
        return _unit(v + 0.15 * rng.standard_normal(16))

    # Two women (a, c) that a pitch split would have merged into one "Female", plus a man (b).
    embs = [noisy(a), noisy(b), noisy(a), None, noisy(c), noisy(c), noisy(b), noisy(a), noisy(c)]
    labels = dubbing.cluster_speakers(embs)
    assert labels == [0, 1, 0, 0, 2, 2, 1, 0, 2]  # numbered by first appearance; None joins its neighbour
    assert len(set(dubbing.cluster_speakers(embs, num_speakers=2))) == 2
    assert dubbing.cluster_speakers([None, None]) == [0, 0]


def test_voice_consistency_flags_mistagged_lines():
    rng = np.random.default_rng(2)
    a = _unit(rng.standard_normal(16))
    b = rng.standard_normal(16)
    b = _unit(b - (b @ a) * a)  # a clearly different voice
    lines = [SubtitleLine(i, i, i + 1, "x", speaker="Ann") for i in range(4)]
    for line, emb in zip(lines, [a, a, a, b]):
        line.features["embedding"] = emb
    dubbing.score_voice_consistency(lines)
    assert lines[0].features["voice_match"] > 0.8
    assert lines[3].features["voice_match"] < 0.4


def test_generate_best_retries_until_voice_matches(tmp_path, monkeypatch):
    target = _unit([1.0, 0.0])
    voices = iter([_unit([0.2, 1.0]), _unit([1.0, 0.1])])  # first try sounds like someone else

    model = _StubModel()
    dubber = dubbing.VideoDubber(model, tmp_path)
    monkeypatch.setattr(dubber, "_embed", lambda wav, sr: next(voices))
    line = SubtitleLine(1, 0.0, 3.0, "hello", speaker="Ann")
    profile = dubbing.SpeakerProfile("Ann")
    best = dubber._generate_best(line, profile, "clone_speaker", "ref.wav", target, dubbing.DubOptions(max_attempts=3))

    assert len(model.calls) == 2 and best["attempt"] == 2 and best["tries"] == 2 and best["voice_ok"]
    assert best["similarity"] > 0.9
    assert model.calls[0]["seed"] != model.calls[1]["seed"]


def test_generate_best_asks_for_faster_speech_when_line_is_too_long(tmp_path):
    model = _StubModel()  # always 1.5 s
    dubber = dubbing.VideoDubber(model, tmp_path)
    line = SubtitleLine(1, 0.0, 1.0, "hello", speaker="Ann")
    dubber._generate_best(
        line, dubbing.SpeakerProfile("Ann"), "clone_speaker", None, None, dubbing.DubOptions(max_attempts=2)
    )
    assert "speaking quickly" not in model.calls[0]["text"]
    assert "speaking quickly" in model.calls[1]["text"]


def test_finalize_edges_drops_noise_burst_after_last_word_and_fades():
    sr = 24000
    rng = np.random.default_rng(3)
    voice = _tone(200, 1.0, sr)
    pause = np.zeros(int(0.3 * sr), dtype=np.float32)
    burst = (0.1 * rng.standard_normal(int(0.3 * sr))).astype(np.float32)  # mumble/breath after the word
    out = dubbing.finalize_edges(np.concatenate([voice, pause, burst]), sr)

    assert 1.0 <= len(out) / sr <= 1.0 + 0.06
    assert abs(out[0]) < 1e-3 and abs(out[-1]) < 1e-3


def test_finalize_edges_never_cuts_continuous_speech():
    sr = 24000
    rng = np.random.default_rng(4)
    voiced = _tone(200, 0.3, sr)
    breathy = (0.1 * rng.standard_normal(int(0.5 * sr))).astype(np.float32)  # unvoiced but still the word
    out = dubbing.finalize_edges(np.concatenate([voiced, breathy]), sr)
    assert len(out) / sr >= 0.79


def test_generate_best_requires_line_to_sound_like_its_own_character(tmp_path, monkeypatch):
    own, other = _unit([1.0, 0.0, 0.0]), _unit([0.0, 1.0, 0.0])
    # try 1: similar to both characters (ambiguous); try 2: clearly the right character
    voices = iter([_unit([0.6, 0.6, 0.5]), _unit([0.7, 0.1, 0.7])])
    model = _StubModel()
    dubber = dubbing.VideoDubber(model, tmp_path)
    monkeypatch.setattr(dubber, "_embed", lambda wav, sr: next(voices))
    line = SubtitleLine(1, 0.0, 3.0, "hello", speaker="Ann")
    best = dubber._generate_best(
        line, dubbing.SpeakerProfile("Ann"), "clone_speaker", None, own, dubbing.DubOptions(), others=[other]
    )
    assert best["attempt"] == 2
    assert best["margin"] > 0.5


def test_pitch_shift_moves_pitch_and_keeps_duration():
    sr = 24000
    wav = _tone(200, 1.0, sr)
    up = dubbing.pitch_shift(wav, sr, 3.0)
    assert abs(len(up) - len(wav)) < 0.02 * sr
    assert dubbing.semitones(dubbing.median_pitch(up, sr), 200) == pytest.approx(3.0, abs=0.4)


def test_generate_best_retries_wrong_pitch_and_corrects_the_kept_line(tmp_path, monkeypatch):
    sr = _StubTTS.sample_rate
    outputs = iter([_tone(120, 1.5, sr), _tone(230, 1.5, sr)])  # try 1 sounds like a man, try 2 like her

    class PitchModel(_StubModel):
        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return next(outputs)

    model = PitchModel()
    dubber = dubbing.VideoDubber(model, tmp_path)
    line = SubtitleLine(1, 0.0, 3.0, "hello there, how are you doing today", speaker="Ann")
    profile = dubbing.SpeakerProfile("Ann", gender="female", description="Adult female voice, medium pitch")
    best = dubber._generate_best(
        line, profile, "clone_speaker", None, None, dubbing.DubOptions(max_attempts=3), target_pitch=210.0
    )
    assert best["tries"] == 2 and best["attempt"] == 2  # the male-sounding try was rejected
    assert model.calls[0]["text"].startswith("(Adult female voice, medium pitch)")
    assert abs(best["pitch_offset"]) < 3.0


def test_fold_octave_fixes_pitch_tracker_octave_errors():
    assert dubbing.fold_octave(92.0, 205.0) == pytest.approx(184.0)  # half-pitch error on a female voice
    assert dubbing.fold_octave(410.0, 205.0) == pytest.approx(205.0)
    assert dubbing.fold_octave(250.0, 205.0) == 250.0  # a real, excited rise is kept
    assert dubbing.fold_octave(None, 205.0) is None


def test_strict_fold_only_undoes_exact_octave_jumps():
    assert dubbing.fold_octave(120.0, 210.0, strict=True) == 120.0  # a male-sounding line stays wrong
    assert dubbing.fold_octave(420.0, 210.0, strict=True) == pytest.approx(210.0)


def test_dubbing_refuses_non_voxcpm2_models(tmp_path):
    class OldTTS:
        sample_rate = 44100

    class OldModel:
        tts_model = OldTTS()

    with pytest.raises(ValueError, match="requires a VoxCPM2 model"):
        dubbing.VideoDubber(OldModel(), tmp_path)
    dubbing.VideoDubber(_StubModel(), tmp_path)  # VoxCPM2 is accepted


def test_collapse_to_two_voices_maps_every_character_to_an_adult_male_or_female_voice():
    lines = [
        SubtitleLine(1, 0, 1, "a", speaker="Alice", gender="female", emotion="happy"),
        SubtitleLine(2, 1, 2, "b", speaker="Bob", gender="male"),
        SubtitleLine(3, 2, 3, "c", speaker="Alice", gender="unknown"),  # follows Alice's gender
        SubtitleLine(4, 3, 4, "d", speaker="Kid", gender="female"),
        SubtitleLine(5, 4, 5, "e", speaker="Speaker 3", gender="unknown"),  # most common gender
    ]
    profiles = dubbing.build_speaker_profiles(lines)
    profiles = dubbing.collapse_to_two_voices(lines, profiles, speaker_style=False)

    assert set(profiles) == {"Male", "Female"}
    assert [l.speaker for l in lines] == ["Female", "Male", "Female", "Female", "Female"]
    assert all(p.voice_mode == "clone_first" for p in profiles.values())
    assert profiles["Male"].description.startswith("Adult male voice")
    assert profiles["Female"].description.startswith("Adult female voice")
    assert lines[0].emotion == "happy"  # emotion is kept per line


def test_two_voice_dub_uses_one_designed_voice_per_gender_with_per_line_emotion(tmp_path):
    lines = [
        SubtitleLine(1, 0.0, 2.0, "Good morning", speaker="Female", gender="female", emotion="happy"),
        SubtitleLine(2, 2.0, 4.0, "I am so tired", speaker="Female", gender="female", emotion="sad"),
        SubtitleLine(3, 4.0, 6.0, "Hello", speaker="Male", gender="male"),
    ]
    profiles = dubbing.collapse_to_two_voices(lines)
    model = _StubModel()  # always a 180 Hz tone: an adult female pitch, too high for an adult male
    dubber = dubbing.VideoDubber(model, tmp_path)
    opts = dubbing.DubOptions(verify_voice=False, max_attempts=2, design_attempts=2, clean_generated=False)

    female_ref = dubber.design_reference(profiles["Female"], lines, opts)
    assert len(model.calls) == 1 and model.calls[0]["reference_wav_path"] is None
    assert model.calls[0]["text"] == "(Adult female voice, mature and warm, medium pitch, calm and neutral)I am so tired"
    assert female_ref is not None

    dubber.design_reference(profiles["Male"], lines, opts)
    assert len(model.calls) == 3  # out of the adult male range -> retried with another seed
    assert model.calls[1]["seed"] != model.calls[2]["seed"]

    happy = dubber._line_text(lines[0], profiles["Female"], "design→clone", opts)
    sad = dubber._line_text(lines[1], profiles["Female"], "design→clone", opts)
    assert happy.startswith("(Adult female voice, mature and warm, medium pitch, happy")
    assert sad.startswith("(Adult female voice, mature and warm, medium pitch, sad")


def test_two_voices_are_styled_after_how_each_character_speaks():
    def line(i, speaker, f0, rate, rms_db, spread, tone=""):
        l = SubtitleLine(i, i, i + 1, "x", speaker=speaker, gender="male", tone=tone)
        l.features = {"f0_median": f0, "rate": rate, "rms_db": rms_db, "pitch_spread": spread}
        return l

    lines = [
        line(1, "Boss", 92, 2.0, -12, 1.5),
        line(2, "Boss", 95, 2.0, -12, 1.5, tone="louder and more forceful"),
        line(3, "Clerk", 145, 4.0, -24, 4.0),
        line(4, "Clerk", 148, 4.0, -24, 4.0),
        line(5, "Pal", 120, 3.0, -18, 2.5),
        line(6, "Pal", 120, 3.0, -18, 2.5),
    ]
    profiles = dubbing.collapse_to_two_voices(lines, dubbing.build_speaker_profiles(lines))

    assert set(profiles) == {"Male"} and {l.speaker for l in lines} == {"Male"}
    assert lines[0].tone == "deep pitch, steady, calm delivery, slow, deliberate pace, strong, projecting voice"
    assert lines[1].tone.endswith(", louder and more forceful")  # the line's own tone is kept after the style
    assert lines[2].tone == "light, higher pitch, expressive delivery, fast-paced, soft-spoken"
    assert lines[4].tone == "medium pitch"


def test_designed_voice_is_redesigned_until_it_matches_the_original(tmp_path, monkeypatch):
    sr = _StubTTS.sample_rate
    # try 1: adult male but far too deep for this speaker; try 2: close to the original's 140 Hz
    outputs = iter([_tone(90, 1.5, sr), _tone(138, 1.5, sr)])

    class PitchModel(_StubModel):
        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return next(outputs)

    model = PitchModel()
    dubber = dubbing.VideoDubber(model, tmp_path)
    original = np.concatenate([_tone(140, 2.0), _tone(140, 2.0)])
    lines = [SubtitleLine(1, 0.0, 2.0, "hi", speaker="Male", gender="male"),
             SubtitleLine(2, 2.0, 4.0, "there", speaker="Male", gender="male")]
    profiles = dubbing.collapse_to_two_voices(lines)
    opts = dubbing.DubOptions(verify_voice=False, clean_generated=False)
    path = dubber.design_reference(profiles["Male"], lines, opts, original)

    assert len(model.calls) == 2 and model.calls[0]["seed"] != model.calls[1]["seed"]
    assert path is not None and not dubber.warnings
    import soundfile as sf
    wav, ref_sr = sf.read(path, dtype="float32")
    assert dubbing.median_pitch(wav, ref_sr) == pytest.approx(138, abs=5)


def test_designed_voice_warns_and_corrects_pitch_when_no_try_matches(tmp_path):
    sr = _StubTTS.sample_rate

    class DeepModel(_StubModel):
        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return _tone(95, 1.5, sr)  # always far below the original's 140 Hz

    model = DeepModel()
    dubber = dubbing.VideoDubber(model, tmp_path)
    lines = [SubtitleLine(1, 0.0, 2.0, "hi", speaker="Male", gender="male")]
    profiles = dubbing.collapse_to_two_voices(lines)
    opts = dubbing.DubOptions(verify_voice=False, clean_generated=False, design_attempts=3, max_attempts=1)
    path = dubber.design_reference(profiles["Male"], lines, opts, _tone(140, 2.0))

    assert len(model.calls) == 3
    assert len(dubber.warnings) == 1 and "did not match the original" in dubber.warnings[0]
    import soundfile as sf
    wav, ref_sr = sf.read(path, dtype="float32")
    assert dubbing.median_pitch(wav, ref_sr) > 95 * 2 ** (1.5 / 12)  # nudged up towards the original


def test_reference_on_profile_replaces_the_designed_voice(tmp_path):
    model = _StubModel()
    dubber = dubbing.VideoDubber(model, tmp_path)
    lines = [SubtitleLine(1, 0.0, 2.0, "hi", speaker="Male", gender="male")]
    profile = dubbing.collapse_to_two_voices(lines)["Male"]
    profile.reference_wav = "mine.wav"
    assert dubber.design_reference(profile, lines, dubbing.DubOptions(), _tone(140, 2.0)) == "mine.wav"
    assert model.calls == []


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_clone_first_copies_the_first_line_then_clones_it_for_the_rest(tmp_path):
    video = tmp_path / "in.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "color=c=black:s=64x64:d=6",
            "-f", "lavfi", "-i", "sine=frequency=210:duration=6",
            "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-c:a", "aac", "-shortest", str(video),
        ],
        check=True,
    )
    srt = tmp_path / "in.srt"
    srt.write_text(
        "1\n00:00:00,000 --> 00:00:00,500\n[Ann] Oh!\n\n"  # too short to copy the voice from
        "2\n00:00:01,000 --> 00:00:02,800\n[Ann] This is my first real line\n\n"
        "3\n00:00:03,000 --> 00:00:04,500\n[Ann] And this one sounds the same\n\n"
        "4\n00:00:04,600 --> 00:00:06,000\n[Ann] So does this one\n",
        encoding="utf-8",
    )
    model = _StubModel()
    dubber = dubbing.VideoDubber(model, tmp_path / "work")
    lines, profiles = dubber.prepare(str(video), str(srt), detect_emotion=False, identify_speakers=False)
    assert profiles["Ann"].voice_mode == "clone_first"

    result = dubber.run(
        str(video), lines, profiles, str(tmp_path / "out" / "dub.mp4"),
        dubbing.DubOptions(verify_voice=False, max_attempts=1, first_line_attempts=1, short_line_seconds=1.0),
    )
    refs = [Path(c["reference_wav_path"]).name for c in model.calls]
    # line 2 is generated first, from several of Ann's original lines; every other line clones that generated line
    assert refs == ["speaker_Ann.wav", "first_Ann.wav", "first_Ann.wav", "first_Ann.wav"]
    assert [r["index"] for r in result.report] == [1, 2, 3, 4]
    assert [r["mode"] for r in result.report] == ["clone→#2", "clone_first", "clone→#2", "clone→#2"]
    assert len({c["seed"] for c in model.calls}) == 1
    # the short "Oh!" gets no long voice prompt (that makes the model babble); longer lines do
    assert model.calls[1]["text"] == "(spoken clearly)Oh!"
    assert model.calls[2]["text"].startswith("(Adult female voice")


def test_target_pitch_follows_the_line_relative_to_the_speakers_own_pitch(tmp_path):
    import soundfile as sf

    dubber = dubbing.VideoDubber(None, tmp_path)
    ref = tmp_path / "ref.wav"
    sf.write(str(ref), _tone(200, 1.5), 16000)  # the character's generated first line: 200 Hz
    audio = np.concatenate([_tone(250, 1.0), _tone(250 * 2 ** (2 / 12), 1.0), _tone(250 * 2 ** (8 / 12), 1.0)])
    lines = [SubtitleLine(i + 1, i, i + 1, "x", speaker="Ann") for i in range(3)]

    usual, higher, shout = (dubber._target_pitch(l, str(ref), audio, speaker_pitch=250.0) for l in lines)
    assert usual == pytest.approx(200, rel=0.03)  # the usual line keeps the reference pitch, not the original's 250 Hz
    assert dubbing.semitones(higher, 200) == pytest.approx(2.0, abs=0.3)
    assert dubbing.semitones(shout, 200) == pytest.approx(3.0, abs=0.1)  # capped so it stays the same person
    assert dubber._target_pitch(lines[2], str(ref), audio) == pytest.approx(200, rel=0.03)


def test_generate_best_retries_until_consistent_with_the_speakers_other_lines(tmp_path, monkeypatch):
    first_line = _unit([1.0, 0.0, 0.0])
    peers = _unit([1.0, 0.1, 0.0])  # the speaker's lines accepted so far
    voices = iter([
        _unit([0.5, 0.0, 0.87]),  # 0.5 to the first line: passes the loose check, not consistent
        _unit([0.52, 0.0, 0.85]),  # still drifting
        _unit([0.95, 0.05, 0.3]),  # consistent
    ])
    model = _StubModel()
    dubber = dubbing.VideoDubber(model, tmp_path)
    monkeypatch.setattr(dubber, "_embed", lambda wav, sr: next(voices))
    line = SubtitleLine(1, 0.0, 3.0, "hello there, how are you doing today", speaker="Ann")
    best = dubber._generate_best(
        line, dubbing.SpeakerProfile("Ann"), "clone→#1", "first.wav", first_line,
        dubbing.DubOptions(max_attempts=8), min_similarity=0.55, peers=peers,
    )
    assert best["tries"] == 3 and best["voice_ok"]
    assert best["consistency"] > 0.9
    assert len({c["seed"] for c in model.calls}) == 3


def test_short_clips_are_still_voice_checked(tmp_path, monkeypatch):
    sr = _StubTTS.sample_rate

    class ShortModel(_StubModel):
        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return _tone(220, 0.35, sr)  # "Mom?" - shorter than a voice embedding needs

    seen = []
    dubber = dubbing.VideoDubber(ShortModel(), tmp_path)
    monkeypatch.setattr(dubber, "_embed", lambda wav, rate: seen.append(len(wav) / rate) or _unit([1.0, 0.0]))
    line = SubtitleLine(1, 0.0, 1.1, "Mom?", speaker="Ann")
    best = dubber._generate_best(
        line, dubbing.SpeakerProfile("Ann"), "clone→#1", "first.wav", _unit([1.0, 0.0]),
        dubbing.DubOptions(clean_generated=False),
    )
    assert seen and min(seen) >= 1.5  # looped up to a checkable length
    assert best["similarity"] == pytest.approx(1.0)


def test_loop_to_length_repeats_short_clips_only():
    sr = 1000
    short = np.ones(300, dtype=np.float32)
    looped = dubbing.loop_to_length(short, sr, 1.5)
    assert len(looped) == 1500 and looped[350] == 0.0 and looped[450] == 1.0
    long = np.ones(2000, dtype=np.float32)
    assert dubbing.loop_to_length(long, sr, 1.5) is long


def test_runaway_or_silent_tries_are_rejected_and_retried(tmp_path):
    sr = _StubTTS.sample_rate
    outputs = iter([_tone(220, 6.0, sr), _tone(220, 0.05, sr), _tone(220, 0.6, sr)])  # babble, blip, the word

    class Model(_StubModel):
        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return next(outputs)

    model = Model()
    dubber = dubbing.VideoDubber(model, tmp_path)
    line = SubtitleLine(5, 0.0, 1.1, "Mom?", speaker="Ann", emotion="surprised")
    profile = dubbing.SpeakerProfile("Ann", description="Adult female voice, high, bright pitch")
    best = dubber._generate_best(line, profile, "clone→#1", None, None, dubbing.DubOptions(clean_generated=False))

    assert best["tries"] == 3 and best["voice_ok"] and best["duration"] < 1.0
    assert model.calls[0]["text"] == "(surprised, spoken clearly)Mom?"  # short line: emotion only, no long voice prompt
    assert all(c["max_len"] for c in model.calls)  # generation length is capped


def test_estimate_speech_seconds_handles_khmer_cjk_and_latin():
    assert dubbing.estimate_speech_seconds("គាត់...") < 0.5  # one Khmer word ("he")
    assert 1.5 < dubbing.estimate_speech_seconds("តែសម្រាប់ម៉ាក់ វាប្រហាក់ប្រហែលគ្នានឹងស្លាប់ដែរ។") < 3.5
    assert dubbing.estimate_speech_seconds("你好") == pytest.approx(2 / 4.5)
    assert dubbing.estimate_speech_seconds("hello there") == pytest.approx(2 / 2.7)


def test_a_one_word_line_in_a_long_slot_is_short_and_read_clearly(tmp_path):
    sr = _StubTTS.sample_rate
    outputs = iter([_tone(220, 3.3, sr), _tone(220, 0.6, sr)])  # mumbling on after the word, then the word

    class Model(_StubModel):
        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return next(outputs)

    model = Model()
    dubber = dubbing.VideoDubber(model, tmp_path)
    line = SubtitleLine(7, 15.8, 17.5, "គាត់...", speaker="Wen Shu", emotion="sad")  # 1.7 s slot
    profile = dubbing.SpeakerProfile("Wen Shu", description="Adult woman's voice")
    best = dubber._generate_best(line, profile, "clone→#2", None, None, dubbing.DubOptions(clean_generated=False))

    assert best["tries"] == 2 and best["duration"] < 1.0  # the 3.3 s take was rejected as babble
    assert model.calls[0]["text"] == "(sad and downcast, spoken clearly)គាត់."
    assert model.calls[0]["max_len"] < 15  # about 2 s at most for one word


def _bursts(n, sr, word=0.25, pause=0.2):
    piece = np.concatenate([_tone(220, word, sr), np.zeros(int(pause * sr), dtype=np.float32)])
    return np.concatenate([piece] * n)[: -int(pause * sr)]


def test_sound_bursts_counts_separate_utterances():
    sr = 24000
    assert len(dubbing.sound_bursts(_bursts(1, sr), sr)) == 1
    assert len(dubbing.sound_bursts(_bursts(4, sr), sr)) == 4
    assert dubbing.allowed_bursts(0.25) == 2 and dubbing.allowed_bursts(1.0) == 5


def test_a_stuttered_short_line_is_redone_or_trimmed_to_the_word(tmp_path):
    sr = _StubTTS.sample_rate

    class Stutter(_StubModel):
        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return _bursts(5, sr) if len(self.calls) < 3 else _bursts(1, sr)  # stutters twice, then the word

    model = Stutter()
    dubber = dubbing.VideoDubber(model, tmp_path)
    line = SubtitleLine(7, 0.0, 1.7, "គាត់...", speaker="Wen Shu", emotion="sad")
    best = dubber._generate_best(line, dubbing.SpeakerProfile("Wen Shu"), "clone→#2", None, None,
                                 dubbing.DubOptions(clean_generated=False))
    assert best["tries"] == 3 and best["bursts"] == 1 and not best["trimmed"]

    class AlwaysStutter(_StubModel):
        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return _bursts(5, sr)

    dubber = dubbing.VideoDubber(AlwaysStutter(), tmp_path)
    best = dubber._generate_best(line, dubbing.SpeakerProfile("Wen Shu"), "clone→#2", None, None,
                                 dubbing.DubOptions(clean_generated=False, max_attempts=2))
    assert best["trimmed"] and len(dubbing.sound_bursts(best["wav"], sr)) <= dubbing.allowed_bursts(
        dubbing.estimate_speech_seconds("គាត់..."))


def test_remove_silences_trims_edges_and_shortens_long_pauses():
    sr = 24000
    silence = lambda sec: np.zeros(int(sec * sr), dtype=np.float32)
    wav = np.concatenate([silence(0.5), _tone(220, 0.4, sr), silence(1.0), _tone(220, 0.4, sr), silence(0.08),
                          _tone(220, 0.3, sr), silence(0.7)])
    out = dubbing.remove_silences(wav, sr, max_pause=0.2)

    # 1.1 s of speech + the long pause cut to ~0.2 s + the short 0.08 s pause kept + tiny edge margins
    assert len(out) / sr == pytest.approx(1.1 + 0.2 + 0.08 + 0.05, abs=0.06)
    bursts = dubbing.sound_bursts(out, sr, gap=0.15)
    assert len(bursts) == 2  # the long pause is still a pause, the 0.08 s one is untouched
    assert abs(out[0]) < 1e-3 and abs(out[-1]) < 1e-3  # faded edges, no clicks
