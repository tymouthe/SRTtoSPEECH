import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("voxcpm")
pkg.__path__ = [str(ROOT / "src" / "voxcpm")]
sys.modules.setdefault("voxcpm", pkg)

from voxcpm import dubbing, script_voice  # noqa: E402
from voxcpm.model.voxcpm2 import VoxCPM2Model  # noqa: E402

SCRIPT = """1
00:00:00,200 --> 00:00:02,800
[Dara|male|kid|sad] ម៉ាក់ កូននឹកម៉ាក់ណាស់

2
00:00:03,000 --> 00:00:05,400
[Srey|female|adult] I am so glad you are home!

3
00:00:05,600 --> 00:00:07,600
[Dara] Where is dad?!

4
00:00:08,000 --> 00:00:10,000
Once upon a time...
"""


def _tone(freq, seconds, sr):
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_parse_voice_tag_reads_name_gender_age_and_emotion_in_any_order():
    tag = script_voice.parse_voice_tag("Dara|kid|male|Sad")
    assert (tag.name, tag.gender, tag.age, tag.emotion) == ("Dara", "male", "kid", "sad")
    girl = script_voice.parse_voice_tag("Mina|girl")
    assert (girl.gender, girl.age, girl.emotion) == ("female", "kid", None)
    assert script_voice.parse_voice_tag("Bob").gender is None
    assert script_voice.parse_voice_tag(None).name is None


def test_read_script_tags_speakers_once_and_guesses_missing_emotions():
    lines, profiles = script_voice.read_script(dubbing.parse_srt(SCRIPT))

    assert [l.speaker for l in lines] == ["Dara", "Srey", "Dara", "Narrator"]
    assert (profiles["Dara"].gender, profiles["Dara"].age) == ("male", "kid")
    assert (profiles["Srey"].gender, profiles["Srey"].age) == ("female", "adult")
    assert profiles["Dara"].description.startswith("Young boy's voice")
    assert all(p.voice_mode == "clone_first" for p in profiles.values())
    assert [l.emotion for l in lines] == ["sad", "happy", "surprised", "hesitant"]
    assert lines[1].features["emotion_guessed"] and "emotion_guessed" not in lines[0].features


def test_tagged_srt_round_trips():
    lines, profiles = script_voice.read_script(dubbing.parse_srt(SCRIPT))
    tagged = script_voice.format_tagged_srt(lines, profiles)
    assert "[Dara|male|kid|surprised] Where is dad?!" in tagged
    again, again_profiles = script_voice.read_script(dubbing.parse_srt(tagged))
    assert [(l.speaker, l.emotion, l.text) for l in again] == [(l.speaker, l.emotion, l.text) for l in lines]
    assert again_profiles["Dara"].age == "kid"
    assert not any(l.features.get("emotion_guessed") for l in again)  # every emotion is now written out


class _StubTTS(VoxCPM2Model):
    sample_rate = 24000
    _encode_sample_rate, patch_size, chunk_size = 16000, 4, 640

    def __init__(self):
        pass


class _StubModel:
    """A kid's pitch for Dara, an adult woman's for everyone else; the first kid try is too low."""

    tts_model = _StubTTS()

    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        kid = "boy" in kwargs["text"]
        designing = kid and kwargs["reference_wav_path"] is None
        too_low = designing and len([c for c in self.calls if "boy" in c["text"]]) == 1
        return _tone(140 if too_low else 300 if kid else 210, 1.5, self.tts_model.sample_rate)


def test_voice_script_makes_one_neutral_voice_per_speaker_and_clones_it_for_every_line(tmp_path):
    lines, profiles = script_voice.read_script(dubbing.parse_srt(SCRIPT))
    model = _StubModel()
    voicer = script_voice.ScriptVoicer(model, tmp_path / "work")
    opts = dubbing.DubOptions(verify_voice=False, clean_generated=False, max_attempts=1)
    result = voicer.voice(lines, profiles, str(tmp_path / "out" / "script.wav"), opts)

    designs = [c for c in model.calls if c["reference_wav_path"] is None]
    dara_designs = [c for c in designs if "boy" in c["text"]]
    assert len(dara_designs) == 2  # the first try was below the kid range and was redone
    # a calm, neutral reading of Dara's first lines joined together (line 1 alone is too short)
    assert "calm and neutral" in dara_designs[0]["text"]
    assert "ម៉ាក់ កូននឹកម៉ាក់ណាស់ Where is dad?!" in dara_designs[0]["text"]
    assert len(designs) == 4  # Dara (twice), Srey, Narrator: one voice each, nothing else is designed

    # every line, the first one too, clones its speaker's voice and adds its own emotion
    line_calls = [c for c in model.calls if c["reference_wav_path"] is not None]
    assert len(line_calls) == 4
    assert line_calls[0]["reference_wav_path"].endswith("voice_Dara.wav") and "sad" in line_calls[0]["text"]
    assert line_calls[2]["reference_wav_path"].endswith("voice_Dara.wav") and "surprised" in line_calls[2]["text"]
    by_index = {r["index"]: r for r in result.report}
    assert by_index[1]["mode"] == by_index[3]["mode"] == "clone→#1 (neutral)"
    assert Path(result.audio_path).exists()
    assert len(list(Path(result.lines_dir).glob("*.wav"))) == 4
    assert "[Dara|male|kid|sad]" in Path(result.tagged_srt_path).read_text(encoding="utf-8")


def test_combined_track_shifts_long_lines_instead_of_squeezing_them(tmp_path):
    script = (
        "1\n00:00:00,000 --> 00:00:01,000\n[Ann|female] First\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\n[Ann] Second\n"
    )
    lines, profiles = script_voice.read_script(dubbing.parse_srt(script))
    voicer = script_voice.ScriptVoicer(_StubModel(), tmp_path / "work")  # every clip is 1.5 s
    opts = dubbing.DubOptions(verify_voice=False, clean_generated=False, max_attempts=1, max_speedup=1.1)
    result = voicer.voice(lines, profiles, str(tmp_path / "script.wav"), opts)

    first, second = result.report
    assert first["speedup"] == pytest.approx(1.1) and first["shift_s"] == 0
    assert second["shift_s"] == pytest.approx(1.5 / 1.1 + 0.15 - 1.0, abs=0.03)
    assert any("moves later lines" in w for w in result.warnings)


def test_line_files_keep_their_natural_length_unless_padding_is_asked_for(tmp_path):
    import soundfile as sf

    script = "1\n00:00:00,000 --> 00:00:03,000\n[Ann|female] Hi\n"  # 3 s slot, every clip is 1.5 s
    for pad, expected in ((False, 1.5), (True, 3.0)):
        lines, profiles = script_voice.read_script(dubbing.parse_srt(script))
        voicer = script_voice.ScriptVoicer(_StubModel(), tmp_path / f"work_{pad}")
        opts = dubbing.DubOptions(verify_voice=False, clean_generated=False, max_attempts=1, pad_to_slot=pad)
        result = voicer.voice(lines, profiles, str(tmp_path / f"out_{pad}" / "script.wav"), opts)
        wav, sr = sf.read(next(Path(result.lines_dir).glob("*.wav")), dtype="float32")
        assert len(wav) / sr == pytest.approx(expected, abs=0.02)
        if pad:
            assert np.abs(wav[int(1.6 * sr):]).max() == 0  # the added part is silent

    assert len(script_voice.pad_with_silence(np.ones(10, dtype=np.float32), 10, 0.5)) == 10  # never shortened


def test_a_line_that_drifts_to_the_other_genders_pitch_is_redone(tmp_path):
    sr = _StubTTS.sample_rate
    outputs = iter([_tone(210, 1.5, sr), _tone(160, 1.5, sr), _tone(205, 1.5, sr)])  # voice, male-ish try, fine

    class Model(_StubModel):
        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return next(outputs)

    script = "1\n00:00:00,000 --> 00:00:02,000\n[Ann|female|adult] Hello there\n"
    lines, profiles = script_voice.read_script(dubbing.parse_srt(script))
    model = Model()
    voicer = script_voice.ScriptVoicer(model, tmp_path / "work")
    opts = dubbing.DubOptions(verify_voice=False, clean_generated=False, max_attempts=3, first_line_attempts=1)
    result = voicer.voice(lines, profiles, str(tmp_path / "script.wav"), opts)

    assert len(model.calls) == 3  # voice + a try at 160 Hz (sounds male) that was redone + the kept one
    assert result.report[0]["gender_ok"] and result.report[0]["pitch_hz"] == pytest.approx(205, abs=8)
    assert profiles["Ann"].description.startswith("Adult woman's voice, clearly feminine")


def test_find_srt_problems_reports_blocks_with_a_broken_timing_line(tmp_path):
    srt = tmp_path / "s.srt"
    srt.write_text(
        "42\n00:02:35,200 --> 00:02:38,500\n[Su Yaxin|female] fine\n\n"
        "43\n00:02:42,00:02:44,500\n[Su Yaxin|female|adult|pleading] broken\n",
        encoding="utf-8",
    )
    problems = script_voice.find_srt_problems(srt)
    assert len(problems) == 1
    assert "Block 43 was skipped" in problems[0] and "00:02:42,00:02:44,500" in problems[0]
    assert len(script_voice.load_script(srt)[0]) == 1


def test_regenerate_redoes_only_the_chosen_lines_with_new_seeds(tmp_path):
    lines, profiles = script_voice.read_script(dubbing.parse_srt(SCRIPT))
    model = _StubModel()
    voicer = script_voice.ScriptVoicer(model, tmp_path / "work")
    opts = dubbing.DubOptions(verify_voice=False, clean_generated=False, max_attempts=1)
    out = str(tmp_path / "out" / "script.wav")
    first = voicer.voice(lines, profiles, out, opts)
    kept = {p.name: p.stat().st_mtime_ns for p in Path(first.lines_dir).glob("*.wav") if not p.name.startswith("0003_")}
    line3_seed = next(c["seed"] for c in model.calls if c["reference_wav_path"] and "Where is dad" in c["text"])

    model.calls.clear()
    lines, profiles = script_voice.read_script(dubbing.parse_srt(SCRIPT))  # e.g. rebuilt from the edited tables
    again = script_voice.ScriptVoicer(model, tmp_path / "work").regenerate(lines, profiles, out, [3], opts)

    assert len(model.calls) == 1  # only line 3; the speaker voices are reused, not designed again
    assert model.calls[0]["reference_wav_path"].endswith("voice_Dara.wav")
    assert model.calls[0]["seed"] != line3_seed
    by_index = {r["index"]: r for r in again.report}
    assert by_index[3]["regenerated"] == 1 and by_index[1]["regenerated"] == 0
    assert len(again.report) == 4 and Path(again.audio_path).exists()
    assert {p.name: p.stat().st_mtime_ns for p in Path(again.lines_dir).glob("*.wav") if not p.name.startswith("0003_")} == kept


def test_regenerate_needs_a_first_run(tmp_path):
    lines, profiles = script_voice.read_script(dubbing.parse_srt(SCRIPT))
    voicer = script_voice.ScriptVoicer(_StubModel(), tmp_path / "work")
    with pytest.raises(ValueError, match="Generate all lines first"):
        voicer.regenerate(lines, profiles, str(tmp_path / "x.wav"), [1], dubbing.DubOptions(verify_voice=False))


def test_failed_lines_lists_inconsistent_and_unclear_gender_lines():
    report = [
        {"index": 1, "voice_ok": True, "gender_ok": True},
        {"index": 2, "voice_ok": False, "gender_ok": True},
        {"index": 3, "voice_ok": True, "gender_ok": False},
    ]
    assert script_voice.failed_lines(report) == [2, 3]


def test_revoice_speaker_designs_a_new_voice_and_redoes_all_their_lines(tmp_path):
    lines, profiles = script_voice.read_script(dubbing.parse_srt(SCRIPT))
    model = _StubModel()
    voicer = script_voice.ScriptVoicer(model, tmp_path / "work")
    opts = dubbing.DubOptions(verify_voice=False, clean_generated=False, max_attempts=1, first_line_attempts=1)
    out = str(tmp_path / "out" / "script.wav")
    voicer.voice(lines, profiles, out, opts)
    old_design = next(c for c in model.calls if c["reference_wav_path"] is None and "Srey" not in c["text"] and "woman" in c["text"])

    model.calls.clear()
    lines, profiles = script_voice.read_script(dubbing.parse_srt(SCRIPT))
    profiles["Srey"].description = "Adult woman's voice, husky and low"  # e.g. edited in the speakers table
    result = script_voice.revoice_speakers(
        script_voice.ScriptVoicer(model, tmp_path / "work"), lines, profiles, out, ["Srey"], opts
    )

    designs = [c for c in model.calls if c["reference_wav_path"] is None]
    assert len(designs) == 1 and "husky and low" in designs[0]["text"]  # only Srey gets a new voice...
    assert designs[0]["seed"] != old_design["seed"]  # ...designed with a new seed
    redone = [c for c in model.calls if c["reference_wav_path"]]
    assert len(redone) == 1 and redone[0]["reference_wav_path"].endswith("voice_Srey.wav")  # all (1) of her lines
    by_index = {r["index"]: r for r in result.report}
    assert by_index[2]["regenerated"] == 1 and by_index[1]["regenerated"] == 0
    assert set(result.voices) == {"Dara", "Srey", "Narrator"}


def test_line_files_have_their_silences_removed_by_default(tmp_path):
    import soundfile as sf

    sr = _StubTTS.sample_rate

    class Padded(_StubModel):
        def generate(self, **kwargs):
            self.calls.append(kwargs)
            z = np.zeros(int(0.6 * sr), dtype=np.float32)
            return np.concatenate([_tone(210, 0.6, sr), z, _tone(210, 0.6, sr)])

    script = "1\n00:00:00,000 --> 00:00:03,000\n[Ann|female] Hello there, how are you\n"
    for remove, expected in ((True, 1.2 + 0.2 + 0.05), (False, 1.8)):
        lines, profiles = script_voice.read_script(dubbing.parse_srt(script))
        voicer = script_voice.ScriptVoicer(Padded(), tmp_path / f"w{remove}")
        opts = dubbing.DubOptions(verify_voice=False, clean_generated=False, max_attempts=1, remove_silence=remove)
        result = voicer.voice(lines, profiles, str(tmp_path / f"o{remove}" / "s.wav"), opts)
        wav, rate = sf.read(next(Path(result.lines_dir).glob("*.wav")), dtype="float32")
        assert len(wav) / rate == pytest.approx(expected, abs=0.08)
        assert result.report[0]["generated_s"] == pytest.approx(expected, abs=0.08)


def test_progress_is_saved_per_line_and_results_can_be_reloaded(tmp_path):
    lines, profiles = script_voice.read_script(dubbing.parse_srt(SCRIPT))

    class Crash(_StubModel):
        def generate(self, **kwargs):
            if len([c for c in self.calls if c["reference_wav_path"]]) >= 2 and kwargs["reference_wav_path"]:
                raise RuntimeError("stopped")
            return super().generate(**kwargs)

    workdir, out = tmp_path / "work", tmp_path / "work" / "script.wav"
    opts = dubbing.DubOptions(verify_voice=False, clean_generated=False, max_attempts=1)
    with pytest.raises(RuntimeError):
        script_voice.ScriptVoicer(Crash(), workdir).voice(lines, profiles, str(out), opts)
    assert script_voice.generated_lines(workdir) == {1, 2}  # kept although the run stopped
    assert script_voice.load_results(workdir, out) is None  # no finished run yet

    lines, profiles = script_voice.read_script(dubbing.parse_srt(SCRIPT))
    script_voice.ScriptVoicer(_StubModel(), workdir).regenerate(lines, profiles, str(out), [3, 4], opts)
    loaded = script_voice.load_results(workdir, out)
    assert loaded is not None and [r["index"] for r in loaded.report] == [1, 2, 3, 4]
    assert Path(loaded.audio_path).exists() and set(loaded.voices) == {"Dara", "Srey", "Narrator"}
