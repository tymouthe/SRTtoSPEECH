import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("voxcpm")
pkg.__path__ = [str(ROOT / "src" / "voxcpm")]
sys.modules.setdefault("voxcpm", pkg)

from voxcpm import video_editor  # noqa: E402

SR = 16_000


def _speechy(seconds, sr=SR, seed=0):
    """A voice-like sound: a 180 Hz buzz with harmonics, wobbling in level like syllables."""
    t = np.arange(int(seconds * sr)) / sr
    buzz = sum(np.sin(2 * np.pi * 180 * k * t) / k for k in range(1, 8))
    syllables = 0.6 + 0.4 * np.sin(2 * np.pi * 4 * t + seed)
    return (0.2 * buzz * syllables).astype(np.float32)


def _soundtrack(parts, total, sr=SR):
    rng = np.random.default_rng(0)
    out = (0.002 * rng.standard_normal(int(total * sr))).astype(np.float32)
    for start, seconds in parts:
        a = int(start * sr)
        out[a:a + int(seconds * sr)] += _speechy(seconds, sr)
    return out


def test_detect_speech_finds_each_spoken_stretch_with_its_timing():
    wav = _soundtrack([(1.0, 1.5), (4.0, 2.0), (8.0, 0.8)], total=10.0)
    regions = video_editor.detect_speech(wav, SR)
    assert len(regions) == 3
    for (a, b), (start, seconds) in zip(regions, [(1.0, 1.5), (4.0, 2.0), (8.0, 0.8)]):
        assert abs(a - start) < 0.12
        assert abs(b - (start + seconds)) < 0.12


def test_detect_speech_merges_short_pauses_and_ignores_silence():
    wav = _soundtrack([(1.0, 1.0), (2.15, 1.0)], total=5.0)
    assert len(video_editor.detect_speech(wav, SR)) == 1
    assert video_editor.detect_speech(np.zeros(SR * 3, dtype=np.float32), SR) == []


def test_align_to_speech_matches_lines_to_the_speech_they_cover():
    regions = [(1.1, 2.4), (4.3, 6.0), (9.0, 9.5)]
    lines = [(1, 1.0, 2.5), (2, 4.0, 6.2), (3, 6.5, 7.0)]
    starts = video_editor.align_to_speech(lines, regions)
    assert starts == {1: 1.1, 2: 4.3}  # line 3 has no speech near it


def test_lines_sharing_one_stretch_of_speech_keep_their_spacing():
    starts = video_editor.align_to_speech([(1, 1.0, 1.8), (2, 1.9, 2.6)], [(1.3, 2.9)])
    assert starts == {1: 1.3, 2: 2.2}


def test_render_mix_places_trims_mutes_and_ducks_clips():
    sr = 1000
    one = np.ones(1000, dtype=np.float32) * 0.5
    clips = [
        ({"start": 1.0, "gain_db": 0.0, "trim_in": 0.2, "trim_out": 0.3}, one),
        ({"start": 3.0, "muted": True}, one),
    ]
    original = np.full(5000, 0.1, dtype=np.float32)
    mixed = video_editor.render_mix(clips, 5.0, sr, original=original, mode="duck")
    assert len(mixed) == 5000
    assert abs(mixed[500] - 0.1) < 1e-3  # original kept where no clip plays
    assert mixed[1200] > 0.5  # the clip (0.5 s long after trimming) over the ducked original
    assert mixed[1600] < 0.15  # past the trimmed end
    assert abs(mixed[3500] - 0.1) < 1e-3  # a muted clip is left out

    silent = video_editor.render_mix(clips, 5.0, sr, original=original, mode="mute")
    assert abs(silent[500]) < 1e-6 and abs(silent[1200] - 0.5) < 1e-3


def test_waveform_peaks_are_scaled_to_one():
    peaks = video_editor.waveform_peaks(_soundtrack([(0.5, 1.0)], total=2.0), SR, per_second=10)
    assert len(peaks) == 20
    assert max(peaks) == 1.0 and peaks[0] < 0.1


def test_editor_state_round_trips(tmp_path):
    state = video_editor.load_state(tmp_path)
    assert state["clips"] == {} and state["mix"]["original"] == "replace"
    state["clips"]["3"] = video_editor.default_clip(1.23456)
    video_editor.save_state(tmp_path, state)
    assert video_editor.load_state(tmp_path)["clips"]["3"]["start"] == 1.235


def test_prepare_video_accepts_an_audio_file(tmp_path):
    import shutil

    import pytest
    import soundfile as sf

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is not installed")
    wav = tmp_path / "episode.wav"
    sf.write(str(wav), _soundtrack([(1.0, 1.5), (4.0, 2.0)], total=7.0), SR)
    info = video_editor.prepare_video(str(wav), tmp_path / "video")
    assert info["has_video"] is False and info["has_audio"] is True
    assert info["preview"].endswith("source.wav")  # plays in the browser as it is
    assert abs(info["duration"] - 7.0) < 0.05
    assert len(info["regions"]) == 2


def test_pieces_of_a_cut_line_play_their_own_part_and_collapse_back():
    sr = 1000
    ramp = np.linspace(0.1, 0.9, 1000).astype(np.float32)  # each sample says where in the line it is
    clips = {
        "7": {"start": 0.0, "trim_in": 0.0, "trim_out": 0.6, "line": 7},  # first 0.4 s
        "7~1": {"start": 2.0, "trim_in": 0.4, "trim_out": 0.0, "line": 7, "lane": 1},  # the rest, moved later
    }
    assert video_editor.clip_line("7~1", clips["7~1"]) == 7 and video_editor.clip_line("9", {}) == 9
    mixed = video_editor.render_mix([(c, ramp) for c in clips.values()], 3.0, sr, None, "mute")
    assert abs(mixed[100] - ramp[100]) < 1e-3
    assert abs(mixed[500]) < 1e-6  # the gap between the pieces
    assert abs(mixed[2000] - ramp[400]) < 1e-3

    one = video_editor.collapse_pieces(dict(clips, **{"8": {"start": 5.0}}), [7])
    assert set(one) == {"7", "8"}
    assert one["7"]["trim_in"] == 0.0 and one["7"]["trim_out"] == 0.0 and one["7"]["start"] == 0.0


def test_clip_speed_changes_length_and_keeps_pitch(tmp_path):
    import shutil

    import pytest
    import soundfile as sf

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is not installed")
    sr = 16_000
    t = np.arange(sr * 2) / sr
    tone = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    def pitch(wav):
        spectrum = np.abs(np.fft.rfft(wav))
        return np.argmax(spectrum) * sr / len(wav)

    fast = video_editor.clip_audio(tone, sr, {"speed": 2.0, "trim_out": 1.0})  # 1 s of audio, twice as fast
    assert abs(len(fast) / sr - 0.5) < 0.05
    assert abs(pitch(fast) - 220) < 10
    slow = video_editor.clip_audio(tone, sr, {"speed": 0.5})
    assert abs(len(slow) / sr - 4.0) < 0.1
    assert abs(pitch(slow) - 220) < 10
    assert video_editor.clip_speed({"speed": 9}) == 2.0 and video_editor.clip_speed({}) == 1.0

    raw = tmp_path / "0003.wav"
    sf.write(str(raw), tone, sr)
    out = video_editor.stretched_file(raw, 1.25, tmp_path / "stretched")
    assert out.exists() and abs(sf.info(str(out)).duration - 1.6) < 0.05
    assert video_editor.stretched_file(raw, 1.25, tmp_path / "stretched") == out  # cached


def test_replace_mode_silences_the_original_speech_each_new_voice_replaces():
    sr = 1000
    voice = np.full(1000, 0.5, dtype=np.float32)  # 1 s
    clips = [({"start": 1.0}, voice)]
    original = np.full(8000, 0.1, dtype=np.float32)
    # Original speech 0.8-2.6 s (a new voice covers it) and 5-6 s (nothing replaces it).
    regions = [(0.8, 2.6), (5.0, 6.0)]
    mixed = video_editor.render_mix(clips, 8.0, sr, original=original, mode="replace", regions=regions)
    assert abs(mixed[1500] - 0.5) < 1e-3  # only the new voice
    assert abs(mixed[2300]) < 1e-3  # the end of the original line is silenced too
    assert abs(mixed[4000] - 0.1) < 1e-3  # music / effects in the gap stay
    assert abs(mixed[5500] - 0.1) < 1e-3  # original speech nobody replaced stays
    assert video_editor.replaced_spans([(1.0, 2.0)], regions) == [(0.8, 2.6)]


def _frame_with_subtitle(seed, height=480, width=270, text=True):
    rng = np.random.default_rng(seed)
    frame = rng.uniform(40, 120, (height, width)).astype(np.float32)  # a busy, darker scene
    if text:  # white strokes of a subtitle line at 76-79 % of the height
        for x in range(60 + seed % 7, 210, 9):
            frame[365:380, x:x + 4] = 250.0
    return frame


def test_find_subtitle_box_finds_the_burned_in_line():
    frames = np.stack([_frame_with_subtitle(i, text=i % 5 != 0) for i in range(20)])
    x, y, w, h = video_editor.find_subtitle_box(frames)
    assert y < 365 / 480 and y + h > 380 / 480  # covers the text rows
    assert x < 60 / 270 and x + w > 214 / 270  # and the whole line
    assert abs((x + w / 2) - 0.5) < 1e-3  # centred: longer lines fit too
    assert video_editor.find_subtitle_box(np.stack([_frame_with_subtitle(i, text=False) for i in range(20)])) is None


def test_subtitle_filter_stays_inside_the_picture():
    assert video_editor.subtitle_filter("off", [0.1, 0.7, 0.8, 0.1], 1080, 1920) is None
    fill = video_editor.subtitle_filter("fill", [0.0, 0.9, 1.0, 0.2], 1080, 1920)
    assert fill.startswith("[0:v]delogo=x=2:y=1728:w=1076:h=190")
    assert "gblur" in video_editor.subtitle_filter("blur", [0.1, 0.7, 0.8, 0.1], 1080, 1920)


def test_export_hides_subtitles_at_the_original_size(tmp_path):
    import shutil
    import subprocess

    import pytest
    import soundfile as sf

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is not installed")
    src = tmp_path / "in.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=360x640:rate=25", "-t", "2", "-c:v",
                    "libx264", "-pix_fmt", "yuv420p", str(src)], check=True)
    sf.write(str(tmp_path / "a.wav"), np.zeros(2 * 16000, dtype=np.float32), 16000)
    out = video_editor.export_video(str(src), str(tmp_path / "a.wav"), tmp_path / "out.mp4",
                                    subtitles={"mode": "fill", "box": [0.1, 0.75, 0.8, 0.08]})
    info = video_editor.probe_video(out)
    assert (info["width"], info["height"], info["codec"]) == (360, 640, "h264")
    assert abs(info["fps"] - 25) < 0.01


def test_subtitle_overlays_show_each_image_only_during_its_time():
    inputs, graph, last = video_editor.subtitle_overlays("clean", [(1.0, 2.5, 0, 1400, "a.png"), (3.0, 4.0, 0, 1380, "b.png")])
    assert inputs == ["-i", "a.png", "-i", "b.png"]
    assert graph == (
        "[clean][2:v]overlay=0:1400:eof_action=repeat:enable='between(t,1.000,2.500)'[s0];"
        "[s0][3:v]overlay=0:1380:eof_action=repeat:enable='between(t,3.000,4.000)'[s1]"
    )
    assert last == "s1"


def test_export_draws_new_subtitles_at_the_original_size(tmp_path):
    import shutil
    import subprocess

    import pytest
    import soundfile as sf
    from PIL import Image

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is not installed")
    src = tmp_path / "in.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=black:size=360x640:rate=25", "-t", "2", "-c:v",
                    "libx264", "-pix_fmt", "yuv420p", str(src)], check=True)
    sf.write(str(tmp_path / "a.wav"), np.zeros(2 * 16000, dtype=np.float32), 16000)
    Image.new("RGBA", (360, 40), (255, 255, 255, 255)).save(tmp_path / "sub.png")  # a white "subtitle" band
    out = video_editor.export_video(str(src), str(tmp_path / "a.wav"), tmp_path / "out.mp4",
                                    subtitles={"mode": "fill", "box": [0.1, 0.75, 0.8, 0.08]},
                                    overlays=[(1.0, 2.0, 0, 500, str(tmp_path / "sub.png"))])
    info = video_editor.probe_video(out)
    assert (info["width"], info["height"], info["codec"]) == (360, 640, "h264")

    def pixel(t):
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(out), "-frames:v", "1", "-vf",
                              "format=gray,crop=2:2:180:520", "-f", "rawvideo", "-"], capture_output=True).stdout
        return raw[0]

    assert pixel(0.5) < 40 and pixel(1.5) > 200  # only shown during its time


def test_level_voices_brings_every_clip_to_the_same_loudness():
    sr = 16_000
    quiet = _speechy(1.5) * 0.05  # a whisper
    loud = _speechy(1.5) * 2.0  # a shout (would clip)
    pause = np.zeros(sr, dtype=np.float32)
    quiet_with_tail = np.concatenate([quiet, pause])  # the pause does not count
    levels = []
    for wav in (quiet_with_tail, loud, _speechy(1.5)):
        part = video_editor.clip_audio(wav, sr, {}, level=True)
        levels.append(video_editor.speech_level(part, sr))
        assert np.abs(part).max() <= 1.0  # louder peaks bent under full scale, never clipped
    assert max(levels) - min(levels) < 1.5, levels
    assert abs(np.median(levels) - video_editor.LEVEL_TARGET_DB) < 1.5
    # The clip's own volume is a change on top of the levelling.
    plus6 = video_editor.clip_audio(quiet, sr, {"gain_db": 6}, level=True)
    assert abs(video_editor.speech_level(plus6, sr) - video_editor.speech_level(video_editor.clip_audio(quiet, sr, {}, level=True), sr) - 6) < 0.5
    assert video_editor.speech_level(np.zeros(sr, dtype=np.float32), sr) is None


def test_level_voices_also_evens_out_a_line_from_the_inside():
    sr = 16_000
    loud_then_quiet = np.concatenate([_speechy(1.0) * 1.0, _speechy(1.0, seed=1) * 0.15])  # 16 dB apart
    first, second = loud_then_quiet[: sr], loud_then_quiet[sr:]
    before = video_editor.speech_level(first, sr) - video_editor.speech_level(second, sr)
    evened = video_editor.clip_audio(loud_then_quiet, sr, {}, level=True)
    after = video_editor.speech_level(evened[200:sr - 4000], sr) - video_editor.speech_level(evened[sr + 4000:], sr)
    assert before > 15 and after < 7, (before, after)  # 3:1 (at most 9 dB each way), glides between words
    silence = np.zeros(sr, dtype=np.float32)
    assert np.abs(video_editor.even_out(np.concatenate([_speechy(1.0), silence]), sr)[-sr // 2:]).max() == 0  # pauses stay silent
