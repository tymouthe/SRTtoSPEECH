import os
import re
import sys
import tempfile
import logging
import random
import numpy as np
import gradio as gr
from typing import Optional, Tuple
from funasr import AutoModel
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import voxcpm
from voxcpm.model.utils import resolve_runtime_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------- Inline i18n (en + zh-CN only) ----------

_USAGE_INSTRUCTIONS_EN = (
    "**VoxCPM2 — Three Modes of Speech Generation:**\n\n"
    "🎨 **Voice Design** — Create a brand-new voice  \n"
    "No reference audio required. Describe the desired voice characteristics "
    "(gender, age, tone, emotion, pace …) in **Control Instruction**, and VoxCPM2 "
    "will craft a unique voice from your description alone.\n\n"
    "🎛️ **Controllable Cloning** — Clone a voice with optional style guidance  \n"
    "Upload a reference audio clip, then use **Control Instruction** to steer "
    "emotion, speaking pace, and overall style while preserving the original timbre.\n\n"
    "🎙️ **Ultimate Cloning** — Reproduce every vocal nuance through audio continuation  \n"
    "Turn on **Ultimate Cloning Mode** and provide (or auto-transcribe) the reference audio's transcript. "
    "The model treats the reference clip as a spoken prefix and seamlessly **continues** from it, faithfully preserving every vocal detail."
    "Note: This mode will disable Control Instruction."
)

_EXAMPLES_FOOTER_EN = (
    "---\n"
    "**💡 Voice Description Examples:**  \n"
    "Try the following Control Instructions to explore different voices:  \n\n"
    "**Example 1 — Gentle & Melancholic Girl**  \n"
    '`Control Instruction`: *"A young girl with a soft, sweet voice. '
    'Speaks slowly with a melancholic, slightly tsundere tone."*  \n'
    "`Target Text`: *\"I never asked you to stay… It's not like I care or anything. "
    "But… why does it still hurt so much now that you're gone?\"*  \n\n"
    "**Example 2 — Laid-Back Surfer Dude**  \n"
    '`Control Instruction`: *"Relaxed young male voice, slightly nasal, '
    'lazy drawl, very casual and chill."*  \n'
    '`Target Text`: *"Dude, did you see that set? The waves out there are totally gnarly today. '
    "Just catching barrels all morning — it's like, totally righteous, you know what I mean?\"*"
)

_USAGE_INSTRUCTIONS_ZH = (
    "**VoxCPM2 — 三种语音生成方式：**\n\n"
    "🎨 **声音设计（Voice Design）**  \n"
    "无需参考音频。在 **Control Instruction** 中描述目标音色特征"
    "（性别、年龄、语气、情绪、语速等），VoxCPM2 即可为你从零创造独一无二的声音。\n\n"
    "🎛️ **可控克隆（Controllable Cloning）**  \n"
    "上传参考音频，同时可选地使用 **Control Instruction** 来指定情绪、语速、风格等表达方式，"
    "在保留原始音色的基础上灵活控制说话风格。\n\n"
    "🎙️ **极致克隆（Ultimate Cloning）**  \n"
    "开启 **极致克隆模式** 并提供参考音频的文字内容（可自动识别）。"
    "模型会将参考音频视为已说出的前文，以**音频续写**的方式完整还原参考音频中的所有声音细节。"
    "注意：该模式与可控克隆模式互斥，将禁用Control Instruction。\n\n"
)

_EXAMPLES_FOOTER_ZH = (
    "---\n"
    "**💡 声音描述示例（中英文均可）：**  \n\n"
    "**示例 1 — 深宫太后**  \n"
    '`Control Instruction`: *"中老年女性，声音低沉阴冷，语速缓慢而有力，'
    '字字深思熟虑，带有深不可测的城府与威慑感。"*  \n'
    '`Target Text`: *"哀家在这深宫待了四十年，什么风浪没见过？你以为瞒得过哀家？"*  \n\n'
    "**示例 2 — 暴躁驾校教练**  \n"
    '`Control Instruction`: *"暴躁的中年男声，语速快，充满无奈和愤怒"*  \n'
    '`Target Text`: *"踩离合！踩刹车啊！你往哪儿开呢？前面是树你看不见吗？'
    '我教了你八百遍了，打死方向盘！你是不是想把车给我开到沟里去？"*  \n\n'
    "---\n"
    "**🗣️ 方言生成指南：**  \n"
    "要生成地道的方言语音，请在 **Target Text** 中直接使用方言词汇和句式，"
    "并在 **Control Instruction** 中描述方言特征。  \n\n"
    "**示例 — 广东话**  \n"
    '`Control Instruction`: *"粤语，中年男性，语气平淡"*  \n'
    '✅ 正确（粤语表达）：*"伙計，唔該一個A餐，凍奶茶少甜！"*  \n'
    '❌ 错误（普通话原文）：*"伙计，麻烦来一个A餐，冻奶茶少甜！"*  \n\n'
    "**示例 — 河南话**  \n"
    '`Control Instruction`: *"河南话，接地气的大叔"*  \n'
    '✅ 正确（河南话表达）：*"恁这是弄啥嘞？晌午吃啥饭？"*  \n'
    '❌ 错误（普通话原文）：*"你这是在干什么呢？中午吃什么饭？"*  \n\n'
    "🤖 **小技巧：** 不知道方言怎么写？可以用豆包、DeepSeek、Kimi 等 AI 助手"
    "将普通话翻译为方言文本，再粘贴到 Target Text 中即可。  \n\n"
)

_I18N_TRANSLATIONS = {
    "en": {
        "reference_audio_label": "🎤 Reference Audio (optional — upload for cloning)",
        "show_prompt_text_label": "🎙️ Ultimate Cloning Mode (transcript-guided cloning)",
        "show_prompt_text_info": "Auto-transcribes reference audio for every vocal nuance reproduced. Control Instruction will be disabled when active.",
        "prompt_text_label": "Transcript of Reference Audio (auto-filled via ASR, editable)",
        "prompt_text_placeholder": "The transcript of your reference audio will appear here …",
        "control_label": "🎛️ Control Instruction (optional — supports Chinese & English)",
        "control_placeholder": "e.g. A warm young woman / 年轻女性，温柔甜美 / Excited and fast-paced",
        "target_text_label": "✍️ Target Text — the content to speak",
        "generate_btn": "🔊 Generate Speech",
        "generated_audio_label": "Generated Audio",
        "advanced_settings_title": "⚙️ Advanced Settings",
        "ref_denoise_label": "Reference audio enhancement",
        "ref_denoise_info": "Apply ZipEnhancer denoising to the reference audio before cloning",
        "normalize_label": "Text normalization",
        "normalize_info": "Normalize numbers, dates, and abbreviations via wetext",
        "cfg_label": "CFG (guidance scale)",
        "cfg_info": "Higher → closer to the prompt / reference; lower → more creative variation",
        "dit_steps_label": "LocDiT flow-matching steps",
        "dit_steps_info": "LocDiT flow-matching steps — more steps → maybe better audio quality, but slower",
        "seed_label": "Seed",
        "seed_info": "Seed used for reproducible generation. Updated with the actual successful seed after generation.",
        "random_seed_label": "Random Seed",
        "random_seed_info": "Generate a new seed before each inference run.",
        "tab_tts": "🗣️ Text to Speech",
        "tab_srt": "📝 SRT → Speech",
        "usage_instructions": _USAGE_INSTRUCTIONS_EN,
        "examples_footer": _EXAMPLES_FOOTER_EN,
    },
    "zh-CN": {
        "reference_audio_label": "🎤 参考音频（可选 — 上传后用于克隆）",
        "show_prompt_text_label": "🎙️ 极致克隆模式（基于文本引导的极致克隆）",
        "show_prompt_text_info": "自动识别参考音频文本，完整还原音色、节奏、情感等全部声音细节。开启后 Control Instruction 将暂时禁用",
        "prompt_text_label": "参考音频内容文本（ASR 自动填充，可手动编辑）",
        "prompt_text_placeholder": "参考音频的文字内容将自动识别并显示在此处 …",
        "control_label": "🎛️ Control Instruction（可选 — 支持中英文描述）",
        "control_placeholder": "如：年轻女性，温柔甜美 / A warm young woman / 暴躁老哥，语速飞快",
        "target_text_label": "✍️ Target Text — 要合成的目标文本",
        "generate_btn": "🔊 开始生成",
        "generated_audio_label": "生成结果",
        "advanced_settings_title": "⚙️ 高级设置",
        "ref_denoise_label": "参考音频降噪增强",
        "ref_denoise_info": "克隆前使用 ZipEnhancer 对参考音频进行降噪处理",
        "normalize_label": "文本规范化",
        "normalize_info": "自动规范化数字、日期及缩写（基于 wetext）",
        "cfg_label": "CFG（引导强度）",
        "cfg_info": "数值越高 → 越贴合提示/参考音色；数值越低 → 生成风格更自由",
        "dit_steps_label": "LocDiT 流匹配迭代步数",
        "dit_steps_info": "LocDiT 流匹配生成迭代步数 — 步数越多 → 可能生成更好的音频质量，但速度变慢",
        "tab_tts": "🗣️ 语音合成",
        "tab_srt": "📝 字幕配音",
        "usage_instructions": _USAGE_INSTRUCTIONS_ZH,
        "examples_footer": _EXAMPLES_FOOTER_ZH,
    },
    "zh-Hans": None,  # alias, filled below
    "zh": None,  # alias, filled below
}
_I18N_TRANSLATIONS["zh-Hans"] = _I18N_TRANSLATIONS["zh-CN"]
_I18N_TRANSLATIONS["zh"] = _I18N_TRANSLATIONS["zh-CN"]

for _d in _I18N_TRANSLATIONS.values():
    if _d is not None:
        for _k, _v in _I18N_TRANSLATIONS["en"].items():
            _d.setdefault(_k, _v)

I18N = gr.I18n(**_I18N_TRANSLATIONS)

DEFAULT_TARGET_TEXT = (
    "VoxCPM2 is a creative multilingual TTS model from ModelBest, " "designed to generate highly realistic speech."
)

_CUSTOM_CSS = """
.logo-container {
    text-align: center;
    margin: 0.5rem 0 1rem 0;
}
.logo-container img {
    height: 80px;
    width: auto;
    max-width: 200px;
    display: inline-block;
}

/* Toggle switch style */
.switch-toggle {
    padding: 8px 12px;
    border-radius: 8px;
    background: var(--block-background-fill);
}
.switch-toggle input[type="checkbox"] {
    appearance: none;
    -webkit-appearance: none;
    width: 44px;
    height: 24px;
    background: #ccc;
    border-radius: 12px;
    position: relative;
    cursor: pointer;
    transition: background 0.3s ease;
    flex-shrink: 0;
}
.switch-toggle input[type="checkbox"]::after {
    content: "";
    position: absolute;
    top: 2px;
    left: 2px;
    width: 20px;
    height: 20px;
    background: white;
    border-radius: 50%;
    transition: transform 0.3s ease;
    box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}
.switch-toggle input[type="checkbox"]:checked {
    background: var(--color-accent);
}
.switch-toggle input[type="checkbox"]:checked::after {
    transform: translateX(20px);
}

"""

_APP_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="gray",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"],
)


# ---------- Model ----------


class VoxCPMDemo:
    def __init__(self, model_id: str = "openbmb/VoxCPM2", device: str = "auto") -> None:
        self.device = resolve_runtime_device(device, "cuda")
        logger.info(f"Running VoxCPM on device: {self.device}")
        self.optimize = self.device.startswith("cuda")

        self.asr_model_id = "iic/SenseVoiceSmall"
        self.asr_device = "cuda:0" if self.device.startswith("cuda") else "cpu"
        self.asr_model: Optional[AutoModel] = None

        self.voxcpm_model: Optional[voxcpm.VoxCPM] = None
        self._model_id = model_id

    def get_or_load_voxcpm(self) -> voxcpm.VoxCPM:
        if self.voxcpm_model is not None:
            return self.voxcpm_model
        logger.info(f"Loading model: {self._model_id}")
        self.voxcpm_model = voxcpm.VoxCPM.from_pretrained(
            self._model_id,
            optimize=self.optimize,
            device=self.device,
        )
        logger.info("Model loaded successfully.")
        return self.voxcpm_model

    def get_or_load_asr_model(self) -> AutoModel:
        if self.asr_model is not None:
            return self.asr_model
        logger.info(f"Loading ASR model: {self.asr_model_id} on device: {self.asr_device}")
        self.asr_model = AutoModel(
            model=self.asr_model_id,
            disable_update=True,
            log_level="DEBUG",
            device=self.asr_device,
        )
        logger.info("ASR model loaded successfully.")
        return self.asr_model

    def prompt_wav_recognition(self, prompt_wav: Optional[str]) -> str:
        if prompt_wav is None:
            return ""
        res = self.get_or_load_asr_model().generate(
            input=prompt_wav,
            language="auto",
            use_itn=True,
        )
        return res[0]["text"].split("|>")[-1]

    def _build_generate_kwargs(
        self,
        *,
        final_text: str,
        audio_path: Optional[str],
        prompt_text_clean: Optional[str],
        cfg_value_input: float,
        do_normalize: bool,
        denoise: bool,
        inference_timesteps: int = 10,
        seed: Optional[int] = None,
    ) -> dict:
        generate_kwargs = dict(
            text=final_text,
            reference_wav_path=audio_path,
            cfg_value=float(cfg_value_input),
            inference_timesteps=inference_timesteps,
            normalize=do_normalize,
            denoise=denoise,
            seed=seed,
        )
        if prompt_text_clean and audio_path:
            generate_kwargs["prompt_wav_path"] = audio_path
            generate_kwargs["prompt_text"] = prompt_text_clean
        return generate_kwargs

    def generate_tts_audio(
        self,
        text_input: str,
        control_instruction: str = "",
        reference_wav_path_input: Optional[str] = None,
        prompt_text: str = "",
        cfg_value_input: float = 2.0,
        do_normalize: bool = True,
        denoise: bool = True,
        inference_timesteps: int = 10,
        seed: Optional[int] = None,
    ) -> Tuple[int, np.ndarray, Optional[int]]:
        current_model = self.get_or_load_voxcpm()

        text = (text_input or "").strip()
        if len(text) == 0:
            raise ValueError("Please input text to synthesize.")

        control = (control_instruction or "").strip()
        # Strip any parentheses (half-width/full-width) from control text to avoid
        # breaking the "(control)text" prompt format expected by the model.
        control = re.sub(r"[()（）]", "", control).strip()
        final_text = f"({control}){text}" if control else text

        audio_path = reference_wav_path_input if reference_wav_path_input else None
        prompt_text_clean = (prompt_text or "").strip() or None

        if audio_path and prompt_text_clean:
            logger.info(f"[Voice Cloning] prompt_wav + prompt_text + reference_wav")
        elif audio_path:
            logger.info(f"[Voice Control] reference_wav only")
        else:
            logger.info(f"[Voice Design] control: {control[:50] if control else 'None'}...")

        logger.info(f"Generating audio for text: '{final_text[:80]}...'")
        generate_kwargs = self._build_generate_kwargs(
            final_text=final_text,
            audio_path=audio_path,
            prompt_text_clean=prompt_text_clean,
            cfg_value_input=cfg_value_input,
            do_normalize=do_normalize,
            denoise=denoise,
            inference_timesteps=inference_timesteps,
            seed=seed,
        )
        wav = current_model.generate(**generate_kwargs)
        last_successful_seed = getattr(current_model.tts_model, "last_successful_seed", seed)
        return (current_model.tts_model.sample_rate, wav, last_successful_seed)


# ---------- Shared helpers ----------


def _notes(warnings: list, done: bool = False) -> str:
    """Warnings kept on the page (pop-ups are easy to miss)."""
    if not warnings:
        return "✅ Done — every line is consistent with its speaker." if done else ""
    return "**⚠️ Warnings**\n\n" + "\n".join(f"- {w}" for w in warnings)


def _table_rows(value) -> list:
    if value is None:
        return []
    if hasattr(value, "values") and hasattr(value, "columns"):  # pandas.DataFrame
        value = value.values.tolist()
    return [row for row in value if any(str(c).strip() for c in row)]


# ---------- SRT → Speech tab ----------

_SRT_INSTRUCTIONS = (
    "**📝 Voice a whole SRT script — no video needed**\n\n"
    "1. Upload an **SRT** whose lines are tagged `[Name|male|kid|sad] text`. After the name, gender (`male` / "
    "`female`, or `boy` / `girl`), age (`adult` / `kid`) and emotion (`happy`, `sad`, `angry`, `fearful`, `surprised`, "
    "… or any word) are optional and in any order; gender and age only need to be given once per speaker. "
    "Untagged lines are read by *Narrator*.\n"
    "2. Click **Analyze** — missing emotions are guessed from punctuation (and English keywords). Check and edit the "
    "tables; download the fully tagged SRT to keep your edits.\n"
    "3. Click **Generate** — each speaker's voice is made once as a **calm, neutral reading of their first line** "
    "(retried until its pitch fits their gender and age, e.g. a kid's). **Every line of that speaker — the first one "
    "too — clones that voice** and adds its own emotion, retried until it is consistent. You get one combined track "
    "on the SRT timing plus every line as its own file.\n"
    "4. A line still marked ⚠? It is pre-selected under **🔁 Lines to regenerate** — click **Regenerate selected "
    "lines** to redo just those lines with new seeds (same speaker voice); the combined track is rebuilt.\n"
    "5. A speaker's voice doesn't fit them? Choose them under **🎭 New voice for a speaker** (edit their gender, age or "
    "description in the Speakers table first if you like) — their voice is designed again and all of their lines are "
    "regenerated."
)
_SRT_SPEAKER_HEADERS = ["Speaker", "Gender", "Age", "Voice description", "Lines"]
_SRT_LINE_HEADERS = ["#", "Start", "End", "Speaker", "Emotion", "Text"]
_SRT_REPORT_HEADERS = [
    "#", "Speaker", "Gender", "Age", "Emotion", "Mode", "Voice match", "Pitch (Hz)", "Status", "Tries", "Slot (s)",
    "Generated (s)", "Text",
]


def _line_status(r: dict) -> str:
    issues = list(r.get("issues") or [])
    if not issues and not r.get("voice_ok", True):
        issues.append("not consistent")
    if not r.get("gender_ok", True) and "unclear gender" not in issues:
        issues.append("unclear gender")
    issues = [f"unclear gender ({r.get('pitch_hz')} Hz)" if i == "unclear gender" else i for i in issues]
    status = "⚠ " + ", ".join(issues) if issues else "✅ OK"
    if r.get("trimmed"):
        status += " · repeats trimmed"
    return status


def build_srt_tab(demo: VoxCPMDemo):
    import shutil

    from voxcpm import dubbing, script_voice

    speaker_models = {}

    def _analyze(srt_path, guess_emotion):
        if not srt_path:
            raise gr.Error("Please upload an SRT file first.")
        try:
            lines, profiles = script_voice.load_script(srt_path, guess_missing_emotion=bool(guess_emotion))
        except Exception as e:
            raise gr.Error(f"Could not read the SRT: {e}")
        if not lines:
            raise gr.Error("No subtitle lines found.")
        problems = script_voice.find_srt_problems(srt_path)
        for message in problems:
            gr.Warning(message, duration=None)
        workdir = tempfile.mkdtemp(prefix="voxcpm_srt_")
        tagged = os.path.join(workdir, Path(srt_path).stem + "_tagged.srt")
        Path(tagged).write_text(script_voice.format_tagged_srt(lines, profiles), encoding="utf-8")
        line_rows = [
            [
                l.index, round(l.start, 3), round(l.end, 3), l.speaker,
                l.emotion + (" (guessed)" if l.features.get("emotion_guessed") else ""), l.text,
            ]
            for l in lines
        ]
        return script_voice.speaker_summary(lines, profiles), line_rows, tagged, workdir, _notes(problems)

    def _read_tables(speaker_rows, line_rows):
        profiles = {}
        for row in _table_rows(speaker_rows):
            name, gender, age, description = (list(row) + [""] * 4)[:4]
            name = str(name).strip()
            if not name:
                continue
            gender = str(gender).strip().lower() or "unknown"
            age = str(age).strip().lower() or "adult"
            if gender not in ("male", "female", "unknown"):
                raise ValueError(f"Gender of {name!r} must be male, female or unknown")
            if age not in script_voice.AGES:
                raise ValueError(f"Age of {name!r} must be adult or kid")
            profiles[name] = dubbing.SpeakerProfile(
                name=name, gender=gender, age=age, voice_mode="clone_first",
                description=str(description).strip() or script_voice.voice_description(gender, age),
            )
        lines = []
        for row in _table_rows(line_rows):
            idx, start, end, speaker, emotion, text = (list(row) + [""] * 6)[:6]
            if not str(text).strip():
                continue
            speaker = str(speaker).strip() or script_voice.DEFAULT_SPEAKER
            lines.append(
                dubbing.SubtitleLine(
                    index=int(float(idx)) if str(idx).strip() else len(lines) + 1,
                    start=dubbing.parse_timestamp(start),
                    end=dubbing.parse_timestamp(end),
                    text=str(text).strip(),
                    speaker=speaker,
                    gender=profiles[speaker].gender if speaker in profiles else "unknown",
                    emotion=str(emotion or "").replace("(guessed)", "").strip() or "neutral",
                )
            )
        if not lines:
            raise ValueError("The lines table is empty.")
        return lines, profiles

    def _run(
        workdir, speaker_rows, line_rows, verify_voice, max_tries, max_speedup, pad_to_slot, remove_silence,
        max_pause, cfg_value, dit_steps,
        progress, only=None, new_voices=(),
    ):
        if not workdir:
            raise gr.Error("Please upload an SRT and click Analyze first.")
        try:
            lines, profiles = _read_tables(speaker_rows, line_rows)
            progress(0, desc="Loading VoxCPM...")
            voicer = script_voice.ScriptVoicer(demo.get_or_load_voxcpm(), workdir)
            if verify_voice:
                progress(0, desc="Loading voice identification model (CAM++)...")
                if "model" not in speaker_models:
                    speaker_models["model"] = dubbing.load_speaker_model()
                voicer.speaker_model = speaker_models["model"]
            options = dubbing.DubOptions(
                verify_voice=bool(verify_voice),
                max_attempts=int(max_tries),
                first_line_attempts=max(int(max_tries), 12),
                max_speedup=float(max_speedup),
                pad_to_slot=bool(pad_to_slot),
                remove_silence=bool(remove_silence),
                max_pause=float(max_pause),
                # Each line must stay close to its speaker's voice in pitch and timbre.
                pitch_tolerance=2.0,
                min_consistency=0.6,
                cfg_value=float(cfg_value),
                inference_timesteps=int(dit_steps),
            )

            def _on_progress(i, n, line):
                verb = "Re-voicing" if new_voices else "Regenerating" if only else "Line"
                progress(i / max(n, 1), desc=f"{verb} {min(i + 1, n)}/{n} — #{line.index} {line.speaker}")

            out = os.path.join(workdir, "script.wav")
            result = voicer.voice(
                lines, profiles, out, options, progress=_on_progress, only=only, new_voices=new_voices
            )
            for message in result.warnings:
                gr.Warning(message, duration=None)
            zip_path = shutil.make_archive(os.path.join(workdir, "script_lines"), "zip", result.lines_dir)
            line_files = sorted(str(p) for p in Path(result.lines_dir).glob("*.wav"))
        except gr.Error:
            raise
        except Exception as e:
            logger.exception("SRT voicing failed")
            raise gr.Error(str(e))

        report_rows = [
            [
                r["index"], r["speaker"], r["gender"], r["age"],
                r["emotion"] + (" (guessed)" if r.get("emotion_guessed") else ""),
                r["mode"] + (f" · regenerated ×{r['regenerated']}" if r.get("regenerated") else ""),
                ("" if r.get("voice_match") is None else f"{r['voice_match']:.2f}")
                + ("" if r.get("consistency") is None else f" (vs own lines {r['consistency']:.2f})")
                + ("" if r.get("voice_ok", True) else " ⚠"),
                f"{r.get('pitch_hz') or ''}" + ("" if r.get("gender_ok", True) else " ⚠ unclear gender"),
                _line_status(r), r["attempts"], round(r["end"] - r["start"], 2), r["generated_s"], r["text"],
            ]
            for r in result.report
        ]
        failed = script_voice.failed_lines(result.report)
        retry_choices = [
            (f"#{r['index']} {r['speaker']}" + (" ⚠" if r["index"] in failed else "") + f" — {r['text'][:30]}", str(r["index"]))
            for r in result.report
        ]
        if new_voices:
            redone = {r["index"] for r in result.report if r["speaker"] in new_voices}
        else:
            redone = set(only or ())
        picked = next((f for f in line_files if int(Path(f).name[:4]) in redone), line_files[0] if line_files else None)
        notes = _notes(result.warnings, done=True)
        if new_voices:
            notes = f"🎭 New voice for {', '.join(sorted(new_voices))} — all of their lines were regenerated.\n\n" + notes
        elif only:
            notes = f"🔁 Regenerated line(s) {', '.join(f'#{i}' for i in sorted(only))}.\n\n" + notes
        speakers = list(dict.fromkeys(r["speaker"] for r in result.report))
        voice_speaker = next(iter(new_voices)) if new_voices else speakers[0]
        return (
            result.audio_path,
            gr.update(value=zip_path, interactive=True, label=f"⬇️ Download all {len(line_files)} lines (.zip)"),
            line_files,
            gr.update(choices=[(Path(f).stem, f) for f in line_files], value=picked),
            picked,
            result.tagged_srt_path,
            report_rows,
            notes,
            gr.update(choices=retry_choices, value=[str(i) for i in failed]),
            gr.update(choices=speakers, value=voice_speaker),
            result.voices,
            result.voices.get(voice_speaker),
        )

    def _generate(
        workdir, speaker_rows, line_rows, verify_voice, max_tries, max_speedup, pad_to_slot, remove_silence,
        max_pause, cfg_value, dit_steps,
        progress=gr.Progress(),
    ):
        return _run(
            workdir, speaker_rows, line_rows, verify_voice, max_tries, max_speedup, pad_to_slot, remove_silence,
        max_pause, cfg_value, dit_steps,
            progress,
        )

    def _regenerate(
        selected, workdir, speaker_rows, line_rows, verify_voice, max_tries, max_speedup, pad_to_slot,
        remove_silence, max_pause, cfg_value, dit_steps, progress=gr.Progress(),
    ):
        if not selected:
            raise gr.Error("Choose the line(s) to regenerate first (failed lines ⚠ are selected after Generate).")
        return _run(
            workdir, speaker_rows, line_rows, verify_voice, max_tries, max_speedup, pad_to_slot, remove_silence,
        max_pause, cfg_value, dit_steps,
            progress, only={int(v) for v in selected},
        )

    def _revoice(
        speaker, workdir, speaker_rows, line_rows, verify_voice, max_tries, max_speedup, pad_to_slot,
        remove_silence, max_pause, cfg_value, dit_steps, progress=gr.Progress(),
    ):
        if not speaker:
            raise gr.Error("Choose the speaker who needs a new voice first (speakers are listed after Generate).")
        return _run(
            workdir, speaker_rows, line_rows, verify_voice, max_tries, max_speedup, pad_to_slot, remove_silence,
        max_pause, cfg_value, dit_steps,
            progress, only=(), new_voices={speaker},
        )

    gr.Markdown(_SRT_INSTRUCTIONS)
    workdir_state = gr.State(None)
    voices_state = gr.State({})
    with gr.Row():
        with gr.Column():
            srt_input = gr.File(label="📝 Tagged subtitles (.srt)", file_types=[".srt"], type="filepath")
            guess_emotion = gr.Checkbox(
                value=True,
                label="Guess missing emotions",
                info="From punctuation (?!, !, …) and English keywords; marked (guessed) so you can check them",
                elem_classes=["switch-toggle"],
            )
            analyze_btn = gr.Button("1️⃣ Analyze script", variant="secondary")
            tagged_output = gr.File(label="Tagged SRT (every line with speaker, gender, age and emotion)")
        with gr.Column():
            audio_output = gr.Audio(label="Combined track (on the SRT timing)", type="filepath")

    with gr.Row():
        with gr.Column():
            zip_output = gr.DownloadButton("⬇️ Download all lines (.zip)", variant="primary", interactive=False)
            line_files_output = gr.File(
                label="⬇️ Every line as its own file — click a file to download just that one", file_count="multiple"
            )
        with gr.Column():
            line_picker = gr.Dropdown(choices=[], label="🎧 Listen to a line", interactive=True)
            line_player = gr.Audio(label="Selected line", type="filepath")

    speakers_table = gr.Dataframe(
        headers=_SRT_SPEAKER_HEADERS,
        datatype=["str", "str", "str", "str", "number"],
        label="🎭 Speakers — gender: male / female / unknown, age: adult / kid; the description is the voice prompt",
        interactive=True,
        wrap=True,
    )
    lines_table = gr.Dataframe(
        headers=_SRT_LINE_HEADERS,
        datatype=["number", "number", "number", "str", "str", "str"],
        label="💬 Lines — edit speaker or emotion (happy, sad, angry, fearful, surprised, … or any word)",
        interactive=True,
        wrap=True,
        max_height=420,
    )
    with gr.Accordion("⚙️ Settings", open=False):
        with gr.Row():
            verify_voice = gr.Checkbox(
                value=True,
                label="Check every line matches its speaker's first line",
                info="CAM++ voice check; regenerates until consistent",
            )
            max_tries = gr.Slider(1, 20, value=8, step=1, label="Max tries per line", info="Each first line gets at least 12")
            max_speedup = gr.Slider(
                1.0, 1.5, value=1.1, step=0.05, label="Max speed-up to fit a line (combined track)",
                info="Longer lines push the next lines later instead of being squeezed (squeezing changes the voice)",
            )
            pad_to_slot = gr.Checkbox(
                value=False,
                label="Pad short lines with silence",
                info="Off: every line file keeps its natural length. On: silence after the speech up to the slot length",
            )
        with gr.Row():
            remove_silence = gr.Checkbox(
                value=True,
                label="Remove silences",
                info="Cut the silence before and after each line and shorten long pauses inside it",
            )
            max_pause = gr.Slider(
                0.05, 0.6, value=0.2, step=0.05, label="Longest pause kept inside a line (s)",
                info="Pauses longer than this are shortened to it",
            )
        with gr.Row():
            cfg_value = gr.Slider(1.0, 3.0, value=2.0, step=0.1, label=I18N("cfg_label"))
            dit_steps = gr.Slider(1, 50, value=10, step=1, label=I18N("dit_steps_label"))
    generate_btn = gr.Button("2️⃣ Generate all lines", variant="primary", size="lg")
    with gr.Row():
        retry_picker = gr.Dropdown(
            choices=[],
            multiselect=True,
            label="🔁 Lines to regenerate — failed lines (⚠) are selected after Generate; edit a line's text or "
            "emotion in the table first if you like",
            scale=4,
        )
        retry_btn = gr.Button("🔁 Regenerate selected lines", variant="secondary", scale=1)
    with gr.Row():
        revoice_picker = gr.Dropdown(
            choices=[],
            label="🎭 New voice for a speaker — their voice is designed again (from their gender, age and description "
            "in the Speakers table) and all of their lines are regenerated",
            scale=2,
        )
        voice_preview = gr.Audio(label="This speaker's voice (what all their lines copy)", type="filepath", scale=2)
        revoice_btn = gr.Button("🎭 New voice — redo all their lines", variant="secondary", scale=1)
    notes_box = gr.Markdown()
    report_table = gr.Dataframe(headers=_SRT_REPORT_HEADERS, label="📊 Report", interactive=False, wrap=True)

    analyze_btn.click(
        fn=_analyze,
        inputs=[srt_input, guess_emotion],
        outputs=[speakers_table, lines_table, tagged_output, workdir_state, notes_box],
    )
    settings = [
        workdir_state, speakers_table, lines_table, verify_voice, max_tries, max_speedup, pad_to_slot,
        remove_silence, max_pause, cfg_value, dit_steps,
    ]
    results = [
        audio_output, zip_output, line_files_output, line_picker, line_player, tagged_output, report_table, notes_box,
        retry_picker, revoice_picker, voices_state, voice_preview,
    ]
    generate_btn.click(fn=_generate, inputs=settings, outputs=results, api_name="voice_srt")
    retry_btn.click(fn=_regenerate, inputs=[retry_picker] + settings, outputs=results, api_name="regenerate_srt_lines")
    revoice_btn.click(fn=_revoice, inputs=[revoice_picker] + settings, outputs=results, api_name="revoice_srt_speaker")
    line_picker.change(fn=lambda path: path, inputs=line_picker, outputs=line_player)
    revoice_picker.change(
        fn=lambda name, voices: (voices or {}).get(name), inputs=[revoice_picker, voices_state], outputs=voice_preview
    )


# ---------- UI ----------


def create_demo_interface(demo: VoxCPMDemo):
    gr.set_static_paths(paths=[Path.cwd().absolute() / "assets"])

    def _coerce_seed(seed_value) -> Optional[int]:
        if seed_value is None or seed_value == "":
            return None
        return int(seed_value)

    def _prepare_seed(use_random_seed: bool, seed_value):
        if use_random_seed:
            return random.randint(0, 2**32 - 1)
        return _coerce_seed(seed_value)

    def _on_random_seed_toggle(checked):
        return gr.update(interactive=not checked)

    def _generate(
        text: str,
        control_instruction: str,
        ref_wav: Optional[str],
        use_prompt_text: bool,
        prompt_text_value: str,
        cfg_value: float,
        do_normalize: bool,
        denoise: bool,
        dit_steps: int,
        seed_value,
    ):
        actual_prompt_text = prompt_text_value.strip() if use_prompt_text else ""
        actual_control = "" if use_prompt_text else control_instruction
        seed = _coerce_seed(seed_value)
        sr, wav_np, last_successful_seed = demo.generate_tts_audio(
            text_input=text,
            control_instruction=actual_control,
            reference_wav_path_input=ref_wav,
            prompt_text=actual_prompt_text,
            cfg_value_input=cfg_value,
            do_normalize=do_normalize,
            denoise=denoise,
            inference_timesteps=int(dit_steps),
            seed=seed,
        )
        return (sr, wav_np), last_successful_seed

    def _on_toggle_instant(checked):
        """Instant UI toggle — no ASR, no blocking."""
        if checked:
            return (
                gr.update(visible=True, value="", placeholder="Recognizing reference audio..."),
                gr.update(visible=False),
            )
        return (
            gr.update(visible=False),
            gr.update(visible=True, interactive=True),
        )

    def _run_asr_if_needed(checked, audio_path):
        """Run ASR after the UI has updated. Only when toggled ON."""
        if not checked or not audio_path:
            return gr.update()
        try:
            logger.info("Running ASR on reference audio...")
            asr_text = demo.prompt_wav_recognition(audio_path)
            logger.info(f"ASR result: {asr_text[:60]}...")
            return gr.update(value=asr_text)
        except Exception as e:
            logger.warning(f"ASR recognition failed: {e}")
            return gr.update(value="")

    with gr.Blocks() as interface:
        gr.HTML(
            '<div class="logo-container">'
            '<img src="/gradio_api/file=assets/voxcpm_logo.png" alt="VoxCPM Logo">'
            "</div>"
        )

        with gr.Tabs():
            with gr.Tab(I18N("tab_tts")):
                gr.Markdown(I18N("usage_instructions"))

                with gr.Row():
                    with gr.Column():
                        reference_wav = gr.Audio(
                            sources=["upload", "microphone"],
                            type="filepath",
                            label=I18N("reference_audio_label"),
                        )
                        show_prompt_text = gr.Checkbox(
                            value=False,
                            label=I18N("show_prompt_text_label"),
                            info=I18N("show_prompt_text_info"),
                            elem_classes=["switch-toggle"],
                        )
                        prompt_text = gr.Textbox(
                            value="",
                            label=I18N("prompt_text_label"),
                            placeholder=I18N("prompt_text_placeholder"),
                            lines=2,
                            visible=False,
                        )
                        control_instruction = gr.Textbox(
                            value="",
                            label=I18N("control_label"),
                            placeholder=I18N("control_placeholder"),
                            lines=2,
                        )
                        text = gr.Textbox(
                            value=DEFAULT_TARGET_TEXT,
                            label=I18N("target_text_label"),
                            lines=3,
                        )

                        with gr.Accordion(I18N("advanced_settings_title"), open=False):
                            DoDenoisePromptAudio = gr.Checkbox(
                                value=False,
                                label=I18N("ref_denoise_label"),
                                elem_classes=["switch-toggle"],
                                info=I18N("ref_denoise_info"),
                            )
                            DoNormalizeText = gr.Checkbox(
                                value=False,
                                label=I18N("normalize_label"),
                                elem_classes=["switch-toggle"],
                                info=I18N("normalize_info"),
                            )
                            cfg_value = gr.Slider(
                                minimum=1.0,
                                maximum=3.0,
                                value=2.0,
                                step=0.1,
                                label=I18N("cfg_label"),
                                info=I18N("cfg_info"),
                            )
                            dit_steps = gr.Slider(
                                minimum=1,
                                maximum=50,
                                value=10,
                                step=1,
                                label=I18N("dit_steps_label"),
                                info=I18N("dit_steps_info"),
                            )
                            with gr.Row():
                                seed_value = gr.Number(
                                    value=random.randint(0, 2**32 - 1),
                                    precision=0,
                                    label=I18N("seed_label"),
                                    info=I18N("seed_info"),
                                    interactive=False,
                                )
                                random_seed = gr.Checkbox(
                                    value=True,
                                    label=I18N("random_seed_label"),
                                    elem_classes=["switch-toggle"],
                                    info=I18N("random_seed_info"),
                                )

                        run_btn = gr.Button(I18N("generate_btn"), variant="primary", size="lg")

                    with gr.Column():
                        audio_output = gr.Audio(label=I18N("generated_audio_label"))
                        gr.Markdown(I18N("examples_footer"))

                show_prompt_text.change(
                    fn=_on_toggle_instant,
                    inputs=[show_prompt_text],
                    outputs=[prompt_text, control_instruction],
                ).then(
                    fn=_run_asr_if_needed,
                    inputs=[show_prompt_text, reference_wav],
                    outputs=[prompt_text],
                )

                random_seed.change(
                    fn=_on_random_seed_toggle,
                    inputs=[random_seed],
                    outputs=[seed_value],
                )

                run_btn.click(
                    fn=_prepare_seed,
                    inputs=[random_seed, seed_value],
                    outputs=[seed_value],
                    show_progress=False,
                ).then(
                    fn=_generate,
                    inputs=[
                        text,
                        control_instruction,
                        reference_wav,
                        show_prompt_text,
                        prompt_text,
                        cfg_value,
                        DoNormalizeText,
                        DoDenoisePromptAudio,
                        dit_steps,
                        seed_value,
                    ],
                    outputs=[audio_output, seed_value],
                    show_progress=True,
                    api_name="generate",
                )

            with gr.Tab(I18N("tab_srt")):
                build_srt_tab(demo)

    return interface


def run_demo(
    server_name: str = "0.0.0.0",
    server_port: int = 8808,
    show_error: bool = True,
    model_id: str = "openbmb/VoxCPM2",
    device: str = "auto",
):
    demo = VoxCPMDemo(model_id=model_id, device=device)
    interface = create_demo_interface(demo)
    interface.queue(max_size=10, default_concurrency_limit=1).launch(
        server_name=server_name,
        server_port=server_port,
        show_error=show_error,
        i18n=I18N,
        theme=_APP_THEME,
        css=_CUSTOM_CSS,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        type=str,
        default="openbmb/VoxCPM2",
        help="Local path or HuggingFace repo ID (default: openbmb/VoxCPM2)",
    )
    parser.add_argument("--port", type=int, default=8808, help="Server port")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Bind address. Use 127.0.0.1 to restrict access to the local machine; "
             "the default 0.0.0.0 exposes the unauthenticated UI/API to the network (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Runtime device: auto, cpu, mps, cuda, or cuda:N (default: auto)",
    )
    args = parser.parse_args()
    run_demo(
        model_id=args.model_id,
        server_name=args.host,
        server_port=args.port,
        device=args.device,
    )
