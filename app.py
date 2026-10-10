import os
import re
import sys
import json
import time
import tempfile
import threading
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
        "tab_editor": "🎬 Video Editor",
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
        "tab_editor": "🎬 视频编辑",
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
    "   Optionally add the **original video** too: each line's tone (louder, softer, faster, …) is then read from the "
    "original speech at its SRT time (and, if you tick it, a rough emotion for lines with none in their tag).\n"
    "2. Click **Analyze** — missing emotions are guessed from punctuation (and English keywords), or read from the "
    "video if you ticked that (an emotion written in a tag always wins). Check and edit the tables; download the fully "
    "tagged SRT to keep your edits.\n"
    "3. Click **Generate** — each speaker's voice is made once as a **calm, neutral reading of their first line** "
    "(retried until its pitch fits their gender and age, e.g. a kid's). **Every line of that speaker — the first one "
    "too — clones that voice** and adds its own emotion, retried until it is consistent. You get one combined track "
    "on the SRT timing plus every line as its own file.\n"
    "4. A line still marked ⚠? It is pre-selected under **🔁 Lines to regenerate** — click **Regenerate selected "
    "lines** to redo just those lines with new seeds (same speaker voice); the combined track is rebuilt.\n"
    "5. A speaker's voice doesn't fit them? Choose them under **🎭 New voice for a speaker** (edit their gender, age or "
    "description in the Speakers table first if you like) — their voice is designed again and all of their lines are "
    "regenerated.\n\n"
    "💾 Your project is saved as you go: if you close or reload this tab, reopening the page brings back your tables and "
    "every line already generated, and a generation in progress keeps running in the background."
)
# Voice: a voice template from the library, or "auto" (designed from the speaker's first line). It is last, so
# projects saved before it still load.
_SRT_SPEAKER_HEADERS = ["Speaker", "Gender", "Age", "Voice description", "Lines", "Voice"]


def _speaker_rows_6(rows) -> list:
    return [(list(r) + [""] * 6)[:5] + [str((list(r) + [""] * 6)[5] or "auto")] for r in rows or []]
# Tone (louder, softer, faster, … read from the original video) is last, so projects saved before it still load.
_SRT_LINE_HEADERS = ["#", "Start", "End", "Speaker", "Emotion", "Text", "Tone"]
_EMOTION_MARKS = ("(guessed)", "(from video)")


def _plain_emotion(value) -> str:
    text = str(value or "")
    for mark in _EMOTION_MARKS:
        text = text.replace(mark, "")
    return text.strip()


def _line_rows_7(rows) -> list:
    """Lines table rows with the Tone column (rows saved before it have 6 columns)."""
    return [(list(r) + [""] * 7)[:7] for r in rows or []]
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


# SRT → Speech projects live here (not in a temp folder), so a closed or reloaded tab can pick them up again.
SRT_PROJECTS_DIR = Path(os.environ.get("VOXCPM_PROJECTS_DIR", Path.home() / ".voxcpm" / "srt_projects"))
# Background generation jobs by project folder: they keep running when the browser tab is closed.
_SRT_JOBS: dict = {}
_SRT_JOBS_LOCK = threading.Lock()


def _srt_current_project() -> Optional[str]:
    pointer = SRT_PROJECTS_DIR / "current.json"
    try:
        workdir = json.loads(pointer.read_text(encoding="utf-8"))["workdir"]
    except (OSError, ValueError, KeyError):
        return None
    return workdir if Path(workdir, "tables.json").exists() else None


def _srt_save_tables(
    workdir: str, speaker_rows, line_rows, tagged: Optional[str] = None, options: Optional[list] = None,
    bump: bool = False,
) -> None:
    """Save the project tables. ``bump`` marks a change made outside the SRT tab (the video editor), so the
    SRT tab reloads them when it is opened again."""
    path = Path(workdir, "tables.json")
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data["speakers"] = [list(r) for r in _table_rows(speaker_rows)]
    data["lines"] = [list(r) for r in _table_rows(line_rows)]
    if tagged:
        data["tagged"] = tagged
    if options is not None:
        data["options"] = list(options)
    if bump:
        data["rev"] = int(data.get("rev", 0)) + 1
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    SRT_PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    (SRT_PROJECTS_DIR / "current.json").write_text(json.dumps({"workdir": workdir}), encoding="utf-8")


def _srt_tables_rev(workdir: Optional[str]) -> int:
    try:
        return int(json.loads(Path(workdir, "tables.json").read_text(encoding="utf-8")).get("rev", 0))
    except (OSError, ValueError, TypeError):
        return -1


# Generation settings of the SRT tab, in order (the video editor reuses the last ones used there).
_SRT_DEFAULT_OPTIONS = [True, 8, 1.1, False, True, 0.2, 2.0, 10]


def _voice_library_dir() -> Path:
    from voxcpm import script_voice

    script_voice.VOICE_LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    return script_voice.VOICE_LIBRARY_DIR


def _template_choices() -> list:
    from voxcpm import script_voice

    library = script_voice.load_voice_library()
    return [("Auto — designed from their first line", script_voice.AUTO_VOICE)] + [
        (f"{name} ({v['gender']}, {v['age']})", name) for name, v in sorted(library.items(), key=lambda kv: (not kv[1].get("starter"), kv[0]))
    ]


def build_srt_tab(demo: VoxCPMDemo, blocks: gr.Blocks) -> dict:
    """The SRT → Speech tab. Returns what the video editor tab needs: ``start_job(workdir, speaker_rows,
    line_rows, options, only, new_voices)`` to regenerate lines, and the hand-over between the two tabs."""
    import shutil

    from voxcpm import dubbing, script_voice

    speaker_models = {}
    no_change = gr.update()

    def _analyze(
        srt_path, guess_emotion, video_path=None, isolate=False, video_emotion=False, auto_voices=True,
        progress=gr.Progress(),
    ):
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
        stem = re.sub(r"[^\w\-]+", "_", Path(srt_path).stem)[:40] or "script"
        workdir = str(SRT_PROJECTS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_{stem}")
        Path(workdir).mkdir(parents=True, exist_ok=True)
        tagged = os.path.join(workdir, Path(srt_path).stem + "_tagged.srt")
        Path(tagged).write_text(script_voice.format_tagged_srt(lines, profiles), encoding="utf-8")
        notes = list(problems)
        video_note = ""
        if video_path:
            try:
                used, music = _video_tones(workdir, lines, video_path, bool(isolate), bool(video_emotion), progress)
                video_note = (
                    f"🎬 Tone read from the video for **{used} of {len(lines)} lines**"
                    + (f" ({len(lines) - used} had no speech in the video at their time)" if used < len(lines) else "")
                    + ("; detected emotions are marked *(from video)*" if video_emotion else "")
                    + ". The video is also ready in the 🎬 Video Editor tab."
                    + (f"\n\n🎵 **{music} line(s) have music under the speech**, which the emotion model hears as anger, "
                       "so their emotion was not taken from the video (they keep the guess). Tick **Separate the voices "
                       "from the music first** and Analyze again to read those too." if music else "")
                    + "\n\n"
                )
                Path(tagged).write_text(script_voice.format_tagged_srt(lines, profiles), encoding="utf-8")
            except Exception as e:
                logger.exception("Reading the tone from the video failed")
                notes.append(f"The tone could not be read from the video: {e}")
        speaker_rows = script_voice.speaker_summary(lines, profiles)
        voice_note = ""
        if auto_voices:
            speaker_rows, voice_note = _pick_voices(workdir, speaker_rows, lines)
        line_rows = [
            [
                l.index, round(l.start, 3), round(l.end, 3), l.speaker,
                l.emotion + (" (guessed)" if l.features.get("emotion_guessed") else "")
                + (" (from video)" if l.features.get("emotion_from_video") else ""),
                l.text, l.tone,
            ]
            for l in lines
        ]
        _srt_save_tables(workdir, speaker_rows, line_rows, tagged)
        return speaker_rows, line_rows, tagged, workdir, video_note + voice_note + _notes(notes)

    def _pick_voices(workdir, speaker_rows, lines):
        """Give every speaker the voice template that fits them best (their gender, age and description, and how
        they sound in the original video if the project has one). Returns (speaker rows, a note)."""
        library = script_voice.load_voice_library()
        rows = _speaker_rows_6(_table_rows(speaker_rows))
        if not library:
            return rows, ("🎙 No voice templates yet — make the starter voices under **🎙 Manage voice templates** "
                          "to have one picked for each speaker.\n\n")
        profiles = {
            str(r[0]): dubbing.SpeakerProfile(name=str(r[0]), gender=str(r[1] or "unknown"), age=str(r[2] or "adult"),
                                              description=str(r[3] or ""))
            for r in rows if str(r[0]).strip()
        }
        counts = {str(r[0]): int(float(r[4] or 0)) for r in rows if str(r[0]).strip()}
        pitches = {}
        video_dir = Path(workdir, "video") if workdir else None
        source = None
        if video_dir and (video_dir / "voices16k.wav").exists():
            source = video_dir / "voices16k.wav"  # voices separated from the music: cleaner pitch
        elif video_dir and (video_dir / "audio16k.wav").exists():
            source = video_dir / "audio16k.wav"
        if source is not None:
            import soundfile as sf

            try:
                audio, sr = sf.read(str(source), dtype="float32")
                pitches = script_voice.speaker_pitches(lines, audio, sr)
            except Exception as e:
                logger.warning("Measuring the speakers' voices in the video failed: %s", e)
        picked = script_voice.match_templates(profiles, library, counts, pitches)
        for row in rows:
            if str(row[0]) in picked:
                row[5] = picked[str(row[0])]
        how = " (matched to how they sound in the video)" if pitches else ""
        return rows, ("🪄 Voices picked from your templates" + how + ": "
                      + ", ".join(f"**{n}** → {t}" for n, t in picked.items())
                      + ". Change any under **🎙 Give a speaker a voice template**.\n\n")

    def _video_tones(workdir, lines, video_path, isolate, read_emotion, progress):
        """Store the video in the project (the video editor uses it too) and read each line's emotion and tone
        from the original speech at its SRT time. Returns (lines with a reading, lines whose emotion was skipped
        because of music under the speech)."""
        import soundfile as sf

        from voxcpm import video_editor

        progress(0.1, desc="Reading the video" + (" and separating the voices from the music (a few minutes)…" if isolate else "…"))
        video_dir = Path(workdir, "video")
        info = video_editor.prepare_video(video_path, video_dir, isolate_voices=isolate)
        state = video_editor.load_state(workdir)
        state["video"] = info
        video_editor.save_state(workdir, state)
        if not info.get("has_audio", True):
            raise ValueError("the video has no sound")
        source = video_dir / "audio16k.wav"
        if info.get("isolated"):
            source = video_dir / "voices16k.wav"  # the separated voices: no music in the measurements
            dubbing.extract_audio(str(video_dir / "voices.wav"), str(source), dubbing.ANALYSIS_SR)
        audio, sr = sf.read(str(source), dtype="float32")
        progress(0.6, desc="Hearing each line's " + ("emotion and tone…" if read_emotion else "tone…"))
        if read_emotion and "emotion" not in speaker_models:
            speaker_models["emotion"] = dubbing.load_emotion_model()
        tones = script_voice.tones_from_audio(
            lines, audio, sr, video_dir / "line_clips", regions=info.get("regions"),
            emotion_model=speaker_models.get("emotion"), music_removed=bool(info.get("isolated")),
            read_emotion=read_emotion,
        )
        return script_voice.apply_tones(lines, tones), sum(1 for t in tones.values() if t.get("music"))

    def _read_tables(speaker_rows, line_rows):
        profiles = {}
        for row in _table_rows(speaker_rows):
            name, gender, age, description, _, voice = (list(row) + [""] * 6)[:6]
            name = str(name).strip()
            if not name:
                continue
            gender = str(gender).strip().lower() or "unknown"
            age = str(age).strip().lower() or "adult"
            if gender not in ("male", "female", "unknown"):
                raise ValueError(f"Gender of {name!r} must be male, female or unknown")
            if age not in script_voice.AGES:
                raise ValueError(f"Age of {name!r} must be adult or kid")
            template = script_voice.template_voice(voice)  # None: designed from their first line
            profiles[name] = dubbing.SpeakerProfile(
                name=name, gender=gender, age=age, voice_mode="clone_first",
                description=str(description).strip() or script_voice.voice_description(gender, age),
                reference_wav=template["path"] if template else None,
            )
        lines = []
        for row in _table_rows(line_rows):
            idx, start, end, speaker, emotion, text, tone = (list(row) + [""] * 7)[:7]
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
                    emotion=_plain_emotion(emotion) or "neutral",
                    tone=str(tone or "").strip(),
                )
            )
        if not lines:
            raise ValueError("The lines table is empty.")
        return lines, profiles

    def _retry_choices(line_rows, report, done):
        """Every line for the regenerate picker; failed (⚠) and not yet generated (⏳) lines are pre-selected."""
        failed = set(script_voice.failed_lines(report)) if report else set()
        by_index = {r["index"]: r for r in report or []}
        choices, selected = [], []
        for row in _table_rows(line_rows):
            index = int(float(row[0]))
            text, speaker = str(row[5]), str(row[3])
            mark = " ⏳" if index not in done else " ⚠" if index in failed else ""
            choices.append((f"#{index} {by_index.get(index, {}).get('speaker', speaker)}{mark} — {text[:30]}", str(index)))
            if mark:
                selected.append(str(index))
        return gr.update(choices=choices, value=selected)

    def _outputs(result, line_rows, notes, focus=()):
        """Everything the page shows for a finished run (also used to restore a reopened page)."""
        line_files = sorted(str(p) for p in Path(result.lines_dir).glob("*.wav"))
        zip_path = shutil.make_archive(str(Path(result.lines_dir).with_name("script_lines")), "zip", result.lines_dir)
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
        picked = next((f for f in line_files if int(Path(f).name[:4]) in set(focus)), line_files[0] if line_files else None)
        speakers = list(dict.fromkeys(r["speaker"] for r in result.report))
        voice_speaker = next((r["speaker"] for r in result.report if r["index"] in set(focus)), speakers[0] if speakers else None)
        done = {r["index"] for r in result.report}
        return (
            result.audio_path,
            gr.update(value=zip_path, interactive=True, label=f"⬇️ Download all {len(line_files)} lines (.zip)"),
            line_files,
            gr.update(choices=[(Path(f).stem, f) for f in line_files], value=picked),
            picked,
            result.tagged_srt_path,
            report_rows,
            notes,
            _retry_choices(line_rows, result.report, done),
            gr.update(choices=speakers, value=voice_speaker),
            result.voices,
            result.voices.get(voice_speaker) if voice_speaker else None,
            gr.update(visible=True),
        )

    def _partial_outputs(workdir, line_rows, notes, tagged=None):
        """What the page shows for a run that is still going or stopped part-way: the lines made so far."""
        lines_dir = Path(workdir, "script_lines")
        line_files = sorted(str(p) for p in lines_dir.glob("*.wav")) if lines_dir.exists() else []
        done = script_voice.generated_lines(workdir)
        return (
            no_change,
            no_change,
            line_files,
            gr.update(choices=[(Path(f).stem, f) for f in line_files], value=line_files[0] if line_files else None),
            line_files[0] if line_files else None,
            tagged if tagged else no_change,
            no_change,
            notes,
            _retry_choices(line_rows, [], done),
            no_change,
            no_change,
            no_change,
            gr.update(visible=bool(line_files)),
        )

    def _job_worker(job, workdir, lines, profiles, options, only, new_voices, verify_voice):
        try:
            job["desc"] = "Loading VoxCPM..."
            voicer = script_voice.ScriptVoicer(demo.get_or_load_voxcpm(), workdir)
            if verify_voice:
                job["desc"] = "Loading voice identification model (CAM++)..."
                if "model" not in speaker_models:
                    speaker_models["model"] = dubbing.load_speaker_model()
                voicer.speaker_model = speaker_models["model"]
            verb = (f"Making {job['speaker']}'s lines" if job.get("speaker") else "Re-voicing" if new_voices
                    else "Regenerating" if only else "Line")

            def _on_progress(i, n, line):
                job["i"], job["n"] = i, n
                job["desc"] = f"{verb} {min(i + 1, n)}/{n} — #{line.index} {line.speaker}"

            out = os.path.join(workdir, "script.wav")
            job["result"] = voicer.voice(
                lines, profiles, out, options, progress=_on_progress, only=only, new_voices=new_voices
            )
            job["status"] = "done"
        except Exception as e:  # reported to the page that is (or will be) open
            logger.exception("SRT voicing failed")
            job["error"], job["status"] = str(e), "failed"

    def _job_note(job) -> str:
        return (
            f"⏳ **Generating in the background — {job['desc'] or 'starting...'}**\n\n"
            "It keeps going even if you close this tab; reopen the page to see the results."
        )

    def _run(
        workdir, speaker_rows, line_rows, verify_voice, max_tries, max_speedup, pad_to_slot, remove_silence,
        max_pause, cfg_value, dit_steps, progress, only=None, new_voices=(), speaker=None,
    ):
        if not workdir:
            raise gr.Error("Please upload an SRT and click Analyze first.")
        job = _start_job(
            workdir, speaker_rows, line_rows,
            [verify_voice, max_tries, max_speedup, pad_to_slot, remove_silence, max_pause, cfg_value, dit_steps],
            only, new_voices, speaker,
        )
        # Follow the job; if the tab is closed it simply carries on in the background.
        while job["status"] == "running":
            progress(job["i"] / max(job["n"], 1), desc=job["desc"] or "Starting...")
            time.sleep(0.5)
        return _finished(job, workdir, line_rows)

    def _start_job(workdir, speaker_rows, line_rows, settings, only=None, new_voices=(), speaker=None):
        """Start a generation in the background (raises gr.Error if the tables are wrong or a run is going).
        ``speaker``: make only that speaker's lines (the others are left for later)."""
        verify_voice, max_tries, max_speedup, pad_to_slot, remove_silence, max_pause, cfg_value, dit_steps = settings
        try:
            lines, profiles = _read_tables(speaker_rows, line_rows)
        except Exception as e:
            raise gr.Error(str(e))
        if speaker is not None:
            only = {l.index for l in lines if l.speaker == speaker}
            if not only:
                raise gr.Error(f"{speaker} has no lines in the script.")
        elif only is not None:
            # Lines that were never generated (e.g. the run was stopped) are always finished too.
            only = set(only) | ({l.index for l in lines} - script_voice.generated_lines(workdir))
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
        with _SRT_JOBS_LOCK:
            if any(j["status"] == "running" for j in _SRT_JOBS.values()):
                raise gr.Error("A generation is already running — wait for it to finish (its progress is shown here).")
            job = {"status": "running", "i": 0, "n": len(lines), "desc": "", "error": None, "result": None,
                   "only": only, "new_voices": set(new_voices), "speaker": speaker}
            _SRT_JOBS[workdir] = job
        _srt_save_tables(workdir, speaker_rows, line_rows, options=list(settings))
        threading.Thread(
            target=_job_worker,
            args=(job, workdir, lines, profiles, options, only, set(new_voices), bool(verify_voice)),
            daemon=True,
        ).start()
        return job

    def _finished(job, workdir, line_rows):
        if job["status"] == "failed":
            raise gr.Error(job["error"])
        result = job["result"]
        for message in result.warnings:
            gr.Warning(message, duration=None)
        notes = _notes(result.warnings, done=True)
        if job.get("speaker"):
            waiting = sorted({str(r[3]) for r in _table_rows(line_rows)
                              if int(float(r[0])) not in {x["index"] for x in result.report}})
            notes = (f"🎙 Made all of **{job['speaker']}**'s lines — listen to them below."
                     + (f" Not generated yet: {', '.join(waiting)} — choose the next speaker, or click **Generate all "
                        "lines** to make the rest." if waiting else "") + "\n\n" + notes)
            focus = sorted(job["only"])
        elif job["new_voices"]:
            notes = f"🎭 New voice for {', '.join(sorted(job['new_voices']))} — all of their lines were regenerated.\n\n" + notes
            focus = [r["index"] for r in result.report if r["speaker"] in job["new_voices"]]
        elif job["only"]:
            notes = f"🔁 Regenerated line(s) {', '.join(f'#{i}' for i in sorted(job['only']))}.\n\n" + notes
            focus = sorted(job["only"])
        else:
            focus = []
        return _outputs(result, line_rows, notes, focus)

    def _generate(
        workdir, speaker_rows, line_rows, verify_voice, max_tries, max_speedup, pad_to_slot, remove_silence,
        max_pause, cfg_value, dit_steps, progress=gr.Progress(),
    ):
        # Some speakers already made (one at a time): make the rest. Nothing or everything made: make all.
        done = script_voice.generated_lines(workdir) if workdir else set()
        total = {int(float(r[0])) for r in _table_rows(line_rows) if str(r[5]).strip()}
        only = set() if done and total - done else None
        return _run(
            workdir, speaker_rows, line_rows, verify_voice, max_tries, max_speedup, pad_to_slot, remove_silence,
            max_pause, cfg_value, dit_steps, progress, only=only,
        )

    def _generate_speaker(
        speaker, workdir, speaker_rows, line_rows, verify_voice, max_tries, max_speedup, pad_to_slot, remove_silence,
        max_pause, cfg_value, dit_steps, progress=gr.Progress(),
    ):
        if not speaker:
            raise gr.Error("Choose the speaker whose lines to make first.")
        return _run(
            workdir, speaker_rows, line_rows, verify_voice, max_tries, max_speedup, pad_to_slot, remove_silence,
            max_pause, cfg_value, dit_steps, progress, speaker=speaker,
        )

    def _regenerate(
        selected, workdir, speaker_rows, line_rows, verify_voice, max_tries, max_speedup, pad_to_slot,
        remove_silence, max_pause, cfg_value, dit_steps, progress=gr.Progress(),
    ):
        if not selected:
            raise gr.Error("Choose the line(s) to regenerate first (failed ⚠ and unfinished ⏳ lines are pre-selected).")
        return _run(
            workdir, speaker_rows, line_rows, verify_voice, max_tries, max_speedup, pad_to_slot, remove_silence,
            max_pause, cfg_value, dit_steps, progress, only={int(v) for v in selected},
        )

    def _revoice(
        speaker, workdir, speaker_rows, line_rows, verify_voice, max_tries, max_speedup, pad_to_slot,
        remove_silence, max_pause, cfg_value, dit_steps, progress=gr.Progress(),
    ):
        if not speaker:
            raise gr.Error("Choose the speaker who needs a new voice first (speakers are listed after Generate).")
        return _run(
            workdir, speaker_rows, line_rows, verify_voice, max_tries, max_speedup, pad_to_slot, remove_silence,
            max_pause, cfg_value, dit_steps, progress, only=(), new_voices={speaker},
        )

    def _restore():
        """Bring a reopened (or reloaded) page back to the last project: tables, generated lines and any run in progress."""
        nothing = (no_change,) * 3 + (no_change,) * 13 + (gr.Timer(active=False),)
        workdir = _srt_current_project()
        if not workdir:
            return nothing
        tables = json.loads(Path(workdir, "tables.json").read_text(encoding="utf-8"))
        speaker_rows, line_rows = _speaker_rows_6(tables.get("speakers", [])), _line_rows_7(tables.get("lines", []))
        tagged = tables.get("tagged")
        head = (speaker_rows, line_rows, workdir)
        job = _SRT_JOBS.get(workdir)
        if job and job["status"] == "running":
            return head + _partial_outputs(workdir, line_rows, _job_note(job), tagged) + (gr.Timer(active=True),)
        output = os.path.join(workdir, "script.wav")
        result = script_voice.load_results(workdir, output)
        done = script_voice.generated_lines(workdir)
        total = len(_table_rows(line_rows))
        if result is not None and len(done) >= total:
            note = "📂 Restored your last project."
            if job and job["status"] == "failed":
                note += f"\n\n⚠️ The last run stopped with an error: {job['error']}"
            return head + _outputs(result, line_rows, note) + (gr.Timer(active=False),)
        note = "📂 Restored your last project." + (
            f" **{len(done)} of {total} lines** were generated before the run stopped — the unfinished lines (⏳) are "
            "selected under **🔁 Lines to regenerate**; click **Regenerate selected lines** to finish them."
            if done else " Click **Generate all lines** to start."
        )
        return head + _partial_outputs(workdir, line_rows, note, tagged) + (gr.Timer(active=False),)

    def _poll(workdir, line_rows):
        """While a background run goes on, update its progress; when it ends, show its results once."""
        job = _SRT_JOBS.get(workdir) if workdir else None
        if not job:
            return (no_change,) * 13 + (gr.Timer(active=False),)
        if job["status"] == "running":
            return _partial_outputs(workdir, line_rows, _job_note(job)) + (gr.Timer(active=True),)
        if job["status"] == "failed":
            return (no_change,) * 7 + (f"⚠️ The run stopped with an error: {job['error']}",) + (no_change,) * 5 + (
                gr.Timer(active=False),
            )
        return _finished(job, workdir, line_rows) + (gr.Timer(active=False),)

    gr.Markdown(_SRT_INSTRUCTIONS)
    workdir_state = gr.State(None)
    voices_state = gr.State({})
    poll_timer = gr.Timer(2.0, active=False)
    with gr.Row():
        with gr.Column():
            srt_input = gr.File(label="📝 Tagged subtitles (.srt)", file_types=[".srt"], type="filepath")
            video_input = gr.File(
                label="🎬 Original video (optional) — each line's emotion and tone are read from its speech",
                file_types=["video", "audio", ".mkv", ".avi", ".mov", ".mp4", ".webm", ".wav", ".mp3", ".m4a"],
                type="filepath",
            )
            video_emotion_input = gr.Checkbox(
                value=False,
                label="Also take each line's emotion from the video (rough)",
                info="Off: only the tone (louder, softer, faster, …) is read. On: lines with no emotion in their tag also get "
                "one from the video — a rough guess (any tense, raised voice is heard as angry)",
                elem_classes=["switch-toggle"],
            )
            auto_voices_input = gr.Checkbox(
                value=True,
                label="🪄 Give each speaker the voice template that fits them best",
                info="From their gender, age and description (and their voice in the original video, if added); "
                "off: a new voice is designed from each speaker's first line",
                elem_classes=["switch-toggle"],
            )
            isolate_input = gr.Checkbox(
                value=False,
                label="Separate the voices from the music first",
                info="Slower (a few minutes, and a one-time download), much more accurate when there is music under the speech",
                elem_classes=["switch-toggle"],
            )
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
            editor_btn = gr.Button(
                "🎬 Next: place the vocals on your video (Video Editor) →", variant="primary", visible=False
            )

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
        datatype=["str", "str", "str", "str", "number", "str"],
        label="🎭 Speakers — gender: male / female / unknown, age: adult / kid; the description is the voice prompt; "
        "Voice: a voice template (choose one below) or auto (designed from their first line)",
        interactive=True,
        wrap=True,
    )
    with gr.Row():
        voice_speaker = gr.Dropdown(choices=[], label="🎙 Give a speaker a voice template — speaker", scale=2)
        voice_template = gr.Dropdown(
            choices=_template_choices(), value=script_voice.AUTO_VOICE, label="Voice (instead of one designed from their first line)",
            scale=3,
        )
        template_preview = gr.Audio(label="Listen to this voice", type="filepath", scale=3)
        apply_voice_btn = gr.Button("✅ Use this voice", variant="secondary", scale=1)
        pick_voices_btn = gr.Button("🪄 Pick the best voices for all speakers", variant="secondary", scale=1)
    with gr.Accordion("🎙 Manage voice templates (kept for all your projects)", open=False):
        gr.Markdown(
            "A voice template is a voice you choose once and use again — for any speaker, in any project — instead of "
            "a new random voice designed from each speaker's first line. Every line of a speaker with a template copies "
            "it and adds its own emotion. **Make the starter voices** once (it loads VoxCPM; about 10 s per voice), make "
            "your own from a description, keep a voice you like from a project, or upload a clip (5–15 s of one person "
            "speaking clearly)."
        )
        starters_btn = gr.Button(
            "✨ Make the starter voices — Man, Woman, Young man, Young woman, Old man, Old woman, Boy, Girl, Kid, Narrator",
            variant="primary",
        )
        library_table = gr.Dataframe(
            headers=["Voice", "Gender", "Age", "Description", "Starter"], label="Voice templates", interactive=False, wrap=True,
        )
        with gr.Row():
            new_voice_name = gr.Textbox(label="New voice: name", placeholder="e.g. Grandpa", scale=2)
            new_voice_gender = gr.Dropdown(["male", "female", "unknown"], value="male", label="Gender", scale=1)
            new_voice_age = gr.Dropdown(list(script_voice.AGES), value="adult", label="Age", scale=1)
            new_voice_desc = gr.Textbox(label="Description (the voice prompt)", placeholder="e.g. Old village chief, deep and slow", scale=4)
            make_voice_btn = gr.Button("🎨 Make this voice", scale=1)
        with gr.Row():
            keep_speaker = gr.Dropdown(choices=[], label="Keep a speaker's voice from this project", scale=2)
            keep_name = gr.Textbox(label="as the template", placeholder="name", scale=2)
            keep_btn = gr.Button("⭐ Save it as a template", scale=1)
        with gr.Row():
            upload_clip = gr.Audio(sources=["upload"], type="filepath", label="Or upload a clip", scale=3)
            upload_name = gr.Textbox(label="as the template", placeholder="name", scale=2)
            upload_gender = gr.Dropdown(["male", "female", "unknown"], value="unknown", label="Gender", scale=1)
            upload_age = gr.Dropdown(list(script_voice.AGES), value="adult", label="Age", scale=1)
            upload_btn = gr.Button("⬆️ Save the clip", scale=1)
        with gr.Row():
            delete_pick = gr.Dropdown(choices=[], label="Delete a voice template", scale=3)
            delete_btn = gr.Button("🗑 Delete", scale=1)
        library_notes = gr.Markdown()
    lines_table = gr.Dataframe(
        headers=_SRT_LINE_HEADERS,
        datatype=["number", "number", "number", "str", "str", "str", "str"],
        label="💬 Lines — edit speaker, emotion (happy, sad, angry, fearful, surprised, … or any word) or tone "
        "(how it is said, e.g. louder and more forceful, speaking faster)",
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
        speaker_gen_picker = gr.Dropdown(
            choices=[], label="…or one speaker at a time — make all of their lines first, listen, then the next speaker",
            scale=4,
        )
        speaker_gen_btn = gr.Button("▶ Generate only this speaker's lines", variant="secondary", scale=1)
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

    # ----- voice templates -----
    def _library_view():
        library = script_voice.load_voice_library()
        rows = [[n, v["gender"], v["age"], v.get("description", ""), "✓" if v.get("starter") else ""]
                for n, v in sorted(library.items(), key=lambda kv: (not kv[1].get("starter"), kv[0]))]
        names = sorted(library)
        return rows, gr.update(choices=_template_choices()), gr.update(choices=names, value=None)

    library_outputs = [library_table, voice_template, delete_pick]

    def _speaker_choices(speaker_rows, current, kept, to_make):
        rows = [r for r in _table_rows(speaker_rows) if str(r[0]).strip()]
        names = [str(r[0]) for r in rows]
        # Speakers with the most lines first in the "one at a time" list (the main characters).
        by_lines = [str(r[0]) for r in sorted(rows, key=lambda r: -float(r[4] or 0))]
        return (gr.update(choices=names, value=current if current in names else (names[0] if names else None)),
                gr.update(choices=names, value=kept if kept in names else None),
                gr.update(choices=by_lines, value=to_make if to_make in names else (by_lines[0] if by_lines else None)))

    def _apply_voice(speaker, template, speaker_rows, workdir):
        if not speaker:
            raise gr.Error("Choose the speaker first.")
        rows = _speaker_rows_6(_table_rows(speaker_rows))
        voice = script_voice.template_voice(template) if template else None
        for row in rows:
            if str(row[0]) == speaker:
                row[5] = template or script_voice.AUTO_VOICE
                if voice:  # the speaker takes the template's gender, age and description
                    row[1], row[2], row[3] = voice["gender"], voice["age"], voice.get("description") or row[3]
        done = script_voice.generated_lines(workdir) if workdir else set()
        note = (f"🎙 **{speaker}** now uses " + (f"the voice template **{template}**." if voice else "a voice designed from their first line.")
                + (" Their lines already made still have the old voice — click **🎭 New voice — redo all their lines** "
                   "(with them selected) to make them again." if done else ""))
        return rows, note

    def _template_audio(template):
        voice = script_voice.template_voice(template) if template else None
        return voice["path"] if voice else None

    def _busy_check():
        if any(j["status"] == "running" for j in _SRT_JOBS.values()):
            raise gr.Error("A generation is running — make voice templates when it has finished.")

    def _template_voicer():
        work = script_voice.VOICE_LIBRARY_DIR / "_work"
        work.mkdir(parents=True, exist_ok=True)
        return script_voice.ScriptVoicer(demo.get_or_load_voxcpm(), work)

    _TEMPLATE_OPTIONS = dict(verify_voice=False, first_line_attempts=12, max_attempts=8, cfg_value=2.0, inference_timesteps=10)

    def _gender_judge():
        """Speaker embeddings of the adult templates (to keep the version of a new voice that sounds most like its
        gender: a boy's voice is as high as a girl's) and the function that embeds a clip. ({}, None) if unavailable."""
        import librosa
        import soundfile as sf

        try:
            if "model" not in speaker_models:
                speaker_models["model"] = dubbing.load_speaker_model()
            model = speaker_models["model"]
        except Exception as e:
            logger.warning("Voice model unavailable for templates: %s", e)
            return {}, None

        def embed(path):
            wav, sr = sf.read(str(path), dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != dubbing.ANALYSIS_SR:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=dubbing.ANALYSIS_SR)
            return dubbing.speaker_embedding(model, wav)

        refs = {"male": [], "female": []}
        for t in script_voice.load_voice_library().values():
            if t.get("age") == "adult" and t.get("gender") in refs:
                e = embed(t["path"])
                if e is not None:
                    refs[t["gender"]].append(e)
        return refs, embed

    def _make_starters(progress=gr.Progress()):
        _busy_check()
        library = script_voice.load_voice_library()
        todo = [(i, v) for i, v in enumerate(script_voice.STARTER_VOICES) if v["name"] not in library]
        if not todo:
            return _library_view() + ("✅ The starter voices are all there.",)
        progress(0, desc="Loading VoxCPM…")
        voicer = _template_voicer()
        made = []
        # Adults first: their voices tell the children's versions apart (which sounds most like a boy / a girl).
        todo.sort(key=lambda iv: iv[1]["age"] != "adult")
        refs, embed = {}, None
        for step, (i, v) in enumerate(todo):
            progress(step / len(todo), desc=f"Making the voice “{v['name']}” ({step + 1}/{len(todo)})…")
            if v["age"] == "kid" and v["gender"] != "unknown" and embed is None:
                refs, embed = _gender_judge()
            script_voice.make_voice_template(
                voicer, v["name"], v["gender"], v["age"], v["description"], seed=1000 + 97 * i,
                options=dubbing.DubOptions(**_TEMPLATE_OPTIONS), starter=True,
                candidates=6 if v["age"] == "kid" else 1, gender_refs=refs, embed=embed,
            )
            made.append(v["name"])
        return _library_view() + (f"✨ Made {len(made)} voice(s): {', '.join(made)}. Listen to them above and give them to your speakers.",)

    def _make_one(name, gender, age, description, progress=gr.Progress()):
        _busy_check()
        if not str(name or "").strip():
            raise gr.Error("Give the new voice a name.")
        progress(0.1, desc="Loading VoxCPM…")
        voicer = _template_voicer()
        progress(0.4, desc=f"Making the voice “{name}”…")
        import zlib

        seed = zlib.crc32(f"{name}|{description}".encode("utf-8")) % 100_000  # same name and description: same voice
        refs, embed = _gender_judge() if gender in ("male", "female") else ({}, None)
        script_voice.make_voice_template(voicer, str(name).strip(), gender, age, str(description or "").strip(), seed=seed,
                                         options=dubbing.DubOptions(**_TEMPLATE_OPTIONS),
                                         candidates=4 if embed else 1, gender_refs=refs, embed=embed)
        return _library_view() + (f"🎨 Made the voice **{name}**.",)

    def _keep_speaker_voice(workdir, speaker, name, speaker_rows):
        if not workdir or not speaker:
            raise gr.Error("Choose a speaker of this project.")
        state_path = Path(workdir, "script_state.json")
        refs = json.loads(state_path.read_text(encoding="utf-8")).get("refs", {}) if state_path.exists() else {}
        if not refs.get(speaker) or not Path(refs[speaker]).exists():
            raise gr.Error(f"{speaker} has no voice yet — generate their lines first.")
        row = next((r for r in _speaker_rows_6(_table_rows(speaker_rows)) if str(r[0]) == speaker), None)
        gender, age, description = (row[1], row[2], row[3]) if row else ("unknown", "adult", "")
        script_voice.save_voice_template(str(name or speaker).strip(), refs[speaker], gender, age, description)
        return _library_view() + (f"⭐ Kept {speaker}'s voice as the template **{str(name or speaker).strip()}**.",)

    def _upload_voice(clip, name, gender, age):
        if not clip:
            raise gr.Error("Upload a clip first (5–15 s of one person speaking clearly).")
        script_voice.save_voice_template(str(name or "").strip() or Path(clip).stem, clip, gender, age, "Uploaded clip")
        return _library_view() + (f"⬆️ Saved the clip as the template **{str(name or '').strip() or Path(clip).stem}**.",)

    def _delete_voice(name):
        if not name:
            raise gr.Error("Choose the voice template to delete.")
        script_voice.delete_voice_template(name)
        return _library_view() + (f"🗑 Deleted the voice template **{name}** — give the speakers who used it another voice (or auto) before generating.",)

    speakers_table.change(fn=_speaker_choices, inputs=[speakers_table, voice_speaker, keep_speaker, speaker_gen_picker],
                          outputs=[voice_speaker, keep_speaker, speaker_gen_picker], show_progress="hidden")
    voice_template.change(fn=_template_audio, inputs=voice_template, outputs=template_preview, show_progress="hidden")
    def _pick_all(workdir, speaker_rows, line_rows):
        if not _table_rows(speaker_rows):
            raise gr.Error("Analyze a script first.")
        try:
            lines, _ = _read_tables(speaker_rows, line_rows)
        except Exception:
            lines = []
        rows, note = _pick_voices(workdir, speaker_rows, lines)
        done = script_voice.generated_lines(workdir) if workdir else set()
        if done:
            note += ("Lines already made keep their old voices — select a speaker under **🎭 New voice** and click "
                     "**New voice — redo all their lines** to remake them with the new voice.")
        return rows, note

    pick_voices_btn.click(fn=_pick_all, inputs=[workdir_state, speakers_table, lines_table], outputs=[speakers_table, notes_box])
    apply_voice_btn.click(fn=_apply_voice, inputs=[voice_speaker, voice_template, speakers_table, workdir_state],
                          outputs=[speakers_table, notes_box])
    starters_btn.click(fn=_make_starters, outputs=library_outputs + [library_notes])
    make_voice_btn.click(fn=_make_one, inputs=[new_voice_name, new_voice_gender, new_voice_age, new_voice_desc],
                         outputs=library_outputs + [library_notes])
    keep_btn.click(fn=_keep_speaker_voice, inputs=[workdir_state, keep_speaker, keep_name, speakers_table],
                   outputs=library_outputs + [library_notes])
    upload_btn.click(fn=_upload_voice, inputs=[upload_clip, upload_name, upload_gender, upload_age],
                     outputs=library_outputs + [library_notes])
    delete_btn.click(fn=_delete_voice, inputs=delete_pick, outputs=library_outputs + [library_notes])
    blocks.load(fn=_library_view, outputs=library_outputs)

    analyze_btn.click(
        fn=_analyze,
        inputs=[srt_input, guess_emotion, video_input, isolate_input, video_emotion_input, auto_voices_input],
        outputs=[speakers_table, lines_table, tagged_output, workdir_state, notes_box],
    )
    settings = [
        workdir_state, speakers_table, lines_table, verify_voice, max_tries, max_speedup, pad_to_slot,
        remove_silence, max_pause, cfg_value, dit_steps,
    ]
    results = [
        audio_output, zip_output, line_files_output, line_picker, line_player, tagged_output, report_table, notes_box,
        retry_picker, revoice_picker, voices_state, voice_preview, editor_btn,
    ]
    generate_btn.click(fn=_generate, inputs=settings, outputs=results, api_name="voice_srt")
    speaker_gen_btn.click(fn=_generate_speaker, inputs=[speaker_gen_picker] + settings, outputs=results,
                          api_name="voice_srt_speaker")
    retry_btn.click(fn=_regenerate, inputs=[retry_picker] + settings, outputs=results, api_name="regenerate_srt_lines")
    revoice_btn.click(fn=_revoice, inputs=[revoice_picker] + settings, outputs=results, api_name="revoice_srt_speaker")
    line_picker.change(fn=lambda path: path, inputs=line_picker, outputs=line_player)
    revoice_picker.change(
        fn=lambda name, voices: (voices or {}).get(name), inputs=[revoice_picker, voices_state], outputs=voice_preview
    )
    seen_rev = gr.State(-1)

    def _restore_with_rev():
        workdir = _srt_current_project()
        return _restore() + (_srt_tables_rev(workdir) if workdir else -1,)

    def _reopen(rev):
        """Back on this tab: reload the project if the video editor changed it (edited or regenerated lines)."""
        workdir = _srt_current_project()
        current = _srt_tables_rev(workdir) if workdir else -1
        if current == rev and not (workdir and _SRT_JOBS.get(workdir, {}).get("status") == "running"):
            return (no_change,) * 3 + (no_change,) * 13 + (no_change, no_change)
        return _restore_with_rev()

    restore_outputs = [speakers_table, lines_table, workdir_state] + results + [poll_timer, seen_rev]
    blocks.load(fn=_restore_with_rev, outputs=restore_outputs, api_name="restore_srt")
    poll_timer.tick(fn=_poll, inputs=[workdir_state, lines_table], outputs=results + [poll_timer])

    def _leave(workdir, speaker_rows, line_rows):
        """Save the tables (with any edits not yet generated) so the video editor sees them."""
        if workdir and Path(workdir, "tables.json").exists():
            _srt_save_tables(workdir, speaker_rows, line_rows)

    return {
        "editor_btn": editor_btn,
        "leave": _leave,
        "start_job": _start_job,
        "reopen": _reopen,
        "reopen_inputs": [seen_rev],
        "reopen_outputs": restore_outputs,
        "enter_inputs": [workdir_state, speakers_table, lines_table],
    }


# ---------- Video Editor tab ----------

_EDITOR_ASSETS = Path(__file__).resolve().parent / "assets" / "video_editor"


def _file_url(path: str, version=None) -> str:
    return f"/gradio_api/file={path}" + (f"?v={version}" if version is not None else "")


def build_editor_tab(tabs: gr.Tabs, editor_tab: gr.Tab, srt: dict) -> None:
    """A small video editor for the SRT → Speech project: the generated lines as clips on a timeline over the
    video, lined up with the speech found in the video; move, trim, mute, regenerate and export."""
    import shutil

    import soundfile as sf

    from gradio.utils import get_upload_folder

    from voxcpm import script_voice, video_editor

    def _project():
        workdir = _srt_current_project()
        if not workdir:
            raise ValueError("No SRT → Speech project yet — analyze and generate an SRT first.")
        return workdir

    def _tables(workdir):
        return json.loads(Path(workdir, "tables.json").read_text(encoding="utf-8"))

    def _job(workdir):
        job = _SRT_JOBS.get(workdir)
        if not job:
            return None
        return {k: job.get(k) for k in ("status", "i", "n", "desc", "error")} | {
            "only": sorted(job.get("only") or []), "new_voices": sorted(job.get("new_voices") or []),
        }

    def _payload(workdir):
        tables = _tables(workdir)
        state = video_editor.load_state(workdir)
        script_state_path = Path(workdir, "script_state.json")
        script_state = json.loads(script_state_path.read_text(encoding="utf-8")) if script_state_path.exists() else {}
        entries, rounds = script_state.get("entries", {}), script_state.get("rounds", {})
        report_path = Path(workdir, "report.json")
        placed = {}
        if report_path.exists():
            for r in json.loads(report_path.read_text(encoding="utf-8")):
                placed[int(r["index"])] = float(r["start"]) + float(r.get("shift_s") or 0)
        lines, changed = [], False
        placed_lines = {video_editor.clip_line(k, c) for k, c in state["clips"].items()}
        for row in _table_rows(tables.get("lines", [])):
            idx, start, end, speaker, emotion, text, tone = (list(row) + [""] * 7)[:7]
            try:
                index = int(float(idx))
            except (TypeError, ValueError):
                continue
            start, end = float(start or 0), float(end or 0)
            raw = Path(workdir, "raw", f"{index:04d}.wav")
            entry = entries.get(str(index))
            line = {
                "index": index, "start": start, "end": end, "speaker": str(speaker or script_voice.DEFAULT_SPEAKER),
                "emotion": _plain_emotion(emotion), "text": str(text), "tone": str(tone or ""),
                "url": None, "duration": 0.0, "status": "⏳ not generated", "ok": False,
            }
            if entry and raw.exists():
                line.update(
                    url=_file_url(str(raw), f"{rounds.get(str(index), 0)}-{int(raw.stat().st_mtime)}"),
                    duration=round(sf.info(str(raw)).duration, 3),
                    status=_line_status(entry),
                    ok=_line_status(entry).startswith("✅"),
                )
            if index not in placed_lines:
                state["clips"][str(index)] = video_editor.default_clip(placed.get(index, start))
                changed = True
            lines.append(line)
        if changed:
            video_editor.save_state(workdir, state)
        video = state.get("video")
        if video and not Path(video.get("preview", "")).exists():
            video = None
        if video and not (state["mix"].get("subs") or {}).get("box"):
            # Subtitles found in the picture are filled in by default; the editor shows (and moves) the area.
            box = video.get("subtitle_box")
            state["mix"]["subs"] = {"mode": "fill" if box else "off", "box": box or [0.1, 0.75, 0.8, 0.07]}
        if video:
            video = dict(video, url=_file_url(video["preview"]))
            if video.get("music") and Path(video["music"]).exists():
                video["music_url"] = _file_url(video["music"])
        return {
            "workdir": workdir,
            "project": Path(workdir).name,
            "lines": lines,
            "speakers": [str(r[0]) for r in _table_rows(tables.get("speakers", []))],
            "clips": state["clips"],
            "mix": state["mix"],
            "tracks": state.get("tracks", {}),
            "subtitles": state.get("subtitles"),
            "video": video,
            "job": _job(workdir),
        }

    def _safe(fn):
        """Server functions answer {"error": ...} instead of failing, so the editor can show the message."""

        def wrapper(arg=None):
            try:
                return fn(arg if isinstance(arg, dict) else {})
            except gr.Error as e:
                return {"error": e.message}
            except Exception as e:
                logger.exception("Video editor: %s failed", fn.__name__)
                return {"error": str(e)}

        wrapper.__name__ = fn.__name__
        return wrapper

    @_safe
    def editor_load(arg):
        workdir = _project()
        state = video_editor.load_state(workdir)
        # Videos added before burned-in subtitles were looked for get looked at now (once).
        if state.get("video") and video_editor.ensure_subtitle_box(state["video"]):
            video_editor.save_state(workdir, state)
        return _payload(workdir)

    @_safe
    def editor_find_subtitles(arg):
        workdir = _project()
        state = video_editor.load_state(workdir)
        if not state.get("video"):
            raise ValueError("Add a video first.")
        state["video"].pop("subtitle_box", None)
        video_editor.ensure_subtitle_box(state["video"])
        box = state["video"].get("subtitle_box")
        if box:
            state["mix"]["subs"] = {"mode": (state["mix"].get("subs") or {}).get("mode", "fill"), "box": box}
        video_editor.save_state(workdir, state)
        return _payload(workdir) | {"found": bool(box)}

    @_safe
    def editor_video(arg):
        workdir = _project()
        path = Path(str(arg.get("path") or "")).resolve()
        if not path.is_file() or Path(get_upload_folder()).resolve() not in path.parents:
            raise ValueError("The video upload did not arrive — please try again.")
        if not video_editor.has_ffmpeg():
            raise ValueError("ffmpeg is needed for the video editor — install it (e.g. `brew install ffmpeg`).")
        info = video_editor.prepare_video(str(path), Path(workdir, "video"), isolate_voices=bool(arg.get("isolate")))
        state = video_editor.load_state(workdir)
        state["video"] = info
        video_editor.save_state(workdir, state)
        return _payload(workdir)

    @_safe
    def editor_detect(arg):
        workdir = _project()
        state = video_editor.load_state(workdir)
        info = state.get("video")
        if not info:
            raise ValueError("Add a video first.")
        sensitivity = float(arg.get("sensitivity", 0.5))
        if arg.get("isolate") and not info.get("isolated"):
            info.update(video_editor.isolate_and_detect(info, Path(workdir, "video"), sensitivity))
        else:
            info["regions"] = video_editor.redetect(info, Path(workdir, "video"), sensitivity)
        info["sensitivity"] = sensitivity
        video_editor.save_state(workdir, state)
        return _payload(workdir)

    @_safe
    def editor_speed(arg):
        """The line's audio at another speed, pitch kept (what the editor plays for a sped-up / slowed clip)."""
        workdir = _project()
        line = int(arg.get("line"))
        raw = Path(workdir, "raw", f"{line:04d}.wav")
        if not raw.exists():
            raise ValueError(f"Line #{line} has not been generated yet.")
        out = video_editor.stretched_file(raw, float(arg.get("speed") or 1), Path(workdir, "stretched"))
        return {"url": _file_url(str(out))}

    @_safe
    def editor_save(arg):
        workdir = _project()
        state = video_editor.load_state(workdir)
        for key in ("clips", "mix", "tracks", "subtitles"):
            if isinstance(arg.get(key), dict):
                state[key] = arg[key]
        video_editor.save_state(workdir, state)
        return {"ok": True}

    def _apply_line_edits(workdir, edits):
        """Write text / emotion / speaker changes made in the editor into the project tables."""
        if not edits:
            return
        tables = _tables(workdir)
        rows = _line_rows_7(_table_rows(tables.get("lines", [])))
        by_index = {int(e["index"]): e for e in edits}
        known = {str(r[0]) for r in _table_rows(tables.get("speakers", []))}
        for row in rows:
            edit = by_index.get(int(float(row[0])))
            if not edit:
                continue
            if edit.get("speaker") is not None:
                if str(edit["speaker"]) not in known:
                    raise ValueError(f"Unknown speaker {edit['speaker']!r}.")
                row[3] = str(edit["speaker"])
            if edit.get("emotion") is not None:
                row[4] = str(edit["emotion"]).strip() or "neutral"
            if edit.get("tone") is not None:
                row[6] = str(edit["tone"]).strip()
            if edit.get("text") is not None and str(edit["text"]).strip():
                row[5] = str(edit["text"]).strip()
        _srt_save_tables(workdir, tables.get("speakers", []), rows, bump=True)

    @_safe
    def editor_edit_lines(arg):
        workdir = _project()
        _apply_line_edits(workdir, arg.get("edits") or [])
        return _payload(workdir)

    @_safe
    def editor_regenerate(arg):
        workdir = _project()
        _apply_line_edits(workdir, arg.get("edits") or [])
        tables = _tables(workdir)
        options = (list(tables.get("options") or []) + _SRT_DEFAULT_OPTIONS[len(tables.get("options") or []):])[:8]
        if arg.get("new_voice"):
            redo = {int(float(r[0])) for r in _table_rows(tables["lines"]) if str(r[3]) == str(arg["new_voice"])}
            srt["start_job"](workdir, tables["speakers"], tables["lines"], options, only=(),
                             new_voices={str(arg["new_voice"])})
        else:
            redo = {int(i) for i in arg.get("indices") or []}
            if not redo:
                raise ValueError("Select the clip(s) to regenerate first.")
            srt["start_job"](workdir, tables["speakers"], tables["lines"], options, only=redo)
        # A regenerated line gets new audio, so a line that was cut into pieces becomes one clip again.
        state = video_editor.load_state(workdir)
        state["clips"] = video_editor.collapse_pieces(state["clips"], redo)
        video_editor.save_state(workdir, state)
        # The SRT tab shows the new results when it is opened again.
        _srt_save_tables(workdir, tables["speakers"], tables["lines"], bump=True)
        return {"job": _job(workdir)}

    @_safe
    def editor_status(arg):
        workdir = _project()
        job = _job(workdir)
        if job and job["status"] != "running":
            return _payload(workdir)
        return {"job": job}

    def _subtitle_images(folder: Path, items: list) -> list:
        """The subtitle images drawn by the editor (base64 PNG) as files, with when and where to show them."""
        import base64

        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
        overlays = []
        for i, item in enumerate(items):
            png = base64.b64decode(str(item.get("png") or ""), validate=True)
            if not png.startswith(b"\x89PNG"):
                raise ValueError("A subtitle image is not a PNG.")
            path = folder / f"{i:04d}.png"
            path.write_bytes(png)
            overlays.append((float(item["start"]), float(item["end"]), int(item.get("x") or 0), int(item.get("y") or 0), str(path)))
        return overlays

    @_safe
    def editor_export(arg):
        workdir = _project()
        state = video_editor.load_state(workdir)
        for key in ("clips", "mix", "tracks", "subtitles"):
            if isinstance(arg.get(key), dict):
                state[key] = arg[key]
        video_editor.save_state(workdir, state)
        info = state.get("video")
        if not info:
            raise ValueError("Add a video first.")
        mix = state["mix"]
        # The original sound switched off (🔇 in the editor) leaves it out whatever mode is chosen.
        mode = mix.get("original", "duck") if mix.get("original_on", True) else "mute"
        muted_tracks = {name for name, t in (state.get("tracks") or {}).items() if t.get("muted")}
        tables = _tables(workdir)
        speakers = {int(float(r[0])): str(r[3]) for r in _table_rows(tables.get("lines", []))}
        clips, sr, audio = [], None, {}
        for key, clip in state["clips"].items():
            line = video_editor.clip_line(key, clip)
            raw = Path(workdir, "raw", f"{line:04d}.wav")
            if not raw.exists() or line not in speakers:
                continue
            if line not in audio:  # a line cut into pieces is read once
                audio[line], rate = sf.read(str(raw), dtype="float32")
                sr = sr or rate
            if speakers[line] in muted_tracks:
                clip = dict(clip, muted=True)
            clips.append((clip, audio[line]))
        if not clips:
            raise ValueError("No generated lines to place on the video yet.")
        out_dir = Path(workdir, "export")
        out_dir.mkdir(exist_ok=True)
        original = full = music = None
        # Parts of the original sound cut in the editor, each with its own volume and sound (none: one part).
        sections = [x for x in (mix.get("orig_sections") or []) if float(x.get("end", 0)) > float(x.get("start", 0))]
        if not mix.get("original_on", True):
            sections = []
        sounds = {x.get("sound") or "auto" for x in sections}
        has_audio = info.get("has_audio", True)

        def music_track():
            if not (info.get("music") and Path(info["music"]).exists()):
                info.update(video_editor.isolate_and_detect(info, Path(workdir, "video")))
                video_editor.save_state(workdir, state)
            return video_editor.read_soundtrack(info["music"], sr, out_dir / "music.wav")

        if has_audio and (mode not in ("mute", "music") or "full" in sounds):
            full = video_editor.read_soundtrack(info["source"], sr, out_dir / "original.wav")
        if has_audio and (mode == "music" or "music" in sounds):
            music = music_track()
        if mode != "mute" and has_audio:
            original = music if mode == "music" else full
        total = float(info["duration"])
        stem = re.sub(r"[^\w\-]+", "_", Path(info.get("name") or "video").stem)[:40] or "video"
        # "Level voices": every clip at the same loudness, as set in the editor's 🎚 Voice levels panel
        level = mix.get("level", True) and {"target": mix.get("level_target"), "strength": mix.get("level_strength")}
        level = video_editor.level_options(level and {k: v for k, v in level.items() if v is not None})
        vocals = video_editor.render_mix(
            clips, total, sr, None, "mute", vocals_gain_db=float(mix.get("vocals_gain_db", 0)), level=level
        )
        vocals_path = out_dir / f"{stem}_vocals.wav"
        sf.write(str(vocals_path), vocals, sr)
        mixed = video_editor.render_mix(
            clips, total, sr, original, mode,
            original_gain_db=float(mix.get("original_gain_db", 0)), vocals_gain_db=float(mix.get("vocals_gain_db", 0)),
            regions=info.get("regions") or [], level=level,
            sections=sections or None, full=full, music=music,
        )
        mix_path = out_dir / f"{stem}_soundtrack.wav"
        sf.write(str(mix_path), mixed, sr)
        version = int(time.time())
        result = {}
        if info.get("has_video", True):
            video_path = out_dir / f"{stem}_dubbed{video_editor.export_extension(info['source'])}"
            video_editor.export_video(
                info["source"], str(mix_path), video_path, subtitles=mix.get("subs"),
                overlays=_subtitle_images(out_dir / "subtitles", arg.get("overlays") or []),
            )
            result = {"video_url": _file_url(str(video_path), version), "video_name": video_path.name}
        # An audio file (no picture) exports as the soundtrack only.
        return result | {
            "vocals_url": _file_url(str(vocals_path), version), "vocals_name": vocals_path.name,
            "soundtrack_url": _file_url(str(mix_path), version), "soundtrack_name": mix_path.name,
        }

    # The editor itself runs in the browser (assets/video_editor) and calls the functions above.
    editor_html = gr.HTML(
        value=json.dumps({"open": 0}),
        html_template=(_EDITOR_ASSETS / "editor.html").read_text(encoding="utf-8"),
        js_on_load=(_EDITOR_ASSETS / "editor.js").read_text(encoding="utf-8"),
        apply_default_css=False,
        elem_id="video-editor",
        server_functions=[
            editor_load, editor_video, editor_detect, editor_save, editor_edit_lines, editor_regenerate,
            editor_status, editor_export, editor_speed, editor_find_subtitles,
        ],
    )

    def _enter(workdir, speaker_rows, line_rows):
        srt["leave"](workdir, speaker_rows, line_rows)
        return json.dumps({"open": time.time()})

    def _open_from_srt(workdir, speaker_rows, line_rows):
        return gr.Tabs(selected="editor"), _enter(workdir, speaker_rows, line_rows)

    srt["editor_btn"].click(
        fn=_open_from_srt, inputs=srt["enter_inputs"], outputs=[tabs, editor_html], show_progress="hidden"
    )
    editor_tab.select(fn=_enter, inputs=srt["enter_inputs"], outputs=editor_html, show_progress="hidden")


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

        with gr.Tabs() as tabs:
            with gr.Tab(I18N("tab_tts"), id="tts"):
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

            with gr.Tab(I18N("tab_srt"), id="srt") as srt_tab:
                srt = build_srt_tab(demo, interface)
            with gr.Tab(I18N("tab_editor"), id="editor") as editor_tab:
                build_editor_tab(tabs, editor_tab, srt)
        # Back on the SRT tab after editing in the video editor: show the edited / regenerated lines.
        srt_tab.select(
            fn=srt["reopen"], inputs=srt["reopen_inputs"], outputs=srt["reopen_outputs"], show_progress="hidden"
        )

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
        css=_CUSTOM_CSS + (_EDITOR_ASSETS / "editor.css").read_text(encoding="utf-8"),
        # Projects and the voice templates (played in the page) are served from where they are kept.
        allowed_paths=[str(SRT_PROJECTS_DIR), str(_voice_library_dir())],
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
