import os
import json
import datetime
import shutil
import secrets
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import gradio as gr
import soundfile as sf
import numpy as np

from config import (
    BASE_DIR,
    SHARE_DIR,
    SHARE_SERVER_PORT,
    REFERENCE_AUDIO_DIR,
    ALLOWED_AUDIO_FORMAT,
)

from utils import (
    get_sovits_model_list,
    sovits_convert_audio,
)

# ── LoRA helpers ──
LORA_DIR = BASE_DIR / "models" / "lora"


def scan_lora_checkpoints() -> list[str]:
    checkpoints: list[str] = []
    if not LORA_DIR.exists():
        return checkpoints
    for entry in sorted(LORA_DIR.iterdir(), reverse=True):
        if entry.is_dir() and (entry / "lora_weights.safetensors").exists():
            checkpoints.append(entry.name)
    return checkpoints


def _get_default_lora_config():
    from voxcpm.model.voxcpm import LoRAConfig

    return LoRAConfig(
        enable_lm=True,
        enable_dit=True,
        r=32,
        alpha=16,
        target_modules_lm=["q_proj", "v_proj", "k_proj", "o_proj"],
        target_modules_dit=["q_proj", "v_proj", "k_proj", "o_proj"],
    )


def _load_lora_config_from_checkpoint(lora_path: Path):
    config_file = lora_path / "lora_config.json"
    if not config_file.exists():
        return None, None
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            info = json.load(f)
        lora_cfg_dict = info.get("lora_config", {})
        base_model = info.get("base_model")
        if lora_cfg_dict:
            from voxcpm.model.voxcpm import LoRAConfig

            return LoRAConfig(**lora_cfg_dict), base_model
    except Exception as exc:
        print(f"[LoRA] Warning: {exc}")
    return None, None


def start_share_server():
    """Serve `SHARE_DIR` directory on SHARE_SERVER_PORT in a background thread."""
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(SHARE_DIR), **kwargs)

        def log_message(self, format, *args):
            pass

    def run():
        server = HTTPServer(("0.0.0.0", SHARE_SERVER_PORT), Handler)
        server.serve_forever()

    t = threading.Thread(target=run, daemon=True)
    t.start()


# ── Global state ──
# VoxCPM instance
_model = None

# OmniVoice instance
_omnivoice_model = None

# 当前激活的模型类型
_current_model_type = "VoxCPM"

# SVC model state
_sovits_models_available = []


def load_model(
    model_type: str,
    model_path: str,
    device: str,
    voxcpm_load_denoiser: bool,
    omnivoice_device_map: str,
    omnivoice_dtype: str,
    lora_enabled: bool=False,
    lora_name: str="NotAvailable",
) -> str:
    global _model, _omnivoice_model, _current_model_type
    try:
        if model_type == "OmniVoice":
            from omnivoice import OmniVoice
            import torch

            dtype_map = {"float16": torch.float16, "float32": torch.float32}
            dt = dtype_map.get(omnivoice_dtype, torch.float16)
            _omnivoice_model = OmniVoice.from_pretrained(
                model_path,
                device_map=omnivoice_device_map,
                dtype=dt,
            )
            _model = None
            _current_model_type = "OmniVoice"
            return f"✅ OmniVoice 加载成功：{model_path}  |  {omnivoice_device_map}  |  {omnivoice_dtype}"
        elif model_type == 'VoxCPM' :
            from voxcpm import VoxCPM
            _omnivoice_model = None

            lora_config = None
            lora_weights_path = None
            if lora_enabled and lora_name and lora_name != "NotAvailable":
                full_lora_path = LORA_DIR / lora_name
                if full_lora_path.exists():
                    lora_weights_path = str(full_lora_path)
                    lora_config, _base_model = _load_lora_config_from_checkpoint(full_lora_path)
                    if lora_config is None:
                        lora_config = _get_default_lora_config()
                    print(f"Loading LoRA from {lora_weights_path}")

            _model = VoxCPM.from_pretrained(
                model_path,
                load_denoiser=voxcpm_load_denoiser,
                device=device,
                lora_config=lora_config,
                lora_weights_path=lora_weights_path,
            )
            _current_model_type = "VoxCPM"
            
            lora_info = f"  |  LoRA：{lora_name}" if (lora_enabled and lora_name and lora_name != "NotAvailable") else ""
            return f"✅ VoxCPM 加载成功：{model_path}  |  设备：{device}  |  降噪器：{'开启' if voxcpm_load_denoiser else '关闭'}{lora_info}"
    except Exception as e:
        _model = None
        _omnivoice_model = None
        return f"❌ 模型加载失败：{e}"


def svc_model_exists(model_name: str) -> bool:
    """检查指定 SVC 模型的必要文件是否存在"""
    if not model_name or model_name == "NotAvailable":
        return False
    
    for m in _sovits_models_available:
        if m['model_name'] == model_name:
            return (m['model_path'].exists() and m['speaker_path'].exists())
    return False

def load_json_file(file_obj):
    if file_obj is None:
        return "", "未选择文件", "[]"
    try:
        fpath = file_obj if isinstance(file_obj, str) else file_obj.name
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return "", "❌ JSON 文件必须是一个字符串数组", "[]"
        preview = "\n".join(f"{i+1}. {t}" for i, t in enumerate(data))
        return preview, f"✅ 已加载 {len(data)} 条文本", json.dumps(data, ensure_ascii=False)
    except Exception as e:
        return "", f"❌ 解析失败：{e}", "[]"


def load_reference_audio() -> list[str] :
    """从预设路径读取所有参考音频文件

    Returns:
        list[str]: 参考音频文件名
    """
    audio_list = []
    for i in REFERENCE_AUDIO_DIR.iterdir() :
        # 仅限支持的音频文件；以 - 分隔的文件命名，第一节是数字
        if i.suffix in ALLOWED_AUDIO_FORMAT \
            and i.stem.split('-')[0].isdigit() :
            audio_list.append(i.name)
    return sorted(audio_list)


def parse_reference_info(audio_name: str) -> dict[str, str] :
    """根据参考音频文件加载对应的文本信息

    Args:
        audio_name (str): 参考音频

    Returns:
        dict[str, str]: { desc: 感情色彩描述, text: 文本 }
    """
    audio_path = REFERENCE_AUDIO_DIR / audio_name
    num, desc = audio_path.stem.split('-', 1)
    text_path = audio_path.with_name(f"{num}.txt")
    return {
        'desc': desc,
        'text': text_path.read_text().strip(),
    }


def parse_textarea(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


#  Share page HTML template
SHARE_PAGE_TEMPLATE = """\
<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Voxbox · 分享音频</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=IBM+Plex+Mono:wght@300;400;500&display=swap');
  :root {{
    --bg:#0d0f14; --panel:#13161e; --border:#252a38;
    --accent:#5d8aff; --accent2:#b66dff; --text:#e8ecf5; --muted:#697089;
  }}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{
    background:var(--bg);color:var(--text);
    font-family:'IBM Plex Mono',monospace;
    min-height:100vh;display:flex;flex-direction:column;
    align-items:center;padding:48px 20px;
  }}
  .card{{
    background:var(--panel);border:1px solid var(--border);
    border-radius:16px;padding:40px;max-width:680px;width:100%;
    box-shadow:0 8px 40px rgba(0,0,0,.5);
  }}
  .logo{{
    font-family:'Syne',sans-serif;font-weight:800;font-size:1.5rem;
    background:linear-gradient(120deg,var(--accent),var(--accent2));
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    margin-bottom:32px;letter-spacing:-0.5px;
  }}
  .section-label{{
    font-family:'Syne',sans-serif;font-size:.68rem;font-weight:600;
    text-transform:uppercase;letter-spacing:.15em;color:var(--accent);
    border-left:3px solid var(--accent);padding-left:10px;margin-bottom:12px;
  }}
  .text-block{{
    background:#0a0c12;border:1px solid var(--border);
    border-radius:10px;padding:20px;font-size:.9rem;
    line-height:1.9;color:var(--text);
    white-space:pre-wrap;word-break:break-word;margin-bottom:28px;
  }}
  .audio-wrap{{margin-bottom:28px}}
  audio{{
    width:100%;border-radius:8px;
    filter:invert(1) hue-rotate(180deg);
  }}
  .meta{{font-size:.72rem;color:var(--muted);line-height:1.9}}
  .meta span{{color:var(--text)}}
  footer{{
    margin-top:24px;font-size:.7rem;color:var(--muted);
    text-align:center;line-height:1.7;
  }}
</style>
</head>
<body>
<div class="card">
  <div class="logo">⟡ Voxbox</div>

  <div class="section-label">生成文本</div>
  <div class="text-block">{text_content}</div>

  <div class="section-label">合成音频</div>
  <div class="audio-wrap">
    <audio controls autoplay>
      <source src="{audio_filename}" type="audio/wav">
      您的浏览器不支持音频播放。
    </audio>
  </div>

  <div class="meta">
    生成时间：<span>{created_at}</span><br>
    文件名：<span>{audio_filename}</span>
  </div>
</div>
<footer>由 Voxbox 生成 · 语音合成</footer>
</body>
</html>
"""


def create_share_page(audio_path: str, texts: list[str], server_host: str) -> tuple[str, str]:
    """
    Creates share/<token>/index.html + copies audio file.
    Returns (share_url, status_message).
    """
    if not audio_path or not os.path.isfile(audio_path):
        return "", "❌ 没有可分享的音频，请先在「批量生成」页完成生成"

    token = secrets.token_urlsafe(12)
    page_dir = SHARE_DIR / token
    page_dir.mkdir(parents=True, exist_ok=True)

    # Copy audio into share dir
    audio_src = Path(audio_path)
    shutil.copy2(audio_path, page_dir / audio_src.name)

    # Build text display
    if len(texts) == 1:
        text_content = texts[0]
    else:
        text_content = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))

    created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = SHARE_PAGE_TEMPLATE.format(
        text_content=text_content,
        audio_filename=audio_src.name,
        created_at=created_at,
    )
    (page_dir / "index.html").write_text(html, encoding="utf-8")

    host = server_host.strip().rstrip("/") if server_host.strip() else f"http://localhost:{SHARE_SERVER_PORT}"
    share_url = f"{host}/{token}/index.html"

    return share_url, f"✅ 分享页面已生成  |  token: {token}"


# ─────────────────────────────────────────────
#  Core generation
# ─────────────────────────────────────────────
def generate_audio(
    config_switch: str,
    text_source: str,
    textarea_text: str,
    json_texts_state: str,
    ref_audio_path: str,
    cfg_value: float,
    inference_timesteps: int,
    output_dir: str,
    gap_seconds: float,
    merge_audio: bool,
    omnivoice_ref_text: str,
    omnivoice_num_step: int,
    omnivoice_speed: float,
    omnivoice_use_duration: bool,
    omnivoice_duration: float,
    use_seed: bool=False,
    seed: int=None,
    svc_convert: bool = False,
    svc_model_name: str = "",
    device: str = 'cuda',
    progress=gr.Progress(),
):
    if _model is None and _omnivoice_model is None:
        return None, "❌ 请先加载模型", "[]"

    if config_switch == "单次生成":
        texts = [textarea_text.strip()] if textarea_text.strip() else []
    else:
        if text_source == "textarea":
            texts = parse_textarea(textarea_text)
        else:
            try:
                texts = json.loads(json_texts_state) if json_texts_state else []
            except Exception:
                texts = []

    if not texts:
        return None, "❌ 文本列表为空，请输入或上传文本", "[]"

    ref_num = ref_audio_path.split('-', 1)[0]
    ref_audio_path = str(REFERENCE_AUDIO_DIR / ref_audio_path)
    if not ref_audio_path or not os.path.isfile(ref_audio_path):
        return None, f"❌ 参考音频路径无效：{ref_audio_path}", "[]"
    
    # Forbidden full path
    out_path = BASE_DIR / output_dir.lstrip('/')
    out_path.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if _current_model_type == "VoxCPM" :
        lora_enabled_tag = '_lora' if _model.lora_enabled else ''
        tag = f"ref{ref_num}_cfg{cfg_value}_step{inference_timesteps}{lora_enabled_tag}_{ts}" 
    elif _current_model_type == 'OmniVoice' :
        tag = f"ref{ref_num}_numstep{omnivoice_num_step}_speed{omnivoice_speed}_{ts}"

    # 原始音频数据
    wav_list: list[np.ndarray] = []

    # 默认用于预览的音频
    default_preview: str = ''

    # SVC 转换后的音频数据
    svc_wav_list: list[np.ndarray] = []

    svc_sample_rate = 32000

    log_lines: list[str] = []

    total_steps = len(texts)

    if _current_model_type == "VoxCPM":
        sample_rate = _model.tts_model.sample_rate
        for i, text in enumerate(texts, start=1):
            try:
                kwargs = {
                    "text": text,
                    "reference_wav_path": ref_audio_path,
                    "cfg_value": cfg_value,
                    "inference_timesteps": int(inference_timesteps),
                }
                if use_seed :
                    print(f"Enabling seed: {seed}")
                    kwargs["seed"] = seed
                
                # 生成原始音频数据
                wav = _model.generate(**kwargs)
                wav_list.append(wav)

                # 命名规则
                fname = out_path / f"{i:05d}_{tag}.wav"
                sf.write(str(fname), wav, sample_rate)

                # 每次生成音频，更新可预览音频
                default_preview = str(fname)

                if svc_convert and svc_model_exists(svc_model_name) :
                    m = [i for i in _sovits_models_available if i['model_name'] == svc_model_name][0]
                    try:
                        # SVC 转换音频
                        sr, converted_wav = sovits_convert_audio(
                            audio_filepath=str(fname),
                            model_name=svc_model_name,
                            model_path=m['model_path'],
                            speaker_path=m['speaker_path'],
                            device=device,
                        )
                        svc_wav_list.append(converted_wav)
                        svc_sample_rate = sr

                        # 命名加上前缀
                        svc_fname = out_path / f"svc_{i:05d}_{tag}.wav"
                        sf.write(str(svc_fname), converted_wav, sr)
                    except Exception as svc_err:
                        log_lines.append(f"⚠️ [{i}/{len(texts)}] → {fname.name} | SVC 转换失败: {svc_err}")
                        svc_wav_list.append(np.zeros(int(svc_sample_rate * 0.1), dtype=np.float32))
                    else :
                        default_preview = str(svc_fname)
                        log_lines.append(f"✅ [{i}/{len(texts)}] → {svc_fname.name}")

                else :
                    log_lines.append(f"✅ [{i}/{len(texts)}] → {fname.name}")

            except Exception as e:
                log_lines.append(f"❌ [{i}/{len(texts)}] 生成失败：{e}")
                wav_list.append(np.zeros(int(sample_rate * 0.1), dtype=np.float32))
            
            # 更新进度条
            progress(i / total_steps, desc=f"Processing: {i}/{total_steps}")

        # 合并音频文件
        merged_path = _merge_wavs(wav_list, sample_rate, out_path, tag, '', gap_seconds, merge_audio)
        svc_merged_path = _merge_wavs(svc_wav_list, svc_sample_rate, out_path, tag, 'svc_', gap_seconds, merge_audio)

    elif _current_model_type == 'OmniVoice' :
        sample_rate = 24000
        for i, text in enumerate(texts, start=1):
            try:
                kwargs = {"text": text, "ref_audio": ref_audio_path}
                if omnivoice_ref_text and omnivoice_ref_text.strip():
                    kwargs["ref_text"] = omnivoice_ref_text.strip()
                kwargs["num_step"] = omnivoice_num_step
                kwargs["speed"] = omnivoice_speed
                if omnivoice_use_duration :
                    kwargs["duration"]  = omnivoice_duration
                result = _omnivoice_model.generate(**kwargs)

                wav = result[0] if isinstance(result, (list, tuple)) else result
                fname = out_path / f"{i:05d}_{tag}.wav"
                sf.write(str(fname), wav, sample_rate)
                wav_list.append(wav)
                default_preview = str(fname)
                log_lines.append(f"✅ [{i}/{len(texts)}] → {fname.name}")
            except Exception as e:
                log_lines.append(f"❌ [{i}/{len(texts)}] 生成失败：{e}")
                wav_list.append(np.zeros(int(sample_rate * 0.1), dtype=np.float32))

            # 更新进度条
            progress(i / total_steps, desc=f"Processing: {i}/{total_steps}")
        
        # 合并音频文件
        merged_path = _merge_wavs(wav_list, sample_rate, out_path, tag, '', gap_seconds, merge_audio)

    log = "\n".join(log_lines)
    preview = svc_merged_path or merged_path or default_preview or None
    texts_json = json.dumps(texts, ensure_ascii=False)

    return preview, log, texts_json


def _merge_wavs(wav_list, sample_rate, out_path, tag, prefix, gap_seconds, merge_audio):
    if not merge_audio or not wav_list:
        return None
    silence = np.zeros(int(sample_rate * gap_seconds), dtype=np.float32)
    parts = []
    for idx, w in enumerate(wav_list):
        parts.append(w)
        if idx < len(wav_list) - 1:
            parts.append(silence)
    merged = np.concatenate(parts)
    merged_fname = out_path / f"{prefix}merged_{tag}.wav"
    sf.write(str(merged_fname), merged, sample_rate)
    return str(merged_fname)


# ─────────────────────────────────────────────
#  Share handler
# ─────────────────────────────────────────────
def handle_share(audio_path: str, texts_json: str, server_host: str):
    try:
        texts = json.loads(texts_json) if texts_json else []
    except Exception:
        texts = []

    if not texts:
        return "❌ 没有文本信息，请先生成音频", ""

    share_url, status = create_share_page(audio_path, texts, server_host)
    return status, share_url


# ─────────────────────────────────────────────
#  Gradio UI
# ─────────────────────────────────────────────
DARK_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=IBM+Plex+Mono:wght@300;400;500&display=swap');

:root {
    --bg:      #0d0f14;
    --panel:   #13161e;
    --border:  #252a38;
    --accent:  #5d8aff;
    --accent2: #b66dff;
    --green:   #3dffa0;
    --red:     #ff5e7a;
    --text:    #e8ecf5;
    --muted:   #697089;
    --radius:  10px;
}

body, .gradio-container {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
.vox-header {
    padding: 28px 0 20px;
    text-align: center;
    border-bottom: 1px solid var(--border);
    margin-bottom: 8px;
}
.vox-header h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800; font-size: 2.4rem; letter-spacing: -1px;
    background: linear-gradient(120deg, var(--accent), var(--accent2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0;
}
.vox-header p { color: var(--muted); font-size: .82rem; margin: 6px 0 0; }

.tab-nav { background: var(--panel) !important; border-bottom: 1px solid var(--border) !important; }
.tab-nav button { font-family: 'Syne', sans-serif !important; font-size: .9rem !important; color: var(--muted) !important; }
.tab-nav button.selected { color: var(--accent) !important; border-bottom: 2px solid var(--accent) !important; }

.gr-block, .gr-box, .panel { background: var(--panel) !important; border: 1px solid var(--border) !important; border-radius: var(--radius) !important; }

label span, .label-wrap span {
    font-family: 'Syne', sans-serif !important; font-size: .78rem !important;
    text-transform: uppercase !important; letter-spacing: .08em !important;
    color: var(--muted) !important;
}

input[type=text], input[type=number], textarea, select {
    background: #1a1e2b !important; border: 1px solid var(--border) !important;
    color: var(--text) !important; border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
input[type=text]:focus, textarea:focus { border-color: var(--accent) !important; outline: none !important; }
input[type=range] { accent-color: var(--accent) !important; }

button.primary {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    border: none !important; color: #fff !important;
    font-family: 'Syne', sans-serif !important; font-weight: 600 !important;
    font-size: .95rem !important; border-radius: 8px !important;
    letter-spacing: .05em !important;
    transition: opacity .2s, transform .15s !important;
}
button.primary:hover { opacity: .88 !important; transform: translateY(-1px) !important; }
button.secondary {
    background: var(--panel) !important; border: 1px solid var(--border) !important;
    color: var(--text) !important; font-family: 'Syne', sans-serif !important;
    border-radius: 8px !important;
}

#gen-log textarea {
    font-family: 'IBM Plex Mono', monospace !important; font-size: .78rem !important;
    background: #0a0c12 !important; color: var(--green) !important;
    border: 1px solid var(--border) !important; min-height: 160px !important;
}

#share-url textarea {
    font-family: 'IBM Plex Mono', monospace !important; font-size: .88rem !important;
    background: #0a0c12 !important; color: var(--accent2) !important;
    border: 1px solid var(--border) !important;
}

audio { filter: invert(1) hue-rotate(180deg); width: 100% !important; }

.section-title {
    font-family: 'Syne', sans-serif; font-size: .7rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: .15em; color: var(--accent);
    border-left: 3px solid var(--accent); padding-left: 10px;
    margin: 16px 0 8px;
}
"""

shortcut_js = """
<script>
function shortcuts(e) {
    var event = document.all ? window.event : e;
    switch (e.target.tagName.toLowerCase()) {
        case "not-a-input":
        case "not-a-textarea":
        break;
        default:
        if (e.key.toLowerCase() == "e" && e.ctrlKey) {
            document.getElementById("gen_btn").click();
        }
    }
}
document.addEventListener('keypress', shortcuts, false);
</script>
"""


def build_ui():
    with gr.Blocks(title="Voxbox") as gr_gui:

        json_texts_state = gr.State("[]")
        generated_texts_state = gr.State("[]")   # ← texts from last generation

        gr.HTML("""
        <div class="vox-header">
            <h1>⟡ Voxbox</h1>
            <p>批量语音合成 · 参考音频克隆 · 音频拼接</p>
        </div>
        """)

        with gr.Tabs():

            # ══════════════════════════════════
            #  Tab 1 – Model
            # ══════════════════════════════════
            with gr.Tab("🔧 模型加载"):
                gr.HTML('<div class="section-title">模型配置</div>')
                with gr.Row():
                    model_type = gr.Dropdown(
                        label="模型类型",
                        choices=["VoxCPM", "OmniVoice"],
                        value="VoxCPM",
                    )
                with gr.Row():
                    model_path = gr.Textbox(label="模型路径", value="openbmb/VoxCPM2",
                                            placeholder="本地目录或 HF repo")
                    device = gr.Dropdown(label="推理设备",
                                         choices=["cuda", "cpu", "mps", "auto"], value="cuda")

                gr.HTML('<div class="section-title" style="margin-top:16px;">高级选项</div>')
                with gr.Column(visible=True) as voxcpm_advanced:
                    voxcpm_load_denoiser = gr.Checkbox(label="启用降噪器 (load_denoiser)", value=False)
                with gr.Column(visible=False) as omnivoice_advanced:
                    omnivoice_device_map = gr.Dropdown(
                        label="设备映射 (device_map)",
                        choices=["cuda", "cpu", "mps", "xpu"],
                        value="cuda",
                    )
                    omnivoice_dtype = gr.Dropdown(
                        label="精度 (dtype)",
                        choices=["float16", "float32"],
                        value="float16",
                    )

                lora_enabled = gr.Checkbox(label="启用 LoRA", value=False)
                lora_list = scan_lora_checkpoints()
                lora_list_first = lora_list[0] if lora_list else 'NotAvailable'
                lora_select = gr.Dropdown(
                    label="",
                    choices=lora_list if lora_list else ['NotAvailable'],
                    value=lora_list_first,
                    visible=False,
                )

                def update_model_fields(model_type):
                    # outputs=[model_path, device, 
                             # omnivoice_advanced, voxcpm_advanced,
                             # voxcpm_params_row, omnivoice_params_row,
                             # omnivoice_ref_advanced, load_btn],
                    if model_type == "OmniVoice":
                        return (gr.update(value="NotAvailable"), gr.update(visible=False),
                                gr.update(visible=True), gr.update(visible=False),
                                gr.update(visible=False), gr.update(visible=True),
                                gr.update(visible=True), gr.update(interactive=False),
                                gr.update(visible=False), gr.update(visible=False))
                    elif model_type == 'VoxCPM' :
                        return (gr.update(value="openbmb/VoxCPM2"), gr.update(visible=True),
                                gr.update(visible=False), gr.update(visible=True),
                                gr.update(visible=True), gr.update(visible=False),
                                gr.update(visible=False), gr.update(interactive=True),
                                gr.update(visible=True), gr.update(visible=False))

                load_btn = gr.Button("⚡ 加载模型", variant="primary")
                model_status = gr.Textbox(label="状态", interactive=False)

                lora_enabled.change(
                    fn=lambda on: gr.update(visible=on),
                    inputs=[lora_enabled],
                    outputs=[lora_select],
                )

                load_btn.click(
                    load_model,
                    inputs=[model_type, model_path, device, voxcpm_load_denoiser, omnivoice_device_map, omnivoice_dtype, lora_enabled, lora_select],
                    outputs=[model_status],
                )

            # ══════════════════════════════════
            #  Tab 2 – Generate
            # ══════════════════════════════════
            with gr.Tab("🎙 批量生成"):

                with gr.Row(equal_height=False):
                    # Model settings
                    with gr.Column(scale=1):
                        gr.HTML('<div class="section-title">模型配置</div>')
                        with gr.Column(visible=True) as voxcpm_params_row :
                            cfg_value = gr.Slider(label="CFG Value",
                                                minimum=1.0, maximum=10.0,
                                                step=0.1, value=3.0)
                            inference_timesteps = gr.Slider(label="Inference Timesteps",
                                                            minimum=1, maximum=50,
                                                            step=1, value=10)
                            seed_enabled = gr.Checkbox(label="启用固定随机种子", value=False)
                            seed_value = gr.Number(
                                label="种子值", minimum=0, maximum=2147483647, step=1, value=42,
                                visible=False
                            )

                        with gr.Column(visible=False) as omnivoice_params_row:
                            num_step_slider = gr.Slider(label="Num Step",
                                                        minimum=1, maximum=64, step=1, value=32)
                            speed_slider    = gr.Slider(label="Speed",
                                                        minimum=0.5, maximum=2.0, step=0.05, value=0.9)

                        with gr.Accordion("⚙️ OmniVoice 高级选项", open=False, visible=False) as omnivoice_ref_advanced :
                            omnivoice_use_duration = gr.Checkbox(
                                label="使用时长控制语速（会覆盖 Speed 参数）",
                            )
                            duration_slider = gr.Slider(label="Duration (s)",
                                                        minimum=1.0, maximum=60.0, step=0.5, value=10.0, visible=False)
                            omnivoice_use_whisper = gr.Checkbox(
                                label="使用 Whisper 自动转录参考文本 (ref_text)",
                            )
                            omnivoice_ref_text_field = gr.Textbox(
                                label="参考文本 (ref_text，不使用 Whisper 时输入)",
                                placeholder="Reference transcription for voice cloning...",
                                #value=parse_reference_info(ref_audio_path.value)['text'],
                            )

                        gr.HTML('<div class="section-title">输出配置</div>')
                        output_dir = gr.Textbox(label="Output Path", value="output",
                                                placeholder="./output")
                        merge_audio = gr.Checkbox(label="Merge into one audio",
                                                  value=True)
                        gap_seconds = gr.Slider(label="音频间隔 (秒)",
                                                minimum=0.0, maximum=5.0,
                                                step=0.1, value=0.7)
                    
                    # Reference audio & voice
                    with gr.Column(scale=1):
                        gr.HTML('<div class="section-title">参考音频</div>')
                        ref_audio_list = load_reference_audio()
                        ref_audio_path = gr.Dropdown(
                            label="Reference Audio",
                            choices=ref_audio_list,
                            value=ref_audio_list[0],
                        )

                        ref_audio_refresh_list = gr.Button('Update Reference', variant='primary')
                        
                        # ── SVC Section (inside right column, before reference audio) ──
                        gr.HTML('<div class="section-title" style="margin-top:16px;">SVC 音色</div>')
                        
                        svc_convert_chk = gr.Checkbox(
                            label="Enable SVC",
                            value=False,
                        )
                        svc_model_section = gr.Row(visible=False)

                        with svc_model_section:
                            svc_model_list = gr.Dropdown(
                                label="SVC Model",
                                choices=[""],
                                value=None,
                            )

                        refresh_svc_btn = gr.Button("Update SVC Model", variant="primary")

                        def _refresh_svc_models():
                            global _sovits_models_available
                            _sovits_models_available = get_sovits_model_list()
                            choices = [m['model_name'] for m in _sovits_models_available] if _sovits_models_available else ["NotAvailable"]
                            default = choices[0] if (choices is not None) else None
                            return gr.update(choices=choices, value=default)
                    
                        def _refresh_ref_audio():
                            paths = load_reference_audio()
                            if not paths:
                                return gr.update(value=None)
                            return gr.update(choices=paths, value=paths[0])

                    with gr.Column(scale=1):
                        # TODO: Remove config_switch
                        gr.HTML('<div class="section-title">生成模式</div>', visible=False)
                        config_switch = gr.Radio(
                            label="生成模式",
                            choices=["批量生成", "单次生成"],
                            value="批量生成",
                            visible=False,
                        )
                        gr.HTML('<div class="section-title">文本输入</div>')
                        text_source = gr.Radio(
                            label="Text Source",
                            choices=["textarea", "json"],
                            value="textarea",
                            visible=False,
                        )
                        textarea_input = gr.Textbox(
                            label="One audio per line",
                            lines=12,
                            placeholder="Line 1\nLine 2\n...",
                        )
                        with gr.Group(visible=False) as json_group:
                            json_file = gr.File(label="上传 JSON 文本列表",
                                                file_types=[".json"])
                            json_preview = gr.Textbox(label="预览", lines=6,
                                                      interactive=False)
                            json_status = gr.Textbox(label="", interactive=False,
                                                     max_lines=1)

                        preview_audio = gr.Audio(label="Preview", type="filepath")

                gen_btn = gr.Button("🚀 Generate", variant="primary", elem_id="gen_btn")

                with gr.Row(equal_height=False):
                    with gr.Column(scale=2):
                        gr.HTML('<div class="section-title">生成日志</div>')
                        gen_log = gr.Textbox(label="", lines=8, interactive=False,
                                             elem_id="gen-log")

                # ── After generation: quick share shortcut ──
                gr.HTML('<div class="section-title" style="margin-top:20px;">快速分享</div>')
                with gr.Row():
                    quick_host = gr.Textbox(
                        label="服务器地址（留空使用默认）",
                        placeholder=f"http://your-server:{SHARE_SERVER_PORT}",
                        scale=3,
                    )
                    quick_share_btn = gr.Button("🔗 生成分享链接", variant="secondary", scale=1)

                quick_share_status = gr.Textbox(label="状态", interactive=False, max_lines=1)
                quick_share_url = gr.Textbox(
                    label="分享链接",
                    interactive=False,
                    placeholder="生成后链接显示在此",
                    elem_id="share-url",
                )

                # Signals & Events
                text_source.change(
                    lambda src: gr.update(visible=(src == "json")),
                    inputs=[text_source],
                    outputs=[json_group],
                )

                json_file.change(
                    load_json_file,
                    inputs=[json_file],
                    outputs=[json_preview, json_status, json_texts_state],
                )

                ref_audio_refresh_list.click(
                    fn=_refresh_ref_audio,
                    inputs=[],
                    outputs=[ref_audio_path],
                )

                ref_audio_path.change(
                    lambda src: gr.update(value=parse_reference_info(src)['text']) if src else gr.nothing(),
                    inputs=[ref_audio_path],
                    outputs=[omnivoice_ref_text_field],
                )

                seed_enabled.change(
                    lambda on: gr.update(visible=on),
                    inputs=[seed_enabled], outputs=[seed_value],
                )

                model_type.change(
                    update_model_fields,
                    inputs=[model_type],
                    outputs=[model_path, device, 
                                omnivoice_advanced, voxcpm_advanced,
                                voxcpm_params_row, omnivoice_params_row,
                                omnivoice_ref_advanced, load_btn,
                                lora_enabled, lora_select],
                )

                omnivoice_use_duration.change(
                    lambda src: gr.update(visible=src),
                    inputs=[omnivoice_use_duration],
                    outputs=[duration_slider],
                )

                omnivoice_use_whisper.change(
                    lambda src: gr.update(visible=not src),
                    inputs=[omnivoice_use_whisper],
                    outputs=[omnivoice_ref_text_field],
                )

                refresh_svc_btn.click(
                    _refresh_svc_models,
                    inputs=[],
                    outputs=[svc_model_list],
                )

                svc_convert_chk.change(
                    lambda on: gr.update(visible=on),
                    inputs=[svc_convert_chk],
                    outputs=[svc_model_section],
                )
                
                gen_btn.click(
                    generate_audio,
                    inputs=[
                        config_switch, text_source, textarea_input, json_texts_state,
                        ref_audio_path,
                        cfg_value, inference_timesteps, output_dir, gap_seconds, merge_audio,
                        omnivoice_ref_text_field, num_step_slider, speed_slider, omnivoice_use_duration, duration_slider, seed_enabled, seed_value,
                        svc_convert_chk, svc_model_list, device
                    ],
                    outputs=[preview_audio, gen_log, generated_texts_state],
                )

                quick_share_btn.click(
                    handle_share,
                    inputs=[preview_audio, generated_texts_state, quick_host],
                    outputs=[quick_share_status, quick_share_url],
                )

            # ══════════════════════════════════
            #  Tab 3 – Share Manager
            # ══════════════════════════════════
            with gr.Tab("🔗 分享管理"):
                gr.HTML('<div class="section-title">分享已生成的音频</div>')
                gr.HTML(f"""
                <div style="font-size:.8rem;color:#697089;line-height:1.9;margin-bottom:20px;">
                    在「批量生成」完成后，可以在这里指定任意音频文件路径来创建分享链接。<br>
                    分享服务运行于端口 <code style="color:#5d8aff">{SHARE_SERVER_PORT}</code>，
                    分享页面保存在 <code style="color:#5d8aff">share_pages/</code> 目录。<br>
                    收听者打开链接即可在浏览器中看到文本并播放音频，无需安装任何软件。
                </div>
                """)

                with gr.Row():
                    with gr.Column(scale=2):
                        gr.HTML('<div class="section-title">音频文件</div>')
                        share_audio_path = gr.Textbox(
                            label="音频文件路径",
                            placeholder="output/merged_cfg3.0_step10_20250101_120000.wav",
                        )
                        gr.HTML('<div class="section-title">文本内容</div>')
                        share_text_input = gr.Textbox(
                            label="文本（每行一条，多条自动编号）",
                            lines=5,
                            placeholder="输入生成该音频时使用的文本",
                        )

                    with gr.Column(scale=1):
                        gr.HTML('<div class="section-title">服务器地址</div>')
                        share_host_input = gr.Textbox(
                            label="Base URL",
                            placeholder=f"http://your-server:{SHARE_SERVER_PORT}",
                            value="",
                        )
                        gr.HTML(f"""
                        <div style="font-size:.72rem;color:#697089;line-height:1.7;margin-top:4px;">
                            留空则使用 <code style="color:#5d8aff">localhost:{SHARE_SERVER_PORT}</code><br>
                            填入公网地址即可分享给他人
                        </div>
                        """)
                        gr.HTML('<div class="section-title" style="margin-top:20px;">预览</div>')
                        share_manager_preview = gr.Audio(
                            label="音频预览",
                            type="filepath",
                            interactive=False,
                        )

                share_manager_btn = gr.Button("🔗 生成分享链接", variant="primary")
                share_manager_status = gr.Textbox(label="状态", interactive=False, max_lines=1)
                gr.HTML('<div class="section-title">分享链接</div>')
                share_manager_url = gr.Textbox(
                    label="",
                    interactive=False,
                    placeholder="点击按钮后，链接显示在此处",
                    elem_id="share-url",
                    lines=2,
                )

                # load audio preview when path changes
                share_audio_path.change(
                    lambda p: p if p and os.path.isfile(p) else None,
                    inputs=[share_audio_path],
                    outputs=[share_manager_preview],
                )

                def handle_share_manager(audio_path, text_input, server_host):
                    texts = parse_textarea(text_input)
                    if not texts:
                        return "❌ 请输入文本内容", ""
                    share_url, status = create_share_page(audio_path, texts, server_host)
                    return status, share_url

                share_manager_btn.click(
                    handle_share_manager,
                    inputs=[share_audio_path, share_text_input, share_host_input],
                    outputs=[share_manager_status, share_manager_url],
                )

        def fill_in_default_values(
            input_ref_audio_path,
        ) :
            text_to_fill = parse_reference_info(input_ref_audio_path)['text']

            # FIXME: Cannot enable svc without manually updating list
            # sovits_model_list = get_sovits_model_list()
            # sovits_model_choices = [m['model_name'] for m in sovits_model_list] if sovits_model_list else ["NotAvailable"]
            # sovits_model_default = sovits_model_choices[0] if sovits_model_choices else None

            return (
                gr.update(value=text_to_fill),
            )

        gr_gui.load(
            fn=fill_in_default_values,
            inputs=[
                ref_audio_path,
            ],
            outputs=[
                omnivoice_ref_text_field,
            ],
            # js=shortcut_js,
        )

    return gr_gui


if __name__ == "__main__":
    start_share_server()
    gr_gui = build_ui()
    gr_gui.launch(css=DARK_CSS, share=False, server_name="0.0.0.0", server_port=17865, inbrowser=True)
