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

os.environ['TRITON_PTXAS_PATH'] = '/usr/bin/ptxas'

#  Share pages directory & static server
SHARE_DIR = Path("share_pages")
SHARE_DIR.mkdir(exist_ok=True)

SHARE_SERVER_PORT = 17866  # separate port for static share pages


def start_share_server():
    """Serve share_pages/ directory on SHARE_SERVER_PORT in a background thread."""
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(SHARE_DIR), **kwargs)

        def log_message(self, format, *args):
            pass  # silence access logs

    def run():
        server = HTTPServer(("0.0.0.0", SHARE_SERVER_PORT), Handler)
        server.serve_forever()

    t = threading.Thread(target=run, daemon=True)
    t.start()


#  Lazy model holder
_model = None


def load_model(model_path: str, load_denoiser: bool, device: str) -> str:
    global _model
    try:
        from voxcpm import VoxCPM
        _model = VoxCPM.from_pretrained(
            model_path,
            load_denoiser=load_denoiser,
            device=device,
        )
        return f"✅ 模型加载成功：{model_path}  |  设备：{device}  |  降噪器：{'开启' if load_denoiser else '关闭'}"
    except Exception as e:
        _model = None
        return f"❌ 模型加载失败：{e}"


#  Text-list helpers
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
    text_source: str,
    textarea_text: str,
    json_texts_state: str,
    text_normalize: bool,
    ref_audio_path: str,
    cfg_value: float,
    inference_timesteps: int,
    output_dir: str,
    gap_seconds: float,
    merge_audio: bool,
    progress=gr.Progress(track_tqdm=True),
):
    if _model is None:
        return None, "❌ 请先加载模型", "[]"

    if text_source == "textarea":
        texts = parse_textarea(textarea_text)
    else:
        try:
            texts = json.loads(json_texts_state) if json_texts_state else []
        except Exception:
            texts = []

    if not texts:
        return None, "❌ 文本列表为空，请输入或上传文本", "[]"

    if not ref_audio_path or not os.path.isfile(ref_audio_path):
        return None, f"❌ 参考音频路径无效：{ref_audio_path}", "[]"

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"cfg{cfg_value}_step{inference_timesteps}_{ts}"

    sample_rate = _model.tts_model.sample_rate
    wav_list = []
    individual_paths = []
    log_lines = []

    for i, text in enumerate(texts, start=1):
        try:
            wav = _model.generate(
                text=text,
                reference_wav_path=ref_audio_path,
                cfg_value=cfg_value,
                inference_timesteps=int(inference_timesteps),
                normalize=text_normalize,
            )
            fname = out_path / f"{i:03d}_{tag}.wav"
            sf.write(str(fname), wav, sample_rate)
            wav_list.append(wav)
            individual_paths.append(str(fname))
            log_lines.append(f"✅ [{i}/{len(texts)}] → {fname.name}")
        except Exception as e:
            log_lines.append(f"❌ [{i}/{len(texts)}] 生成失败：{e}")
            wav_list.append(np.zeros(int(sample_rate * 0.1), dtype=np.float32))

    merged_path = None
    if merge_audio and wav_list:
        silence = np.zeros(int(sample_rate * gap_seconds), dtype=np.float32)
        parts = []
        for idx, w in enumerate(wav_list):
            parts.append(w)
            if idx < len(wav_list) - 1:
                parts.append(silence)
        merged = np.concatenate(parts)
        merged_fname = out_path / f"merged_{tag}.wav"
        sf.write(str(merged_fname), merged, sample_rate)
        merged_path = str(merged_fname)
        log_lines.append(f"\n🎵 合并音频已保存：{merged_fname.name}  (间隔 {gap_seconds}s)")

    log = "\n".join(log_lines)
    preview = merged_path if merged_path else (individual_paths[-1] if individual_paths else None)
    texts_json = json.dumps(texts, ensure_ascii=False)

    return preview, log, texts_json


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


def build_ui():
    with gr.Blocks(css=DARK_CSS, title="Voxbox") as demo:

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
                    model_path = gr.Textbox(label="模型路径", value="openbmb/VoxCPM2",
                                            placeholder="本地目录或 HF repo")
                    device = gr.Dropdown(label="推理设备",
                                         choices=["cuda", "cpu", "mps", "auto"], value="cuda")
                load_denoiser = gr.Checkbox(label="启用降噪器 (load_denoiser)", value=False)
                load_btn = gr.Button("⚡ 加载模型", variant="primary")
                model_status = gr.Textbox(label="状态", interactive=False)

                load_btn.click(
                    load_model,
                    inputs=[model_path, load_denoiser, device],
                    outputs=[model_status],
                )

            # ══════════════════════════════════
            #  Tab 2 – Generate
            # ══════════════════════════════════
            with gr.Tab("🎙 批量生成"):

                with gr.Row(equal_height=False):

                    with gr.Column(scale=1):
                        gr.HTML('<div class="section-title">文本输入</div>')
                        text_source = gr.Radio(
                            label="文本来源",
                            choices=["textarea", "json"],
                            value="textarea",
                        )
                        textarea_input = gr.Textbox(
                            label="手动输入（每行一条）",
                            lines=8,
                            placeholder="第一条文本\n第二条文本\n...",
                        )
                        with gr.Group(visible=False) as json_group:
                            json_file = gr.File(label="上传 JSON 文本列表",
                                                file_types=[".json"])
                            json_preview = gr.Textbox(label="预览", lines=6,
                                                      interactive=False)
                            json_status = gr.Textbox(label="", interactive=False,
                                                     max_lines=1)

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

                        text_normalize = gr.Checkbox(label="文本标准化（自动转换数字等）",
                                                  value=False)
                        gr.HTML('<div class="section-title">参考音频</div>')
                        ref_audio_path = gr.Textbox(
                            label="参考音频路径", value="reference_audio/ref-2.wav",
                            placeholder="/path/to/reference.wav",
                        )

                    with gr.Column(scale=1):
                        gr.HTML('<div class="section-title">生成参数</div>')
                        cfg_value = gr.Slider(label="CFG Value",
                                              minimum=1.0, maximum=10.0,
                                              step=0.5, value=3.0)
                        inference_timesteps = gr.Slider(label="Inference Timesteps",
                                                        minimum=1, maximum=50,
                                                        step=1, value=10)

                        gr.HTML('<div class="section-title">输出配置</div>')
                        output_dir = gr.Textbox(label="输出目录", value="output",
                                                placeholder="./output")
                        merge_audio = gr.Checkbox(label="拼接所有音频为一个完整音频",
                                                  value=True)
                        gap_seconds = gr.Slider(label="音频间隔 (秒)",
                                                minimum=0.0, maximum=5.0,
                                                step=0.1, value=0.5)

                        gr.HTML('<div class="section-title">文件命名示例</div>')
                        gr.HTML("""
                        <div style="font-size:.75rem;color:#697089;line-height:1.8;
                                    font-family:'IBM Plex Mono',monospace;">
                            单条：<code style="color:#5d8aff">001_cfg3.0_step10_20250101_120000.wav</code><br>
                            合并：<code style="color:#b66dff">merged_cfg3.0_step10_20250101_120000.wav</code>
                        </div>
                        """)

                gen_btn = gr.Button("🚀 开始批量生成", variant="primary")

                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        gr.HTML('<div class="section-title">预览音频</div>')
                        preview_audio = gr.Audio(label="最终音频预览", type="filepath")

                    with gr.Column(scale=1):
                        gr.HTML('<div class="section-title">生成日志</div>')
                        gen_log = gr.Textbox(label="", lines=8, interactive=False,
                                             elem_id="gen-log")

                # ── After generation: quick share shortcut ──
                # gr.HTML('<div class="section-title" style="margin-top:20px;">快速分享</div>')
                # with gr.Row():
                #     quick_host = gr.Textbox(
                #         label="服务器地址（留空使用默认）",
                #         placeholder=f"http://your-server:{SHARE_SERVER_PORT}",
                #         scale=3,
                #     )
                #     # quick_share_btn = gr.Button("🔗 生成分享链接", variant="secondary", scale=1)

                # quick_share_status = gr.Textbox(label="状态", interactive=False, max_lines=1)
                # quick_share_url = gr.Textbox(
                #     label="分享链接",
                #     interactive=False,
                #     placeholder="生成后链接显示在此",
                #     elem_id="share-url",
                # )

                gen_btn.click(
                    generate_audio,
                    inputs=[
                        text_source, textarea_input, json_texts_state,
                        text_normalize, ref_audio_path,
                        cfg_value, inference_timesteps,
                        output_dir, gap_seconds, merge_audio,
                    ],
                    outputs=[preview_audio, gen_log, generated_texts_state],
                )

                # quick_share_btn.click(
                #     handle_share,
                #     inputs=[preview_audio, generated_texts_state, quick_host],
                #     outputs=[quick_share_status, quick_share_url],
                # )

            # ══════════════════════════════════
            #  Tab 3 – Share Manager
            # ══════════════════════════════════
            # with gr.Tab("🔗 分享管理"):
            #     gr.HTML('<div class="section-title">分享已生成的音频</div>')
            #     gr.HTML(f"""
            #     <div style="font-size:.8rem;color:#697089;line-height:1.9;margin-bottom:20px;">
            #         在「批量生成」完成后，可以在这里指定任意音频文件路径来创建分享链接。<br>
            #         分享服务运行于端口 <code style="color:#5d8aff">{SHARE_SERVER_PORT}</code>，
            #         分享页面保存在 <code style="color:#5d8aff">share_pages/</code> 目录。<br>
            #         收听者打开链接即可在浏览器中看到文本并播放音频，无需安装任何软件。
            #     </div>
            #     """)

            #     with gr.Row():
            #         with gr.Column(scale=2):
            #             gr.HTML('<div class="section-title">音频文件</div>')
            #             share_audio_path = gr.Textbox(
            #                 label="音频文件路径",
            #                 placeholder="output/merged_cfg3.0_step10_20250101_120000.wav",
            #             )
            #             gr.HTML('<div class="section-title">文本内容</div>')
            #             share_text_input = gr.Textbox(
            #                 label="文本（每行一条，多条自动编号）",
            #                 lines=5,
            #                 placeholder="输入生成该音频时使用的文本",
            #             )

            #         with gr.Column(scale=1):
            #             gr.HTML('<div class="section-title">服务器地址</div>')
            #             share_host_input = gr.Textbox(
            #                 label="Base URL",
            #                 placeholder=f"http://your-server:{SHARE_SERVER_PORT}",
            #                 value="",
            #             )
            #             gr.HTML(f"""
            #             <div style="font-size:.72rem;color:#697089;line-height:1.7;margin-top:4px;">
            #                 留空则使用 <code style="color:#5d8aff">localhost:{SHARE_SERVER_PORT}</code><br>
            #                 填入公网地址即可分享给他人
            #             </div>
            #             """)
            #             gr.HTML('<div class="section-title" style="margin-top:20px;">预览</div>')
            #             share_manager_preview = gr.Audio(
            #                 label="音频预览",
            #                 type="filepath",
            #                 interactive=False,
            #             )

            #     share_manager_btn = gr.Button("🔗 生成分享链接", variant="primary")
            #     share_manager_status = gr.Textbox(label="状态", interactive=False, max_lines=1)
            #     gr.HTML('<div class="section-title">分享链接</div>')
            #     share_manager_url = gr.Textbox(
            #         label="",
            #         interactive=False,
            #         placeholder="点击按钮后，链接显示在此处",
            #         elem_id="share-url",
            #         lines=2,
            #     )

            #     # load audio preview when path changes
            #     share_audio_path.change(
            #         lambda p: p if p and os.path.isfile(p) else None,
            #         inputs=[share_audio_path],
            #         outputs=[share_manager_preview],
            #     )

            #     def handle_share_manager(audio_path, text_input, server_host):
            #         texts = parse_textarea(text_input)
            #         if not texts:
            #             return "❌ 请输入文本内容", ""
            #         share_url, status = create_share_page(audio_path, texts, server_host)
            #         return status, share_url

            #     share_manager_btn.click(
            #         handle_share_manager,
            #         inputs=[share_audio_path, share_text_input, share_host_input],
            #         outputs=[share_manager_status, share_manager_url],
            #     )

    return demo


if __name__ == "__main__":
    # start_share_server()
    demo = build_ui()
    demo.launch(share=False, server_name="0.0.0.0", server_port=17865, inbrowser=True)
