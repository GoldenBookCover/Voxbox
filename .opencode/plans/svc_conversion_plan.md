# SVC 转换功能新增计划 (Final)

## 涉及文件

- `src/main.py` — UI 和核心逻辑（主工作区）
- `src/utils.py` — **已实现完整逻辑**，无需修改

---

## 关键技术洞察

查看 `utils.py:124-234`，`sovits_convert_audio()` 函数内部已完成：
1. 加载 SoVITS 模型（第189-193行 SynthesizerInfer.load_svc_model）
2. 处理 whisper, hubert, pitch 等中间处理
3. 执行 SVC 推理并返回原始音频

**因此不需要 `_active_svc_model` 预缓存机制**。只需验证文件路径即可。

---

## 1. src/main.py — SVC 音色模型管理函数

### 1.1 新增 `svc_model_exists(model_name: str)` 函数（放在 load_svc_model 之后）

```python
def svc_model_exists(model_name: str) -> bool:
    """检查指定 SVC 模型的必要文件是否存在"""
    if not model_name or model_name == "无可用模型":
        return False
    
    for m in _sovits_models_available:
        if m['model_name'] == model_name:
            return (m['model_path'].exists() and m['speaker_path'].exists())
    return False
```


---

## 2. src/main.py — `generate_audio` 函数修改

### 2.1 新增参数（第273-280行左右）

在现有签名末尾添加两个新参数，**并确保 `progress=gr.Progress()` 始终作为函数的最后一个参数**：

```python
def generate_audio(
    # ... all existing params ...
    svc_convert: bool,                  # ← NEW
    svc_model_name: str,                # ← NEW
    progress=gr.Progress(),             # ← 必须保持在最后
):
```

### 2.2 VoxCPM for 循环内修改（约第328行）

在 `wav = _model.generate(**kwargs)` 之后写入 wav_list/individual_paths **前后**，加入 SVC 转换步骤：

```python
for i, text in enumerate(texts, start=1):
    try:
        kwargs = { ... }
        if use_seed:
            kwargs["seed"] = seed
        wav = _model.generate(**kwargs)
        
        # ← SVC conversion inside the loop (方案 A)
        if svc_convert and svc_model_name and svc_model_name != "无可用模型":
            for m in _sovits_models_available:
                if m['model_name'] == svc_model_name:
                    try:
                        out_pfx = out_path / f"svc_{i:05d}_{tag}.wav"
                        sr, converted_wav = utils.sovits_convert_audio(
                            audio_filepath=str(out_path / f"{i:05d}_{tag}.wav"),
                            model_name=svc_model_name,
                            model_path=m['model_path'],
                            speaker_path=m['speaker_path'],
                        )
                        sf.write(str(out_pfx), converted_wav, sr)
                        individual_paths[i-1] = str(out_pfx)  # replace path
                        log_lines[i-1] += " → SVC转换完成"
                        wav = converted_wav
                    except Exception as svc_err:
                        log_lines[i-1] += f" [SVC失败: {svc_err}]"
                    break
        
        fname = out_path / f"{i:05d}_{tag}.wav"
        sf.write(str(fname), wav, sample_rate)  # write base wav (may be unchanged if svc_convert=False)
```

**进度管理：** 
- `for` 循环中已有 `progress(i / total_steps, ...)` — SVC转换完成后的本次迭代，进度条正常更新
- SVC耗时不影响主进度逻辑（与音频生成共用一个进度点）

OmniVoice 分支不需要修改。

---

## 3. src/main.py — UI 修改（`build_ui()`）

### 3.1 "输出配置"之后添加 SVC 开关

在 `gap_seconds` slider 之后（约第697行后，`with gr.Column(scale=2):` 内的内容末尾），添加：

```python
svc_convert = gr.Checkbox(
    label="开启 SVC 转换",
    value=False,
)
```

### 3.2 "参考音频"之后添加 SVC 列（放在 ref_audio_refresh_list 之前）

在 `with gr.Column(scale=1):`（ref_audio 所在的 Column，约第698行）内，`ref_audio_refresh_list` **之前**：

```python
# ── SVC Section ──
gr.HTML('<div class="section-title">SVC 音色</div>')
svc_section = gr.Column(visible=False)

with svc_section:
    with gr.Row():
        svc_model_dropdown = gr.Dropdown(
            label="选择模型",
            choices=["无可用模型"],
            value=None,
        )
        load_svc_btn = gr.Button("加载音色", variant="primary")
    
    svc_validation_status = gr.Textbox(label="状态", interactive=False)

# 显示/隐藏 SVC section
svc_convert.change(
    lambda on: gr.update(visible=on),
    inputs=[svc_convert],
    outputs=[svc_section],
)
```

### 3.3 `_sovits_models_available` 刷新逻辑

**建议采用方案：在 `svc_section` 添加一个独立的"刷新模型列表"按钮，以保持原有点击事件的职责分离。**

```python
# ── SVC Section ── (inside svc_section:)
with gr.Row():
    refresh_svc_btn = gr.Button("🔄 刷新模型列表", variant="secondary")
    
refresh_svc_btn.click(
    fn=refresh_models_and_return,
    inputs=[svc_model_dropdown],
    outputs=[svc_model_dropdown, svc_validation_status],
)

def refresh_models_and_return(dropdown):
    models = get_sovits_model_list()
    choices = [m['model_name'] for m in models] if models else ["无可用模型"]
    default = choices[0] if (choices and dropdown is not None) else None
    return gr.update(choices=choices, value=default), "已刷新"
```

### 3.4 `gen_btn.click()` inputs 增加参数

修改约第769行的 `gen_btn.click()`：

```python
gen_btn.click(
    generate_audio,
    inputs=[
        config_switch, text_source, textarea_input, json_texts_state,
        ref_audio_path,
        cfg_value, inference_timesteps, output_dir, gap_seconds, merge_audio,
        omnivoice_ref_text_field, num_step_slider, speed_slider, omnivoice_use_duration, duration_slider, seed_enabled, seed_value,
        svc_convert,       # ← NEW
        svc_model_dropdown,# ← NEW  
    ],
    outputs=[preview_audio, gen_log, generated_texts_state],
)
```

---

## 4. 完整的文件修改点汇总

| 步骤 | 位置/行号范围 | 修改内容 | 类型 |
|------|--------------|---------|------|
| 1 | ~第301-358行区域 | `generate_audio()` 增加 `svc_convert`, `svc_model_name` 参数 | 修改签名 |
| 2 | ~第328行（VoxCPM for 循环内） | SVC转换代码块 + progress同步更新 | 新增代码 |
| 5 | 新函数 | `svc_model_exists()` + `refresh_models_and_return()` | 新增函数 |
| 6 | ~第702行（输出配置/gap_seconds之后） | `svc_convert` Checkbox | 新增 UI |
| 7 | ~第704行（参考音频Column内、刷新按钮之前） | SVC下拉+加载按钮Section (visible=False) | 新增 UI |
| 8 | svc_section 中 | "刷新模型列表" + "加载音色"按钮 | 新增 UI+事件 |
| 9 | ~第769行 `ref_audio_refresh_list` 附近 | refresh_svc_btn callback | 绑定事件 |
| 10 | ~第813行 `gen_btn.click()` inputs | 增加 `svc_convert`, `svc_model_dropdown` | 修改调用 |

---

## 5. 注意事项

1. **SVC 转换输出的文件命名**：`svc_原文件名.wav`，不破坏原始音频
2. **进度条**：每个文本在生成 + SVC 转换后统一更新 progress（与方案 A 对齐）
3. **无可用模型**：下拉显示 "无可用模型"，SVC不执行
4. **SVC 转换失败**：记录日志，继续处理下一条文本（不中断批量生成）
