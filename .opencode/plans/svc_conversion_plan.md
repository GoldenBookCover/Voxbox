# SVC 转换功能新增计划

## 涉及文件

- `src/main.py` — UI 和核心逻辑（主工作区）
- `src/utils.py` — 已有 `get_sovits_model_list` 和 `sovits_convert_audio`，无需修改

---

## 1. src/main.py — 全局状态

### 1.1 新增 SVC 相关全局变量（放在第 50 行 `_current_model_type` 之后）

```python
# SVC model state
_sovits_models = []          # get_sovits_model_list() 返回的模型列表
_active_svc_model = None     # 当前加载的 SovitsInferenceConfig


class _SovitsModelState:
    """SVC 模型加载状态"""
    def __init__(self, config, model_name):
        self.config = config
        self.model_name = model_name
```

---

## 2. src/main.py — SVC 模型管理函数

### 2.1 新增 `unload_svc_model()` 函数（放在全局变量之后）

```python
def unload_svc_model():
    """卸载当前 SVC 模型"""
    global _active_svc_model
    if _active_svc_model is not None:
        print(f"Unload SVC model: {_active_svc_model.model_name}")
        del _active_svc_model
        import gc; gc.collect()
        torch.cuda.empty_cache()
        _active_svc_model = None
    return "✅ 模型已卸载"
```

### 2.2 新增 `load_svc_model()` 函数

```python
def load_svc_model(model_name: str):
    """加载选中的 SVC 音色模型，并卸载之前的旧模型"""
    global _active_svc_model
    
    if not model_name:
        return "❌ 未选择模型", gr.update(choices=[])
    
    if not _sovits_models:
        return "❌ 无可用的 SVC 模型，请先下载模型文件", gr.update(choices=[])
    
    # 找到对应的模型信息
    chosen = None
    for m in _sovits_models:
        if m['model_name'] == model_name:
            chosen = m
            break
    
    if not chosen:
        return f"❌ 未找到模型: {model_name}", gr.update(choices=[m['model_name'] for m in _sovits_models])
    
    # 卸载旧模型
    unload_svc_model()
    
    # 加载新模型
    from utils import SovitsInferenceConfig
    
    config = SovitsInferenceConfig(
        config=utils.SOVITS_DEFAULT_CONFIG,  # base.yaml
        model=chosen['model_path'],
        spk=chosen['speaker_path'],
        voice=model_name,
    )
    
    _active_svc_model = config
    return f"✅ 已加载音色: {model_name}（模型: {chosen['model_path'].name}）", gr.update(value=model_name)
```

### 2.3 修改 `_sovits_models` 的获取时机

在 `build_ui()` 中调用 `load_reference_audio()` 的同级位置，添加初始模型列表加载：

```python
# 在 build_ui() 开头部分，ref_audio_list = load_reference_audio() 附近
_sovits_models = utils.get_sovits_model_list()
svc_model_choices = [m['model_name'] for m in _sovits_models] if _sovits_models else ["无可用模型"]
```

---

## 3. src/main.py — `generate_audio` 函数修改

### 3.1 新增 svc_convert 参数

在 `generate_audio()` 函数签名中增加：

```python
def generate_audio(
    # ... existing params ...
    svc_convert: bool,                          # ← 新增
    svc_model_name: str,                        # ← 新增
):
```

### 3.2 在生成 wav 后添加 SVC 转换逻辑

在每个模型类型的 for 循环 **之后**（或写入文件之前的合并之前），当 `svc_convert=True` 时：

```python
if svc_convert and _active_svc_model is not None:
    # 遍历 individual_paths，对每个 wav 进行转换
    converted_paths = []
    new_wav_list = []
    
    for idx, fpath in enumerate(individual_paths):
        try:
            out_pfx = out_path / f"svc_{Path(fpath).name}"
            sample_rate, audio = utils.sovits_convert_audio(
                audio_filepath=fpath,
                model_name=_active_svc_model.voice,
                model_path=_active_svc_model.model,
                speaker_path=_active_svc_model.spk,
            )
            sf.write(str(out_pfx), audio.dtype, sample_rate)
            converted_paths.append(str(out_pfx))
            individual_paths[idx] = str(out_pfx)  # 替换路径
            log_lines[idx] = log_lines[idx].replace(
                Path(fpath).name, f"svc_{Path(fpath).name}"
            )
        except Exception as e:
            log_lines.append(f"❌ SVC转换失败[{idx}]: {e}")
```

同时修改返回值中的 `merged_path` 处理——拼接时用更新后的 `individual_paths`。

最后把 `wav_list` 中所有元素重置为原始文件路径（因为 svc 输出是新的音频文件）。

### 3.3 `generate_audio` 返回类型调整

新增两个返回值：
1. `svc_model_loaded: str` — 是否已加载可用的 SVC 模型状态
2. `updated_svc_choices` — gr.update()，用于刷新下拉列表（因为加载了新版图后刷新）

实际更简洁的方案是只增加返回值到 UI 的输出组件：
```python
return preview, log, texts_json, gr.update(choices=[m['model_name'] for m in _sovits_models])
```

---

## 4. src/main.py — UI 修改（`build_ui()`）

### 4.1 "输出配置"之后添加 SVC 开关

在 `gap_seconds` slider 之后，放在同一个 Column 或单独一行：

```python
svc_convert = gr.Checkbox(
    label="开启 Svc 转换",
    value=False,
)

# 用于联动显示的 State
_sovits_models_state = gr.State([])   # ← 在函数外/文件全局设置，或作为 build_ui() 内的变量
```

### 4.2 "参考音频"之后添加 SVC 下拉列表 + 加载按钮

在 `ref_audio_refresh_list` 和预览区域之间添加：

```python
# Svc section (conditional)
svc_section = gr.Column(visible=False)   # ← 用 visible 控制显隐

with svc_section:
    gr.HTML('<div class="section-title">Svc 音色</div>')
    
    with gr.Row():
        # 下拉菜单显示模型列表，默认选第一个
        svc_model_dropdown = gr.Dropdown(
            label="Select model",
            choices=["无可用模型"],
            value=None,
            allow_custom_value=False,
        )
        
        load_svc_btn = gr.Button("加载音色", variant='primary')
    
    # 状态提示
    svc_status = gr.Textbox(label="状态", interactive=False)

# svc_convert switch → visibility toggle for grpcoup
svc_convert.change(
    lambda on: gr.update(visible=on),
    inputs=[svc_convert],
    outputs=[svc_section],
)
```

### 4.3 按钮联动

```python
load_svc_btn.click(
    load_svc_model,
    inputs=[svc_model_dropdown],
    outputs=[svc_status, svc_model_dropdown],   # refresh dropdown
)
```

### 4.4 `generate_audio` 调用增加参数

修改 `gen_btn.click()` 的 inputs：

```python
gen_btn.click(
    generate_audio,
    inputs=[
        config_switch, text_source, textarea_input, json_texts_state,
        ref_audio_path,
        cfg_value, inference_timesteps, output_dir, gap_seconds, merge_audio,
        omnivoice_ref_text_field, num_step_slider, speed_slider, omnivoice_use_duration, duration_slider, seed_enabled, seed_value,
        svc_convert,                          # ← 新增
    ],
    outputs=[preview_audio, gen_log, generated_texts_state],
)
```

### 4.5 初始刷新模型列表

在 `build_ui()` 末尾的 `demo.load()` 部分添加：

```python
# 页面加载时获取模型列表
def init_svc_models(svc_on):
    if not svc_on:   # if svc_convert initially False → no need to load
        return gr.update(choices=["无可用模型"], value=None), gr.Column(visible=False)
    
    models = utils.get_sovits_model_list()
    choices = [m['model_name'] for m in models] if models else ["无可用模型"]
    default = choices[0] if choices else None
    
    return (
        gr.update(choices=choices, value=default),
        gr.Column(visible=True),
    )

# 在 demo.load 部分追加一个 fn
def refresh_svc_models(svc_on, current_value):
    models = utils.get_sovits_model_list()
    choices = [m['model_name'] for m in models] if models else ["无可用模型"]
    default = choices[0] if (choices and not current_value) else current_value
    return gr.update(choices=choices, value=default)

demo.load(
    fn=refresh_svc_models,
    inputs=[svc_convert, svc_model_dropdown],  # ← 如果下拉存在的话
    outputs=[svc_model_dropdown],
)
```

---

## 5. 完整的文件修改点汇总

| 行号范围 | 修改内容 | 类型 |
|----------|---------|------|
| ~51 行后 | 添加 `_sovits_models = []`, `_active_svc_model = None` 全局变量 | 新增代码 |
| 新增函数 | `load_svc_model(model_name)` | 新增函数 |
| 新增函数 | `unload_svc_model()` | 新增函数 |
| ~267 行 | `generate_audio()` 增加 `svc_convert`, `svc_model_name` 参数 | 修改签名 |
| ~345/368 行（VoxCPM/OmniVoice for loop 后） | 添加 SVC 转换条件分支 | 新增代码 |
| ~692 行 | "输出配置"之后添加 `svc_convert` Checkbox | 新增 UI 组件 |
| ~695-718 行 | "参考音频"之后添加 SVC 下拉 + Load 按钮（封装在 Column(visible=False)） | 新增 UI 组件 |
| ~720 行后 | `svc_convert.change()` 控制 Column 显隐 | 绑定事件 |
| ~769 行 | `gen_btn.click()` inputs 增加 `svc_convert` | 修改调用 |
| ~848 行 | `demo.load()` 增加模型列表刷新 | 新增/修改 |

---

## 6. 注意事项

1. **SVC 转换输出的文件命名**：在原有文件名加前缀 `svc_`，即 `00001_cfg3_step10.wav` → `svc_00001_cfg3_step10.wav`
2. **SVC 仅针对 wav 文件生效**，不对合并的 merged 音频做转换（用户可单独查看生成的 svc wav）
3. **切换模型时**：`load_svc_model` 内先调用 `unload_svc_model()` 释放内存
4. **无可用模型时的处理**：下拉列表显示 "无可用模型"，加载按钮提示不可用
5. **SVC 转换失败**：不中断整体流程，记录日志到 gen_log
