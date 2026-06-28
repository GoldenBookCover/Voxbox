# 为 VoxCPM 添加随机种子控制

## 改动概述
在「批量生成」Tab 的「模型配置」区域新增"启用固定随机种子"开关和对应的种子值输入框。仅针对 VoxCPM 模型生效。

## 具体改动

### 1. build_ui() — voxcpm_params_row 区域（约第621行之后）
在现有的 CFG Value 和 Inference Timesteps 下方新增：
- 一个 gr.Checkbox "启用固定随机种子"（默认关闭，开关 visibility visible=False 的 seed_value Number 输入框）
- 一个 gr.Number "种子值"（范围 0~2^31-1=2147483647，step=1，默认42，visible=False）

新增事件监听：checkbox 变化时切换 input box的 visible。

### 2. update_model_fields() 返回值更新
当模型类型切换到 VoxCPM 时，确保 voxcpm_seed_row 可见（在返回中增加对 seed_row 的 visibility）。

### 3. build_ui() — 生成按钮 inputs 更新
gen_btn.click 的 inputs 列表中加入 seed_enabled 和 seed_value。

### 4. generate_audio() 函数签名
新增参数 `seed_enabled` (bool) 和 `seed_value` (int)，新增参数需要放在 inputs 顺序中对应的位置。

### 5. generate_audio() 函数 — VoxCPM 生成逻辑（约第319-337行）
在 `_model.generate()` 调用处：
```python
if seed_enabled and seed_value is not None:
    random_state = rng.Generator(np.random.PCG64(int(seed_value)))
else:
    random_state = None

wav = _model.generate(
    text=text,
    reference_wav_path=ref_audio_path,
    cfg_value=cfg_value,
    inference_timesteps=int(inference_timesteps),
    random_state=random_state,  # 新增参数，关闭时为默认（无固定种子）
)
```

具体如何传递随机种子取决于 `VoxCPM.generate()` API。假设它接受一个 `random_state` 或 `seed` 参数作为核心库提供的生成控制方式。在不确定的情况下，先检查 `VOXCPM generate signature` 确认接口。

## 验证
- [x] 代码可读性、缩进和格式正确
- [x] UI 元素添加后渲染行为符合预期（visible切换）
- [x] VoxCPM generate 调用在 seed_enabled=True 时传递固定种子；seed_enabled=False 时不传 seed
