# ruff: noqa: E402
# Above allows ruff to ignore E402: module level import not at top of file

import re
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any
from dataclasses import dataclass
from importlib import resources

import numpy as np
import torch
import yaml
from pydownloader import Downloader

from config import (
    BASE_DIR,
    MODEL_PATH,
)

# sovits
from sovits_svc import svc_inference

from sovits_svc.hubert.inference import hubert_infer
from sovits_svc.pitch.inference import pitch_infer
from sovits_svc.whisper.inference import whisper_infer
from sovits_svc.svc_ensure_models import ensure_models

from omegaconf import OmegaConf
# from huggingface_hub import snapshot_download
from sovits_svc.vits.models import SynthesizerInfer
from sovits_svc.pitch import load_csv_pitch


custom_ema_model, pre_custom_path = None, ""

SOVITS_MODEL_PATH = MODEL_PATH / 'sovits'
SOVITS_DEFAULT_CONFIG = resources.files("sovits_svc") / 'configs' / 'base.yaml'


def get_drive_id(url):
    """ 通过网盘文件url获取id """
    pattern = r"(?:https?://)?(?:www\.)?drive\.google\.com/(?:file/d/|folder/d/|open\?id=|uc\?id=|drive/folders/)([a-zA-Z0-9_-]+)"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    else:
        return url


def get_sovits_model_list(model_root_path: Path=SOVITS_MODEL_PATH) -> list[dict[str, Any]] :
    """Find all sovits models under `model_root_path`

    Args:
        model_root_path (Path): path to search for models

    Returns:
        dict[str, Any]: models
    """
    ensure_models(MODEL_PATH)
    model_list = []

    if not model_root_path.exists() :
        return model_list

    for model_folder in model_root_path.iterdir():
        if not model_folder.is_dir() :
            continue
        model_path = None
        speaker_path = None

        for model_file in model_folder.iterdir() :
            if model_file.suffix == '.pth' :
                model_path = model_file
            elif model_file.suffix == '.npy' :
                speaker_path = model_file

        if model_path and speaker_path :
            model_list.append({
                "model_name": model_folder.name,
                "model_path": model_path,
                "speaker_path": speaker_path,
            })

    return model_list


@dataclass
class SovitsInferenceConfig :
    config: Path = None
    model: Path = None
    wave: Path = None
    spk: Path = None
    ppg: Path = None
    vec: Path = None
    pit: Path = None
    shift: int = 0
    pit_type: str = 'rmvpe'
    enable_retrieval: bool = False
    retrieval_index_prefix: str = ""
    retrieval_ratio: float = 0.5
    n_retrieval_vectors: int = 3
    hubert_index_path: Path = None
    whisper_index_path: Path = None
    debug: bool = False
    voice: str = ""


def sovits_convert_audio(
        audio_filepath: Path,
        model_name: str,
        model_path: Path,
        speaker_path: Path,
        shift: int=0,
        temp_dir: Path=BASE_DIR / 'temp',
        device='cpu',
    ) :
    args = SovitsInferenceConfig(
        model=model_path,
        spk=speaker_path,
        config=SOVITS_DEFAULT_CONFIG,
        voice=model_name,
        wave=audio_filepath,
        shift=shift,
    )
    config_file = model_path.with_name('config.yaml')
    custom_whisper = None
    if config_file.exists() :
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                custom_whisper = config.get('custom_whisper')
        except (FileNotFoundError, yaml.YAMLError):
            print(f"Error reading {config_file} or parsing YAML")
    print("custom whisper", custom_whisper)

    # 清空临时目录
    if temp_dir.is_dir() :
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    rmvpe_pretrain_path = MODEL_PATH / "rmvpe_pretrain" / "rmvpe2.pt"

    if (args.ppg is None):
        args.ppg = temp_dir / "svc_tmp.ppg.npy"
        print(
            f"Auto run : python whisper/inference.py -w {args.wave} -p {args.ppg}")
        whisper_infer(args.wave, args.ppg, custom_whisper, MODEL_PATH)

    if (args.vec is None):
        args.vec = temp_dir / "svc_tmp.vec.npy"
        print(
            f"Auto run : python hubert/inference.py -w {args.wave} -v {args.vec}")
        hubert_infer(args.wave, args.vec, str(MODEL_PATH))

    if (args.pit is None):
        args.pit = temp_dir / "svc_tmp.pit.csv"
        print(
            f"Auto run : python pitch/inference.py -w {args.wave} -p {args.pit}")
        pitch_infer(args.wave, args.pit, args.pit_type, rmvpe_pretrain_path)

    device = torch.device(device)
    hp = OmegaConf.load(args.config)
    model = SynthesizerInfer(
        hp.data.filter_length // 2 + 1,
        hp.data.segment_size // hp.data.hop_length,
        hp)
    svc_inference.load_svc_model(args.model, model)
    retrieval = svc_inference.create_retrival(args)
    model.eval()
    model.to(device)

    spk = np.load(args.spk)
    spk = torch.FloatTensor(spk)

    ppg = np.load(args.ppg)
    ppg = np.repeat(ppg, 2, 0)  # 320 PPG -> 160 * 2
    ppg = torch.FloatTensor(ppg)
    # ppg = torch.zeros_like(ppg)

    vec = np.load(args.vec)
    vec = np.repeat(vec, 2, 0)  # 320 PPG -> 160 * 2
    vec = torch.FloatTensor(vec)
    # vec = torch.zeros_like(vec)

    pit = load_csv_pitch(args.pit)
    print("pitch shift: ", args.shift)
    if (args.shift == 0):
        pass
    else:
        pit = np.array(pit)
        source = pit[pit > 0]
        source_ave = source.mean()
        source_min = source.min()
        source_max = source.max()
        print(f"source pitch statics: mean={source_ave:0.1f}, \
                min={source_min:0.1f}, max={source_max:0.1f}")
        shift = args.shift
        shift = 2 ** (shift / 12)
        pit = pit * shift
    pit = torch.FloatTensor(pit)

    shift_info = ''
    if args.shift > 0:
        shift_info = "(+" + str(args.shift) + ")"
    elif args.shift < 0:
        shift_info = "(" + str(args.shift) + ")"
    out_audio = svc_inference.svc_infer(model, retrieval, spk, pit, ppg, vec, hp, device, temp_dir)
    return (hp.data.sampling_rate, out_audio)
