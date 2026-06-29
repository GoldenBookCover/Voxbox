import os.path
import shutil
import zipfile
from pathlib import Path

from pydownloader import Downloader


def ensure_models(model_path: Path):
    dl = Downloader()
    if not os.path.exists(model_path / "crepe/assets"):
        os.makedirs(model_path / "crepe/assets")
    if not os.path.exists(model_path / "crepe/assets/full.pth"):
        dl.download("https://github.com/maxrmorrison/torchcrepe/raw/master/torchcrepe/assets/full.pth",
                      model_path / "crepe/assets/full.pth")

    if not os.path.exists(model_path / "hubert_pretrain"):
        os.makedirs(model_path / "hubert_pretrain")
    if not os.path.exists(model_path / "hubert_pretrain/hubert-soft-0d54a1f4.pt"):
        dl.download("https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt",
                      model_path / "hubert_pretrain/hubert-soft-0d54a1f4.pt")

    if not os.path.exists(model_path / "speaker_pretrain"):
        os.makedirs(model_path / "speaker_pretrain")
    if not os.path.exists(model_path / "speaker_pretrain/best_model.pth.tar"):
        dl.download("https://drive.google.com/file/d/1UPjQ2LVSIt3o-9QMKMJcdzT8aZRZCI-E/view?usp=drive_link",
                       model_path / "speaker_pretrain/best_model.pth.tar")

    if not os.path.exists(model_path / "speaker_pretrain/config.json"):
        dl.download(
            "https://raw.githubusercontent.com/PlayVoice/so-vits-svc-5.0/bigvgan-mix-v2/speaker_pretrain/config.json",
            model_path / "speaker_pretrain/config.json")

    if not os.path.exists(model_path / "vits_pretrain"):
        os.makedirs(model_path / "vits_pretrain")
    if not os.path.exists(model_path / "vits_pretrain/sovits5.0.pretrain.pth"):
        dl.download("https://github.com/PlayVoice/so-vits-svc-5.0/releases/download/5.0/sovits5.0.pretrain.pth",
                      model_path / "vits_pretrain/sovits5.0.pretrain.pth")

    if not os.path.exists(model_path / "whisper_pretrain"):
        os.makedirs(model_path / "whisper_pretrain")
    if not os.path.exists(model_path / "whisper_pretrain/large-v2.pt"):
        dl.download(
            "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
            model_path / "whisper_pretrain/large-v2.pt")

    if not os.path.exists(model_path / "rmvpe_pretrain"):
        os.makedirs(model_path / "rmvpe_pretrain")
    if not os.path.exists(model_path / "rmvpe_pretrain/rmvpe2.pt"):
        dl.download("https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip", model_path / "rmvpe.zip")
        with zipfile.ZipFile(model_path / "rmvpe.zip", "r") as zip_ref:
            zip_ref.extractall("rmvpe_pretrain")
            shutil.move(model_path / "rmvpe_pretrain/model.pt", model_path / "rmvpe_pretrain/rmvpe2.pt")
        os.remove(model_path / "rmvpe.zip")