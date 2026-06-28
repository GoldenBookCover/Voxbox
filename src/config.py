import os
from pathlib import Path


os.environ['TRITON_PTXAS_PATH'] = '/usr/bin/ptxas'
#os.environ['MODELSCOPE_CACHE'] = ''
#os.environ["FUNASR_USE_HF"] = "1"
#os.environ["HF_ENDPOINT"] = "https://huggingface.co"

BASE_DIR = Path(__file__).resolve().parent.parent

# Share pages directory & static server
SHARE_DIR = BASE_DIR / "share_pages"
# separate port for static share pages
SHARE_SERVER_PORT = 17866

# Auto load reference audios
REFERENCE_AUDIO_DIR = BASE_DIR / 'reference_audio'
ALLOWED_AUDIO_FORMAT = ('.wav', )

# Sovits model
MODEL_PATH = BASE_DIR / 'models'
