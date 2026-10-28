"""Yiren ASR Pipeline: Whisper / FunASR / Cloud ASR 自动调度。

引擎列表:
    whisper_runner  — faster_whisper (本地, 默认首选)
    funasr_runner   — FunASR Paraformer (本地备用)
    cloud_asr_runner— 阿里云百炼 DashScope ASR (云端兜底)

统一对外接口:
    from yiren_ingest.asr_pipeline import transcribe_audio
"""

from .asr_dispatcher import transcribe_audio
from .config import (
    get_engine_config, BASE_CONFIG,
    WHISPER_DEFAULTS, FUNASR_DEFAULTS, CLOUD_DEFAULTS,
)
