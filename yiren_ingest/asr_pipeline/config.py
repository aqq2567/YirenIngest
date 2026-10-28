"""ASR Pipeline 共享配置：所有引擎的默认参数集中管理。

领域词表来自 `yiren_ingest/domain_terms.py`（单一真相源）：
- Whisper 用 GAME_DOMAIN_TERMS 作为 initial_prompt
- FunASR 用 GAME_HOTWORDS 作为热词
"""

from ..domain_terms import GAME_DOMAIN_TERMS, GAME_HOTWORDS

# ============================================================
# 基座配置：所有引擎共享的参数
# ============================================================
BASE_CONFIG = {
    "language": "zh",
    "device": "auto",
    "cache_dir": None,       # 模型存储目录，None=使用各引擎默认位置
}


def resolve_device(device: str | None) -> str:
    """把 auto/空值解析为实际设备；显式 cuda/cpu 原样返回。

    延迟导入 torch，避免只读取配置时加载重型依赖。
    """
    if device and device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

# ============================================================
# Whisper 引擎专属默认值
# ============================================================
WHISPER_DEFAULTS = {
    "model_size": "large-v3",
    "compute_type": "float16",
    "cpu_threads": 16,
    "beam_size": 5,
    "initial_prompt": GAME_DOMAIN_TERMS,
    "vad_filter": True,
    "compression_ratio_threshold": 2.4,
    "no_speech_threshold": 0.6,
}

# ============================================================
# FunASR 引擎专属默认值
# ============================================================
FUNASR_DEFAULTS = {
    "model": "paraformer-zh",
    "vad_model": "fsmn-vad",
    "punc_model": "ct-punc",
    "ncpu": 16,
    "use_vad": True,
    "hotwords": GAME_HOTWORDS,
}

# ============================================================
# Cloud ASR 引擎专属默认值
# ============================================================
CLOUD_DEFAULTS = {
    "provider": "aliyun",           # aliyun / tencent / azure
    "model": "qwen3-asr-flash",     # Chat Completions 接口，base64 本地音频
    "api_key": "",                  # 百炼 API Key（默认读 DASHSCOPE_API_KEY）
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "language": "zh",
}

# ============================================================
# 配置合并：基座 < 引擎默认 < 用户覆盖
# ============================================================
ENGINE_DEFAULTS = {
    "whisper": WHISPER_DEFAULTS,
    "funasr": FUNASR_DEFAULTS,
    "cloud": CLOUD_DEFAULTS,
}


def get_engine_config(engine: str, user_config: dict | None = None) -> dict:
    """获取合并后的引擎配置。

    优先级：用户配置 > 引擎默认 > 基座配置。
    用户 config 中未被引擎识别的键会被静默保留（各引擎自行忽略）。
    """
    engine_key = engine.lower()
    if engine_key not in ENGINE_DEFAULTS:
        raise ValueError(f"未知引擎: {engine}，可选: {list(ENGINE_DEFAULTS)}")

    merged = {}
    merged.update(BASE_CONFIG)
    merged.update(ENGINE_DEFAULTS[engine_key])
    if user_config:
        merged.update(user_config)
    return merged
