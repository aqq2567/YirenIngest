"""Yiren ASR 引擎：Whisper 高精度语音转写 (faster_whisper)。

模型单例复用，支持多线程并发 transcribe()——单份权重在 VRAM 中，
CTranslate2 内部 inter_threads 负责调度并发 GPU 算子。
"""

import os
import math
import threading
import logging
from datetime import datetime, timezone, timedelta

from .config import get_engine_config

logger = logging.getLogger(__name__)

_whisper_available = None
_device_cache = None  # 缓存设备检测结果
TZ_CHINA = timezone(timedelta(hours=8))

# ── 模型单例 ──────────────────────────────────────────────────────────
_shared_model = None
_shared_model_key = None
_model_lock = threading.Lock()

# ── 并发控制 ──────────────────────────────────────────────────────────
# faster-whisper 的 transcribe() 包含非线程安全的音频预处理（Mel 特征提取）。
# 信号量限制同时执行 transcribe() 的线程数，模型权重在 VRAM 中仅一份。
# 默认 2 是安全值（已验证）；RTX 5090 32GB 可尝试 3-4，但需实测稳定性。
_transcribe_semaphore = None
_semaphore_lock = threading.Lock()


def _set_concurrency(max_calls: int):
    """设置全局 ASR 并发上限（由 orchestrator 在启动时调用）。"""
    global _transcribe_semaphore
    with _semaphore_lock:
        _transcribe_semaphore = threading.BoundedSemaphore(max_calls)
        logger.info(f"ASR 并发上限: {max_calls}")


def _detect_device_and_compute() -> tuple[str, str]:
    """检测最优 ASR 设备配置。返回 (device, compute_type)。

    GPU 可用 → ("cuda", "float16")，否则 → ("cpu", "int8")。
    结果被缓存，只需检测一次。
    """
    global _device_cache
    if _device_cache is not None:
        return _device_cache
    try:
        import torch
        if torch.cuda.is_available():
            _device_cache = ("cuda", "float16")
            logger.info(f"检测到 CUDA (GPU: {torch.cuda.get_device_name(0)})，使用 GPU 加速")
            return _device_cache
    except ImportError:
        pass
    _device_cache = ("cpu", "int8")
    logger.info("未检测到 CUDA，使用 CPU + int8 模式")
    return _device_cache


def _check_whisper() -> bool:
    """惰性检查 faster_whisper 是否可用。"""
    global _whisper_available
    if _whisper_available is None:
        try:
            from faster_whisper import WhisperModel  # noqa: F401
            _whisper_available = True
        except ImportError:
            _whisper_available = False
    return _whisper_available


def _get_shared_model(config: dict | None = None):
    """线程安全懒加载：返回全局复用的 WhisperModel 实例。

    首次调用时加载模型到 VRAM 并缓存。后续调用直接返回已缓存的实例，
    无需重新加载。CTranslate2 的 WhisperModel 本身线程安全——多个线程
    可同时对其调用 transcribe()。

    当配置（model_size / device / compute_type）变化时自动重建。
    """
    global _shared_model, _shared_model_key
    from faster_whisper import WhisperModel

    cfg = get_engine_config("whisper", config)
    model_size = cfg["model_size"]
    user_device = (config or {}).get("device")
    user_compute = (config or {}).get("compute_type")
    auto_device, auto_compute = _detect_device_and_compute()
    device = user_device if user_device and user_device != "auto" else auto_device
    compute_type = (
        user_compute if user_compute and user_compute != "auto" else auto_compute
    )

    config_key = (model_size, device, compute_type, cfg["cpu_threads"],
                   cfg.get("cache_dir"))

    if _shared_model is not None and _shared_model_key == config_key:
        return _shared_model

    with _model_lock:
        # 双重检查：锁内再次确认配置未变
        if _shared_model is not None and _shared_model_key == config_key:
            return _shared_model

        logger.info(f"加载 Whisper 模型单例 [{model_size}] device={device} "
                     f"compute={compute_type}")
        try:
            _shared_model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                cpu_threads=cfg["cpu_threads"],
                download_root=cfg.get("cache_dir"),
            )
        except Exception as e:
            msg = str(e)
            if ("cublas" in msg.lower() or "cudnn" in msg.lower()) and device == "cuda":
                logger.warning(
                    f"CUDA 运行时库不可用 ({msg[:200]})，降级为 CPU + int8"
                )
                device = "cpu"
                compute_type = "int8"
                config_key = (model_size, device, compute_type,
                               cfg["cpu_threads"], cfg.get("cache_dir"))
                _shared_model = WhisperModel(
                    model_size, device=device, compute_type=compute_type,
                    cpu_threads=cfg["cpu_threads"],
                    download_root=cfg.get("cache_dir"),
                )
            else:
                raise
        _shared_model_key = config_key
        return _shared_model


def transcribe_audio(
    audio_path: str,
    transcript_id: str,
    series_id: str,
    source_file: str,
    config: dict | None = None,
) -> dict:
    """对单个音频文件执行高精度 ASR 转写，返回标准 video_transcript dict。

    使用全局共享 WhisperModel 实例（线程安全，可多路并发）。
    """
    if not _check_whisper():
        raise ImportError("faster_whisper 未安装，请执行: pip install faster_whisper")

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    cfg = get_engine_config("whisper", config)
    model_size = cfg["model_size"]
    model = _get_shared_model(config)

    # 信号量限流：防止并发 transcribe() 过多导致 CUDA 崩溃
    sem = _transcribe_semaphore
    if sem is not None:
        sem.acquire()
    try:
        logger.info(f"开始转写: {audio_path}")
        try:
            segments, info = model.transcribe(
                audio_path,
                vad_filter=cfg["vad_filter"],
                beam_size=cfg["beam_size"],
                initial_prompt=cfg["initial_prompt"],
                language=cfg["language"],
                condition_on_previous_text=False,
                compression_ratio_threshold=cfg["compression_ratio_threshold"],
                no_speech_threshold=cfg["no_speech_threshold"],
            )
        except Exception as e:
            raise RuntimeError(
                f"Whisper 转写失败 ({os.path.basename(audio_path)}): {e}"
            ) from e

        segments_out = []
        confidences = []
        low_conf_count = 0
        for idx, seg in enumerate(segments):
            confidence = round(math.exp(seg.avg_logprob), 4) if seg.avg_logprob is not None else 1.0
            confidences.append(confidence)
            if confidence < 0.7:
                low_conf_count += 1
            segments_out.append({
                "segment_id": f"seg_{idx + 1:04d}",
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
                "speaker": "narrator",
                "confidence": confidence,
            })

        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        result = {
            "transcript_id": transcript_id,
            "series_id": series_id,
            "source_file": source_file,
            "source_audio": audio_path,
            "language": info.language,
            "asr_backend": f"whisper-{model_size}",
            "asr_quality": {
                "estimated_cer": round(max(0.0, min(1.0, 1.0 - avg_conf)), 4),
                "low_confidence_segments": low_conf_count,
            },
            "ingested_at": datetime.now(TZ_CHINA).isoformat(),
            "segments": segments_out,
        }

        total_dur = (
            result["segments"][-1]["end"] - result["segments"][0]["start"]
            if result["segments"]
            else 0.0
        )
        logger.info(
            f"转写完成: {len(result['segments'])} 段, 时长 {total_dur:.1f}s, "
            f"estimated_cer={result['asr_quality']['estimated_cer']}, "
            f"low_conf_segments={low_conf_count}"
        )
        return result
    finally:
        if sem is not None:
            sem.release()


# ============================================================
# 冒烟测试
# ============================================================
if __name__ == "__main__":
    print("=== whisper_runner 冒烟测试 ===\n")

    # 缺少依赖时抛 ImportError
    if not _check_whisper():
        try:
            transcribe_audio("fake.wav", "TR_T_001", "T_001", "test.mp4")
            assert False
        except ImportError as e:
            print(f"[OK] ImportError: {e}")
    else:
        print("[OK] faster_whisper 可用")
        try:
            transcribe_audio("nonexistent.wav", "TR_T_001", "T_001", "test.mp4")
            assert False
        except FileNotFoundError as e:
            print(f"[OK] FileNotFoundError: {e}")

    print("[PASS]")
