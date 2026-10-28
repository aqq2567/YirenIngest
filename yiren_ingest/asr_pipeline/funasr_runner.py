"""Yiren ASR 引擎：FunASR Paraformer 高精度语音转写 (funasr)。

前置依赖：
    pip install funasr
    ffmpeg（VAD 模型必需，Windows: choco install ffmpeg / winget install ffmpeg）

VAD 依赖 ffmpeg 子进程加载音频。若 ffmpeg 未安装，VAD 自动禁用并给出警告，
ASR 本身仍可正常工作（仅无语音端点检测）。
"""

import os
import re
import shutil
import logging
from datetime import datetime, timezone, timedelta

from .config import get_engine_config, resolve_device

logger = logging.getLogger(__name__)

_funasr_available = None
_ffmpeg_available = None
TZ_CHINA = timezone(timedelta(hours=8))

# 句末标点切分 regex：按 。！？；换行 切分，保留标点在句末
_SENTENCE_SPLIT_RE = re.compile(r'[。！？；\n]')


def _split_text_by_sentences(text: str, timestamps: list) -> list[dict]:
    """将全文按句末标点切分为句子，按字符比例从词级时间戳估算每句时间。

    FunASR paraformer-zh + PUNC 返回带标点的全文文本 + 词级时间戳数组，
    但无逐句的 sentence_info。本函数按句末标点切分句子，根据每句在原文中
    的字符位置，按比例映射到时间戳数组，估算 start_ms / end_ms。
    """
    if not text or not timestamps:
        return []

    total_chars = len(text)
    total_ts = len(timestamps)

    # 用 finditer 找所有句末标点位置，避免 re.split 的空白/空串问题
    boundaries = [0]
    for m in _SENTENCE_SPLIT_RE.finditer(text):
        boundaries.append(m.end())
    if boundaries[-1] < total_chars:
        boundaries.append(total_chars)

    sentences = []
    for i in range(len(boundaries) - 1):
        sent = text[boundaries[i]:boundaries[i + 1]].strip()
        if not sent:
            continue
        c_start = boundaries[i]
        c_end = boundaries[i + 1]

        # 按字符位置比例映射到时间戳索引
        ts_i = min(int(c_start / total_chars * total_ts), total_ts - 1)
        ts_j = min(int(c_end / total_chars * total_ts), total_ts - 1)
        start_ms = timestamps[ts_i][0]
        end_ms = timestamps[ts_j][1]

        sentences.append({
            "text": sent,
            "start_ms": start_ms,
            "end_ms": end_ms,
        })

    return sentences


def _check_ffmpeg() -> bool:
    """惰性检查 ffmpeg 是否在 PATH 中可用（FunASR VAD 模型依赖）。"""
    global _ffmpeg_available
    if _ffmpeg_available is None:
        _ffmpeg_available = shutil.which("ffmpeg") is not None
    return _ffmpeg_available


def _check_funasr() -> bool:
    """惰性检查 funasr 是否可用。"""
    global _funasr_available
    if _funasr_available is None:
        try:
            from funasr import AutoModel  # noqa: F401
            _funasr_available = True
        except ImportError:
            _funasr_available = False
    return _funasr_available


def transcribe_audio(
    audio_path: str,
    transcript_id: str,
    series_id: str,
    source_file: str,
    config: dict | None = None,
) -> dict:
    """对单个音频文件执行高精度 ASR 转写，返回标准 video_transcript dict（对齐 §3.4）。

    config 键: model("paraformer-zh"), vad_model("fsmn-vad"), punc_model("ct-punc"),
    device("auto"), ncpu(16), use_vad(True), hotwords(学科术语), batch_size_s(300)

    Requires: pip install funasr
    """
    if not _check_funasr():
        raise ImportError("funasr 未安装，请执行: pip install funasr")
    from funasr import AutoModel

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    cfg = get_engine_config("funasr", config)
    model_name = cfg["model"]

    # ffmpeg 可用性检测：FunASR VAD 模型内部硬编码调用 ffmpeg 子进程加载音频，
    # Windows 默认不含 ffmpeg，缺失时自动禁用 VAD 并给出明确警告。
    use_vad = cfg.get("use_vad", True) and bool(cfg.get("vad_model"))
    if use_vad and not _check_ffmpeg():
        logger.warning(
            "ffmpeg 未安装，VAD 模型将禁用。ASR 仍可正常工作，但缺少语音端点检测。\n"
            "安装 ffmpeg：  choco install ffmpeg -y    (Windows)\n"
            "            sudo apt install ffmpeg      (Ubuntu)\n"
            "            brew install ffmpeg           (macOS)"
        )
        use_vad = False

    logger.info(f"加载 FunASR 模型 [{model_name}]，首次使用将自动下载")

    # 组装 AutoModel 参数（引擎不认识的键静默忽略）
    model_kwargs = {
        "model": model_name,
        "device": resolve_device(cfg.get("device")),
        "ncpu": cfg.get("ncpu", 16),
        "hub": cfg.get("hub", "ms"),
    }
    if use_vad:
        model_kwargs["vad_model"] = cfg["vad_model"]
    if cfg.get("punc_model"):
        model_kwargs["punc_model"] = cfg["punc_model"]
    # hotwords：新版 FunASR 直接支持字符串
    if cfg.get("hotwords"):
        model_kwargs["hotword"] = cfg["hotwords"]

    try:
        model = AutoModel(**model_kwargs)
    except Exception as e:
        msg = str(e)
        if any(kw in msg for kw in ("Network", "ConnectError", "unreachable", "download")):
            raise ConnectionError(
                f"无法下载 FunASR 模型 [{model_name}]，请检查网络。\n原始错误: {msg}"
            )
        # GPU OOM / CUDA 不可用 → 自动降级 CPU
        device = model_kwargs.get("device", "cpu")
        if device != "cpu" and any(
            kw in msg.lower()
            for kw in ("memory", "oom", "out of memory", "cuda", "cublas", "cudnn")
        ):
            logger.warning(
                f"FunASR GPU 不可用 ({msg[:200]})，降级为 CPU"
            )
            model_kwargs["device"] = "cpu"
            model = AutoModel(**model_kwargs)
        else:
            raise

    logger.info(f"开始转写: {audio_path}")
    try:
        output = model.generate(
            input=audio_path,
            batch_size_s=cfg.get("batch_size_s", 300),
        )
    except Exception as e:
        raise RuntimeError(f"FunASR 转写失败: {e}")

    # 解析 FunASR 输出 → 标准 segments 结构
    # output 为列表 [per_file_results]，per_file_results 为 list[dict] 或 dict
    raw = output[0] if output else []

    segments = []
    confidences = []
    low_conf_count = 0

    if isinstance(raw, list) and raw:
        # 标准格式: [{"text": "...", "start": 100, "end": 3200}, ...]
        for idx, seg in enumerate(raw):
            if not isinstance(seg, dict):
                continue
            text = seg.get("text", "").strip()
            if not text:
                continue

            # 时间戳（FunASR 输出毫秒，转换为秒）
            start_ms = seg.get("start", 0)
            end_ms = seg.get("end", 0)
            if (start_ms == 0 and end_ms == 0) and "timestamp" in seg:
                ts = seg["timestamp"]
                if ts and isinstance(ts, list) and ts[0]:
                    start_ms = ts[0][0] if isinstance(ts[0], list) else 0
                    end_ms = ts[-1][-1] if isinstance(ts[-1], list) else 0

            start_sec = round(start_ms / 1000.0, 2) if start_ms else round(idx * 5.0, 2)
            end_sec = round(end_ms / 1000.0, 2) if end_ms else round((idx + 1) * 5.0, 2)

            confidence = seg.get("confidence", seg.get("score", 1.0))
            if isinstance(confidence, (int, float)):
                confidence = round(float(confidence), 4)
            else:
                confidence = 1.0
            confidences.append(confidence)
            if confidence < 0.7:
                low_conf_count += 1

            segments.append({
                "segment_id": f"seg_{idx + 1:04d}",
                "start": start_sec,
                "end": end_sec,
                "text": text,
                "speaker": "narrator",
                "confidence": confidence,
            })

    elif isinstance(raw, dict):
        # dict 格式有多种变体，按优先级尝试提取 segments：
        #   1) sentence_info 数组 → 逐段展开（最理想，有逐段文本+时间戳）
        #   2) timestamp 数组（无逐段文本）→ 仅用首尾时间恢复整体 anchors
        #   3) 纯 text → 兜底，时间戳归零
        full_text = raw.get("text", "").strip()
        sentence_info = raw.get("sentence_info", [])
        timestamp = raw.get("timestamp", [])

        if sentence_info and isinstance(sentence_info, list):
            for idx, si in enumerate(sentence_info):
                text = si.get("text", "").strip()
                if not text:
                    continue
                start_sec = round(si.get("start", idx * 5.0) / 1000.0, 2)
                end_sec = round(si.get("end", (idx + 1) * 5.0) / 1000.0, 2)
                conf = si.get("confidence", 1.0)
                if isinstance(conf, (int, float)):
                    conf = round(float(conf), 4)
                else:
                    conf = 1.0
                confidences.append(conf)
                if conf < 0.7:
                    low_conf_count += 1
                segments.append({
                    "segment_id": f"seg_{idx + 1:04d}",
                    "start": start_sec,
                    "end": end_sec,
                    "text": text,
                    "speaker": "narrator",
                    "confidence": conf,
                })

        elif (timestamp and isinstance(timestamp, list)
              and isinstance(timestamp[0], list)):
            # timestamp = [[start_ms, end_ms], ...] 词级时间戳
            # 无 sentence_info，需按标点将全文切分为句子再映射时间
            if full_text:
                sent_parts = _split_text_by_sentences(full_text, timestamp)
                if sent_parts:
                    for idx, sp in enumerate(sent_parts):
                        conf = 0.85  # ASR 原文 + PUNC 标点，置信度经验值
                        confidences.append(conf)
                        if conf < 0.7:
                            low_conf_count += 1
                        segments.append({
                            "segment_id": f"seg_{idx + 1:04d}",
                            "start": round(sp["start_ms"] / 1000.0, 2),
                            "end": round(sp["end_ms"] / 1000.0, 2),
                            "text": sp["text"],
                            "speaker": "narrator",
                            "confidence": conf,
                        })
                else:
                    # 切分失败（无标点），回退为单段
                    overall_start = round(timestamp[0][0] / 1000.0, 2)
                    overall_end = round(timestamp[-1][-1] / 1000.0, 2)
                    segments.append({
                        "segment_id": "seg_0001",
                        "start": overall_start,
                        "end": overall_end,
                        "text": full_text,
                        "speaker": "narrator",
                        "confidence": 0.5,
                    })
                    confidences.append(0.5)

        elif full_text:
            # 纯文本，无任何时序信息
            segments.append({
                "segment_id": "seg_0001",
                "start": 0.0,
                "end": 0.0,
                "text": full_text,
                "speaker": "narrator",
                "confidence": 0.5,
            })
            confidences.append(0.5)

    # 兜底：没有任何有效段
    if not segments:
        logger.warning("FunASR 未返回可解析的内容，返回空 segments")
        segments.append({
            "segment_id": "seg_0001",
            "start": 0.0,
            "end": 0.0,
            "text": "",
            "speaker": "narrator",
            "confidence": 0.0,
        })
        confidences = [0.0]

    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    result = {
        "transcript_id": transcript_id,
        "series_id": series_id,
        "source_file": source_file,
        "source_audio": audio_path,
        "language": cfg.get("language", "zh"),
        "asr_backend": f"funasr-{model_name}",
        "asr_quality": {
            "estimated_cer": round(max(0.0, min(1.0, 1.0 - avg_conf)), 4),
            "low_confidence_segments": low_conf_count,
        },
        "ingested_at": datetime.now(TZ_CHINA).isoformat(),
        "segments": segments,
    }

    total_dur = segments[-1]["end"] - segments[0]["start"] if segments else 0.0
    logger.info(
        f"转写完成: {len(segments)} 段, 时长 {total_dur:.1f}s, "
        f"estimated_cer={result['asr_quality']['estimated_cer']}, "
        f"low_conf_segments={low_conf_count}"
    )
    return result


# ============================================================
# 冒烟测试
# ============================================================
if __name__ == "__main__":
    print("=== funasr_runner 冒烟测试 ===\n")

    if not _check_funasr():
        try:
            transcribe_audio("fake.wav", "TR_T_001", "T_001", "test.mp4")
            assert False
        except ImportError as e:
            print(f"[OK] ImportError: {e}")
    else:
        print("[OK] funasr 可用")
        try:
            transcribe_audio("nonexistent.wav", "TR_T_001", "T_001", "test.mp4")
            assert False
        except FileNotFoundError as e:
            print(f"[OK] FileNotFoundError: {e}")

    print("[PASS]")
