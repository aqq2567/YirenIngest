"""Yiren ASR 引擎：阿里云百炼 DashScope Cloud ASR（Chat Completions API）。

认证：设置环境变量 DASHSCOPE_API_KEY 或在 config 中传入 api_key。
      使用前需在 https://bailian.console.aliyun.com/ 开通语音识别服务。

模型：qwen3-asr-flash（通过 OpenAI 兼容 Chat Completions 接口调用）
      API 接受 base64 编码的本地音频文件，返回纯文本转写结果。
      时间戳通过标点切句 + 字符比例分配估算。
"""

import os
import re
import wave
import base64
import logging
import subprocess
import tempfile
import shutil
from datetime import datetime, timezone, timedelta

from ..utils import load_project_env
from .config import get_engine_config

load_project_env()

logger = logging.getLogger(__name__)

TZ_CHINA = timezone(timedelta(hours=8))

# 单段音频最大时长（秒），超出自动切分
MAX_SEGMENT_SEC = 90

# 句末标点切分 regex：按 。！？；换行 切分
_SENTENCE_SPLIT_RE = re.compile(r'[。！？；\n]')


def _split_text_by_sentences(text: str, total_duration_s: float) -> list[dict]:
    """将全文按句末标点切分为句子，按字符比例估算每句起止时间。

    Chat Completions API 返回纯文本无时间戳，本函数按标点切分后，
    根据每句在全文中的字符位置比例分配时间。
    """
    if not text:
        return []

    total_chars = len(text)

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

        start_sec = round(c_start / total_chars * total_duration_s, 2)
        end_sec = round(c_end / total_chars * total_duration_s, 2)

        sentences.append({
            "text": sent,
            "start": start_sec,
            "end": end_sec,
        })

    return sentences


_openai_available = None


def _check_openai() -> bool:
    """惰性检查 openai 是否可用。"""
    global _openai_available
    if _openai_available is None:
        try:
            from openai import OpenAI  # noqa: F401
            _openai_available = True
        except ImportError:
            _openai_available = False
    return _openai_available


def _encode_audio_base64(audio_path: str) -> str:
    """将音频文件编码为 base64 data URI。

    自动检测 WAV 采样率/声道数，非 16kHz mono 时通过 ffmpeg 转换，
    确保 base64 大小不超 DashScope API 的 20MB data-uri 限制。
    """
    ext = os.path.splitext(audio_path)[1].lower()
    encode_path = audio_path
    tmp_path = None

    # 检查 WAV 是否需要降采样到 16kHz mono
    need_convert = False
    if ext == ".wav":
        try:
            with wave.open(audio_path, 'rb') as wf:
                need_convert = wf.getframerate() != 16000 or wf.getnchannels() != 1
        except Exception:
            logger.debug(f"WAV 头读取失败，将尝试转码: {audio_path}")

    if need_convert:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="asr16k_")
        os.close(tmp_fd)
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", audio_path,
                "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
                tmp_path,
            ], capture_output=True, check=True)
            encode_path = tmp_path
            logger.debug(
                f"音频降采样: {os.path.basename(audio_path)} → 16kHz mono "
                f"({os.path.getsize(audio_path)} → {os.path.getsize(tmp_path)} bytes)"
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning(
                "ffmpeg 不可用，无法降采样音频。"
                "90s 段在 44.1kHz+ 立体声下 base64 将超出 API 20MB 限制。"
            )
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            tmp_path = None

    try:
        with open(encode_path, "rb") as f:
            audio_data = base64.b64encode(f.read()).decode("utf-8")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    mime_map = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
    }
    mime = mime_map.get(ext, "audio/wav")

    return f"data:{mime};base64,{audio_data}"


def _call_chat_asr(
    audio_path: str,
    model: str,
    api_key: str,
    base_url: str,
) -> tuple[str, float]:
    """调用 Chat Completions API 转写单段音频，返回 (文本, 音频时长秒)。"""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)

    audio_data_uri = _encode_audio_base64(audio_path)

    completion = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [{
                "type": "input_audio",
                "input_audio": {
                    "data": audio_data_uri,
                },
            }],
        }],
        extra_body={
            "asr_options": {"enable_itn": False},
        },
    )

    text = completion.choices[0].message.content or ""

    # 从 usage 获取音频时长
    duration_s = 0.0
    if completion.usage and hasattr(completion.usage, 'seconds'):
        duration_s = float(completion.usage.seconds)

    return text.strip(), duration_s


def transcribe_audio(
    audio_path: str,
    transcript_id: str,
    series_id: str,
    source_file: str,
    config: dict | None = None,
) -> dict:
    """使用阿里云百炼 DashScope ASR 转写，返回标准 video_transcript dict。

    通过 OpenAI 兼容 Chat Completions 接口调用 qwen3-asr-flash 模型，
    音频以 base64 直接内嵌在请求中，无需 OSS 上传。
    时间戳通过标点切句 + 字符比例分配估算。
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    try:
        from openai import OpenAI  # noqa: F401
    except ImportError:
        raise ImportError("openai 未安装，请执行: pip install openai")

    cfg = get_engine_config("cloud", config)

    api_key = cfg.get("api_key") or os.getenv("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise ValueError(
            "缺少 DashScope API Key。请设置环境变量 DASHSCOPE_API_KEY 或 "
            "在 config 中传入 api_key。获取地址: https://bailian.console.aliyun.com/"
        )

    model = cfg.get("model", "qwen3-asr-flash")
    base_url = cfg.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")

    # --- 音频切分（长音频 → 分段处理）---
    from ..audio_processing import get_audio_duration_sec, segment_audio

    try:
        duration = get_audio_duration_sec(audio_path)
    except Exception:
        logger.debug(f"获取音频时长失败，使用默认值 0: {audio_path}")
        duration = 0.0

    if duration <= MAX_SEGMENT_SEC:
        audio_segments = [(audio_path, 0.0)]
    else:
        logger.info(f"音频过长 ({duration:.0f}s)，按 {MAX_SEGMENT_SEC}s 切分")
        tmp_dir = tempfile.mkdtemp(prefix="cloud_asr_")
        seg_paths = segment_audio(audio_path, tmp_dir, MAX_SEGMENT_SEC)

        audio_segments = []
        time_offset = 0.0
        for seg_path in seg_paths:
            audio_segments.append((seg_path, time_offset))
            try:
                time_offset += get_audio_duration_sec(seg_path)
            except Exception:
                logger.debug(f"获取分段时长失败，使用默认值: {seg_path}")
                time_offset += MAX_SEGMENT_SEC

    # --- 逐段调用 API ---
    all_segments: list[dict] = []

    for seg_idx, (seg_path, time_offset) in enumerate(audio_segments):
        logger.info(
            f"转写分段 {seg_idx + 1}/{len(audio_segments)}: "
            f"{os.path.basename(seg_path)}"
        )

        try:
            text, seg_duration = _call_chat_asr(seg_path, model, api_key, base_url)
        except Exception as e:
            logger.error(f"分段 {seg_idx + 1} API 调用失败: {e}")
            all_segments.append({
                "segment_id": f"seg_{len(all_segments) + 1:04d}",
                "start": round(time_offset, 2),
                "end": round(time_offset + MAX_SEGMENT_SEC, 2),
                "text": f"[API 调用失败: {e}]",
                "speaker": "narrator",
                "confidence": 0.0,
            })
            continue

        if not text:
            logger.warning(f"分段 {seg_idx + 1} 返回空文本")
            continue

        # 如果 API 未返回时长，回退从音频文件获取
        if seg_duration <= 0:
            try:
                seg_duration = get_audio_duration_sec(seg_path)
            except Exception:
                logger.debug(f"获取分段时长失败: {seg_path}")
                seg_duration = MAX_SEGMENT_SEC

        # 句末标点切分 → 估算时间戳
        sentence_parts = _split_text_by_sentences(text, seg_duration)

        if not sentence_parts:
            # 无标点，整段作为一个 segment
            all_segments.append({
                "segment_id": f"seg_{len(all_segments) + 1:04d}",
                "start": round(time_offset, 2),
                "end": round(time_offset + seg_duration, 2),
                "text": text,
                "speaker": "narrator",
                "confidence": 0.80,
            })
        else:
            for sp in sentence_parts:
                all_segments.append({
                    "segment_id": f"seg_{len(all_segments) + 1:04d}",
                    "start": round(sp["start"] + time_offset, 2),
                    "end": round(sp["end"] + time_offset, 2),
                    "text": sp["text"],
                    "speaker": "narrator",
                    "confidence": 0.80,
                })

    # --- 清理临时文件 ---
    if duration > MAX_SEGMENT_SEC and audio_segments:
        first_seg_dir = os.path.dirname(audio_segments[0][0])
        if first_seg_dir != os.path.dirname(audio_path):
            shutil.rmtree(first_seg_dir, ignore_errors=True)

    # --- 构建返回值 ---
    if not all_segments:
        logger.warning("Cloud ASR 未返回可解析的内容")
        all_segments = [{
            "segment_id": "seg_0001",
            "start": 0.0, "end": 0.0,
            "text": "", "speaker": "narrator", "confidence": 0.0,
        }]

    confidences = [s.get("confidence", 0.0) for s in all_segments]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    low_conf_count = sum(
        1 for s in all_segments if s.get("confidence", 0.0) < 0.7
    )

    result = {
        "transcript_id": transcript_id,
        "series_id": series_id,
        "source_file": source_file,
        "source_audio": audio_path,
        "language": cfg.get("language", "zh"),
        "asr_backend": f"cloud-dashscope-{model}",
        "asr_quality": {
            "estimated_cer": round(max(0.0, min(1.0, 1.0 - avg_conf)), 4),
            "low_confidence_segments": low_conf_count,
        },
        "ingested_at": datetime.now(TZ_CHINA).isoformat(),
        "segments": all_segments,
    }

    total_dur = (
        all_segments[-1]["end"] - all_segments[0]["start"]
        if all_segments else 0.0
    )
    logger.info(
        f"Cloud ASR 完成: {len(all_segments)} 段, 时长 {total_dur:.1f}s, "
        f"estimated_cer={result['asr_quality']['estimated_cer']}"
    )
    return result


# ============================================================
# 冒烟测试
# ============================================================
if __name__ == "__main__":
    print("=== cloud_asr_runner 冒烟测试 ===\n")

    deps_ok = True
    try:
        from openai import OpenAI  # noqa: F401
        print("[OK] openai SDK 可用")
    except ImportError:
        print("[WARN] openai 未安装 (pip install openai)")
        deps_ok = False

    if deps_ok:
        try:
            transcribe_audio("nonexistent.wav", "TR_T_001", "T_001", "test.mp4")
        except FileNotFoundError as e:
            print(f"[OK] FileNotFoundError: {e}")
        except Exception as e:
            print(f"[OK] 其他异常 (预期内): {type(e).__name__}: {e}")

    print("\n[DONE]")
