"""
Yiren 音频切分：将长音频按阈值切分以防止 Whisper OOM。

使用 Python 标准库 wave 进行无损切分，无需额外依赖。
"""

import os
import math
import wave
import logging

logger = logging.getLogger(__name__)


def get_audio_duration_sec(audio_path: str) -> float:
    """获取音频文件时长（秒）。

    优先使用 wave 模块直接读取（WAV 无损格式），
    不可用时回退到 moviepy。
    """
    # 尝试 wave 模块（标准库，支持 WAV）
    try:
        with wave.open(audio_path, 'r') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / rate if rate > 0 else 0.0
    except (wave.Error, FileNotFoundError):
        pass

    # 回退：moviepy（支持更多格式）
    try:
        from moviepy import AudioFileClip
        clip = AudioFileClip(audio_path)
        duration = clip.duration
        clip.close()
        return duration
    except ImportError:
        raise ImportError(
            "无法读取音频时长。请确保文件为 WAV 格式，或安装 moviepy:\n"
            "  pip install moviepy"
        )


def segment_audio(
    audio_path: str,
    output_dir: str,
    max_duration_sec: float = 1800.0,
) -> list[str]:
    """将长音频文件按最大时长切分为多段，防止 Whisper OOM。

    切分策略：按 max_duration_sec 等分，每段保存为独立 WAV 文件。
    如果音频时长短于 max_duration_sec，直接返回原文件路径（不复制）。

    Args:
        audio_path: 输入音频文件路径
        output_dir: 切分后的音频文件输出目录
        max_duration_sec: 单段最大时长（秒），默认 1800（30 分钟）

    Returns:
        切分后的 WAV 文件路径列表（按时间顺序）
    """
    os.makedirs(output_dir, exist_ok=True)

    duration = get_audio_duration_sec(audio_path)
    if duration <= max_duration_sec:
        logger.info(
            f"音频时长 {duration:.1f}s 未超过阈值 {max_duration_sec:.0f}s，无需切分"
        )
        return [audio_path]

    num_segments = math.ceil(duration / max_duration_sec)
    logger.info(
        f"长音频切分: {audio_path} ({duration:.1f}s) -> {num_segments} 段 "
        f"(阈值 {max_duration_sec:.0f}s)"
    )

    base_name = os.path.splitext(os.path.basename(audio_path))[0]

    with wave.open(audio_path, 'rb') as wf_in:
        nchannels = wf_in.getnchannels()
        sampwidth = wf_in.getsampwidth()
        framerate = wf_in.getframerate()
        total_frames = wf_in.getnframes()

        segment_paths = []
        for i in range(num_segments):
            start_frame = i * int(max_duration_sec * framerate)
            end_frame = min((i + 1) * int(max_duration_sec * framerate), total_frames)

            wf_in.setpos(start_frame)
            frames_data = wf_in.readframes(end_frame - start_frame)

            seg_path = os.path.join(output_dir, f"{base_name}_seg{i + 1:03d}.wav")
            with wave.open(seg_path, 'wb') as wf_out:
                wf_out.setnchannels(nchannels)
                wf_out.setsampwidth(sampwidth)
                wf_out.setframerate(framerate)
                wf_out.writeframes(frames_data)

            seg_dur = (end_frame - start_frame) / framerate
            logger.info(f"  段 {i + 1}/{num_segments}: {seg_path} ({seg_dur:.1f}s)")
            segment_paths.append(seg_path)

    return segment_paths


# ============================================================
# 本地测试入口
# ============================================================
if __name__ == "__main__":
    import tempfile, struct

    print("=== segmenter unit tests ===\n")

    tmp_dir = tempfile.mkdtemp()
    test_wav = os.path.join(tmp_dir, "test_long.wav")

    # 创建 5 秒测试 WAV
    with wave.open(test_wav, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        for _ in range(16000 * 5):
            wf.writeframes(struct.pack('<h', 0))

    # 测试 1：不超过阈值时不切分
    result = segment_audio(test_wav, tmp_dir, max_duration_sec=10)
    assert len(result) == 1 and result[0] == test_wav, "短音频不应切分"
    print(f"[OK] 短音频不切分: {len(result)} segments")

    # 测试 2：超过阈值时切分
    result = segment_audio(test_wav, tmp_dir, max_duration_sec=2)
    assert len(result) == 3, f"5s/2s 应切为 3 段，实际: {len(result)}"
    for p in result:
        assert os.path.exists(p), f"分段文件应存在: {p}"
    print(f"[OK] 长音频切分: {len(result)} segments")

    # 测试时长
    dur = get_audio_duration_sec(test_wav)
    assert 4.9 < dur < 5.1, f"duration out of range: {dur}"
    print(f"[OK] get_audio_duration_sec: {dur:.2f}s")

    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("\n[PASS] segmenter tests passed")
