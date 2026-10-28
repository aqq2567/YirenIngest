"""
Yiren 音频提取：从视频分离音频轨道 + 按时间切片 + 保存音频样本供 TTS 训练。

依赖: moviepy（惰性导入）、wave（标准库，用于 WAV 切片）。
"""

import os
import wave
import shutil
import logging

logger = logging.getLogger(__name__)


def extract_audio_from_video(video_path: str, audio_output_path: str) -> str:
    """从视频提取音频轨道，输出 WAV 文件。需要 pip install moviepy。"""
    try:
        from moviepy import VideoFileClip
    except ImportError:
        raise ImportError("moviepy 未安装，请执行: pip install moviepy")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    os.makedirs(os.path.dirname(audio_output_path) or ".", exist_ok=True)

    clip = VideoFileClip(video_path)
    try:
        if clip.audio is None:
            raise ValueError(f"视频文件无音轨: {video_path}，请检查文件是否包含音频流。")
        clip.audio.write_audiofile(audio_output_path, logger=None)
    finally:
        clip.close()

    logger.info(f"音频提取完成: {video_path} -> {audio_output_path}")
    return audio_output_path


def extract_audio_segment(
    audio_path: str,
    start_sec: float,
    end_sec: float,
    output_path: str,
) -> str:
    """从音频文件按起止时间截取一段 WAV。WAV 用标准库 wave，其余回退 moviepy。"""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")
    if start_sec < 0:
        raise ValueError(f"起始时间不能为负: {start_sec}")
    if end_sec <= start_sec:
        raise ValueError(f"结束时间 {end_sec} 必须大于起始时间 {start_sec}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # WAV 路径：wave 标准库无损切片
    try:
        with wave.open(audio_path, 'rb') as wf_in:
            framerate = wf_in.getframerate()
            nchannels = wf_in.getnchannels()
            sampwidth = wf_in.getsampwidth()
            total_frames = wf_in.getnframes()
            total_dur = total_frames / framerate if framerate > 0 else 0.0

            if start_sec >= total_dur:
                raise ValueError(f"起始时间 {start_sec}s 超出音频总时长 {total_dur:.1f}s")

            start_frame = int(start_sec * framerate)
            end_frame = min(int(end_sec * framerate), total_frames)
            wf_in.setpos(start_frame)
            frames_data = wf_in.readframes(end_frame - start_frame)

        with wave.open(output_path, 'wb') as wf_out:
            wf_out.setnchannels(nchannels)
            wf_out.setsampwidth(sampwidth)
            wf_out.setframerate(framerate)
            wf_out.writeframes(frames_data)

        logger.info(f"切片完成: [{start_sec:.1f}s-{end_sec:.1f}s] -> {output_path}")
        return output_path
    except (wave.Error, FileNotFoundError, EOFError):
        logger.debug(f"WAV 切片不可用，回退 moviepy: {audio_path}")

    # moviepy 回退
    try:
        from moviepy import AudioFileClip
    except ImportError:
        raise ImportError(
            "无法切片非 WAV 格式音频。请安装 moviepy: pip install moviepy\n"
            f"或先将音频转为 WAV: {audio_path}"
        )

    clip = AudioFileClip(audio_path)
    try:
        if start_sec >= clip.duration:
            raise ValueError(f"起始时间 {start_sec}s 超出音频总时长 {clip.duration:.1f}s")
        end = min(end_sec, clip.duration)
        clip.subclipped(start_sec, end).write_audiofile(output_path, logger=None)
    finally:
        clip.close()

    logger.info(f"切片完成(moviepy): [{start_sec:.1f}s-{end_sec:.1f}s] -> {output_path}")
    return output_path


# ============================================================
# 本地测试入口
# ============================================================
if __name__ == "__main__":
    import tempfile, struct
    from .segmenter import get_audio_duration_sec

    print("=== extractor unit tests ===\n")

    tmp_dir = tempfile.mkdtemp()
    test_wav = os.path.join(tmp_dir, "test_source.wav")
    with wave.open(test_wav, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        for _ in range(16000):
            wf.writeframes(struct.pack('<h', 0))

    # 测试 extract_audio_segment (WAV 切片)
    test_slice = os.path.join(tmp_dir, "test_slice.wav")
    result = extract_audio_segment(test_wav, 0.1, 0.6, test_slice)
    assert os.path.exists(result), "extract_audio_segment failed"
    dur = get_audio_duration_sec(result)
    assert 0.4 < dur < 0.7, f"切片时长异常: {dur}"
    print(f"[OK] extract_audio_segment: {dur:.2f}s")

    # 测试 extract_audio_segment 边界校验
    try:
        extract_audio_segment(test_wav, 999.0, 1000.0, test_slice)
        assert False, "应抛出 ValueError"
    except ValueError:
        print(f"[OK] extract_audio_segment 超出时长校验正确")

    try:
        from moviepy import VideoFileClip  # noqa: F401
        print("[INFO] moviepy 可用")
    except ImportError:
        print("[INFO] moviepy 未安装 — 视频提取需: pip install moviepy")

    import shutil as _shutil
    _shutil.rmtree(tmp_dir, ignore_errors=True)
    print("\n[PASS] extractor tests passed")
