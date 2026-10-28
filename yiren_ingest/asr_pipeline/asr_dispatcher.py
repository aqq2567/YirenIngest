"""Yiren ASR 调度器：按优先级自动选择可用引擎，失败自动降级。

策略（config.asr_backend）：
    "auto"    — whisper → funasr → cloud 依次尝试（默认）
    "whisper" — 仅用本地 Whisper
    "funasr"  — 仅用本地 FunASR
    "cloud"   — 仅用云端 DashScope ASR

所有引擎 share 同一个 config dict，不识别的键被各引擎静默忽略。
"""

import logging
from .config import get_engine_config

logger = logging.getLogger(__name__)

# 引擎优先级（auto 模式下按此顺序尝试）
_AUTO_PRIORITY = ["funasr", "whisper", "cloud"]


def _get_available_engine(engine_name: str):
    """惰性导入并检查引擎是否可用。返回 (transcribe_fn, is_available: bool)。"""
    if engine_name == "whisper":
        from .whisper_runner import transcribe_audio, _check_whisper
        return transcribe_audio, _check_whisper()
    elif engine_name == "funasr":
        from .funasr_runner import transcribe_audio, _check_funasr
        return transcribe_audio, _check_funasr()
    elif engine_name == "cloud":
        from .cloud_asr_runner import transcribe_audio, _check_openai
        return transcribe_audio, _check_openai()
    else:
        raise ValueError(f"未知 ASR 引擎: {engine_name}")


def transcribe_audio(
    audio_path: str,
    transcript_id: str,
    series_id: str,
    source_file: str,
    config: dict | None = None,
) -> dict:
    """自动选择可用 ASR 引擎并转写，失败自动降级。

    config 键（调度器自身使用，其余透传）:
        asr_backend: "auto"（默认）/ "whisper" / "funasr" / "cloud"
    """
    cfg = config or {}
    backend = cfg.get("asr_backend", "auto")
    if backend not in ("auto", "whisper", "funasr", "cloud"):
        logger.warning(f"未知 asr_backend={backend}，回退为 auto")
        backend = "auto"

    # 确定尝试顺序
    if backend == "auto":
        candidates = list(_AUTO_PRIORITY)
    else:
        candidates = [backend]

    errors = []
    tried = []

    for engine_name in candidates:
        try:
            fn, available = _get_available_engine(engine_name)
        except Exception as e:
            errors.append(f"{engine_name}: 导入失败 — {e}")
            continue

        if not available:
            logger.info(f"跳过 {engine_name}（依赖未安装）")
            continue

        tried.append(engine_name)
        logger.info(f"尝试 {engine_name} 引擎: {audio_path}")

        try:
            result = fn(audio_path, transcript_id, series_id, source_file, config)
            logger.info(f"{engine_name} 转写成功: {len(result.get('segments', []))} 段")
            return result
        except Exception as e:
            msg = f"{engine_name}: {e}"
            logger.warning(f"{engine_name} 转写失败: {e}")
            errors.append(msg)
            continue

    # 所有引擎均失败
    tried_str = ", ".join(tried) if tried else "无"
    errors_str = "; ".join(errors) if errors else "无可用引擎"
    raise RuntimeError(
        f"ASR 转写失败 — 已尝试: [{tried_str}]，错误: {errors_str}"
    )


# ============================================================
# 冒烟测试
# ============================================================
if __name__ == "__main__":
    print("=== asr_dispatcher 冒烟测试 ===\n")

    # 测试 1：指定不存在文件的转写 —— 验证调度器能正确传递错误
    try:
        transcribe_audio(
            "nonexistent_file.wav", "TR_TEST_001", "T_TEST_001", "test.mp4",
        )
        assert False, "应抛出异常"
    except RuntimeError as e:
        print(f"[OK] 全引擎失败正确抛出 RuntimeError")

    # 测试 2：验证 asr_backend 配置生效
    from .config import get_engine_config
    cfg = get_engine_config("whisper", {"asr_backend": "whisper"})
    assert cfg.get("asr_backend") == "whisper", "asr_backend 应透传"
    print(f"[OK] asr_backend 配置透传正确")

    # 测试 3：未知 backend 不抛异常（回退 auto）
    try:
        transcribe_audio(
            "nonexistent.wav", "TR_TEST_002", "T_TEST_001", "test.mp4",
            config={"asr_backend": "unknown_backend"}
        )
    except RuntimeError:
        pass  # 预期：回退 auto 后全引擎失败
    print(f"[OK] 未知 backend 回退 auto 正确")

    print("\n[PASS]")
