"""YirenIngest 命令行入口。

用法：
    python -m yiren_ingest.run --series_id zhangchulan_guide --source ./videos/ep01.mp4
    python -m yiren_ingest.run --series_id test --source_dir ./videos/ --force_reprocess
    python -m yiren_ingest.run --series_id test --source ep01.mp4 --no-refine --no-summary
    python -m yiren_ingest.run --series_id test --source ep01.mp4 --asr_backend whisper --no-refine --no-summary
"""

import argparse
import json
import logging
import os
import sys
from glob import glob

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from yiren_ingest.utils import load_project_env

load_project_env()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("yiren.run")

_HW_DIR = os.path.dirname(os.path.abspath(__file__))
_HW_BASE = os.path.join(_HW_DIR, "hardware_config.yaml")
_HW_LOCAL = os.path.join(_HW_DIR, "hardware_config.local.yaml")

_SOURCE_EXTS = [
    "*.mp4", "*.mkv", "*.mov", "*.avi", "*.webm", "*.flv",
    "*.wav", "*.mp3", "*.m4a", "*.flac", "*.ogg", "*.aac",
]


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _load_hardware_config() -> dict:
    try:
        import yaml
    except ImportError:
        logger.debug("PyYAML 未安装，跳过硬件配置加载")
        return {}

    if not os.path.exists(_HW_BASE):
        return {}
    try:
        with open(_HW_BASE, "r", encoding="utf-8") as f:
            base = yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"硬件配置读取失败: {_HW_BASE}: {e}")
        return {}

    if os.path.exists(_HW_LOCAL):
        try:
            with open(_HW_LOCAL, "r", encoding="utf-8") as f:
                local = yaml.safe_load(f) or {}
            logger.info(f"已加载硬件配置 (含本地覆盖): {_HW_LOCAL}")
            return _deep_merge(base, local)
        except Exception as e:
            logger.warning(f"本地硬件配置读取失败: {_HW_LOCAL}: {e}")
    return base


def _hw_to_flat_config(hw: dict) -> dict:
    cfg: dict = {}
    asr = hw.get("asr", {}) or {}
    if asr.get("model"):
        cfg["asr_model"] = asr["model"]
    if asr.get("device") and asr["device"] != "auto":
        cfg["asr_device"] = asr["device"]
    if asr.get("compute_type") and asr["compute_type"] != "auto":
        cfg["asr_compute_type"] = asr["compute_type"]
    if asr.get("cpu_threads") is not None:
        cfg["asr_cpu_threads"] = asr["cpu_threads"]
    for key in ("whisper_workers", "funasr_workers", "cloud_workers"):
        if asr.get(key) is not None:
            cfg[key] = asr[key]
    refine = hw.get("refine", {}) or {}
    if refine.get("workers") is not None:
        cfg["postprocess_workers"] = refine["workers"]
    return cfg


def _collect_sources(args) -> list[str]:
    sources = list(args.source or [])
    if args.source_dir:
        for ext in _SOURCE_EXTS:
            sources.extend(
                glob(os.path.join(args.source_dir, "**", ext), recursive=True)
            )
    seen = set()
    unique = []
    for path in sources:
        normalized = os.path.abspath(path)
        if normalized not in seen:
            seen.add(normalized)
            unique.append(path)
    return unique


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="YirenIngest — 《异人之下》攻略视频摄取工具"
    )
    parser.add_argument("--series_id", type=str, required=True,
                        help="系列 ID，如 zhangchulan_guide")
    parser.add_argument("--source", type=str, nargs="+", default=[],
                        help="源视频/音频路径列表")
    parser.add_argument("--source_dir", type=str, default=None,
                        help="源目录（递归扫描 mp4/mkv/mov/wav/mp3 等）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出目录，默认 data/series/{series_id}")

    parser.add_argument("--refine", action=argparse.BooleanOptionalAction,
                        default=True, help="启用 LLM 文本精修（默认开启）")
    parser.add_argument("--summary", action=argparse.BooleanOptionalAction,
                        default=True, help="启用章节+全文总览（默认开启）")
    parser.add_argument("--refine_api", type=str, default=None,
                        help="LLM API 地址（默认 DeepSeek）")
    parser.add_argument("--refine_api_key", type=str, default=None,
                        help="LLM API Key（默认 DEEPSEEK_API_KEY）")
    parser.add_argument("--refine_model", type=str, default=None,
                        help="精修模型名（默认 deepseek-v4-pro）")
    parser.add_argument("--refine_timeout", type=int, default=None,
                        help="精修 API 超时秒数")
    parser.add_argument("--refine_enable_thinking", action="store_true",
                        default=False, help="启用 DeepSeek 思考模式")
    parser.add_argument("--refine_max_tokens", type=int, default=None,
                        help="精修最大输出 token 数")
    parser.add_argument("--summary_model", type=str, default=None,
                        help="摘要模型名（默认与精修相同）")
    parser.add_argument("--summary_timeout", type=int, default=None,
                        help="摘要 API 超时秒数")
    parser.add_argument("--summary_max_tokens", type=int, default=None,
                        help="摘要最大输出 token 数")

    parser.add_argument("--asr_backend", type=str, default="funasr",
                        choices=["auto", "whisper", "funasr", "cloud"],
                        help="ASR 引擎：funasr(默认)/whisper/cloud/auto")
    parser.add_argument("--asr_model", type=str, default=None,
                        help="Whisper 模型规格 large-v3/medium/small")
    parser.add_argument("--asr_device", type=str, default=None,
                        help="auto / cuda / cpu（默认 auto，由 hardware_config 控制）")
    parser.add_argument("--asr_compute_type", type=str, default=None,
                        choices=["auto", "float16", "float32", "int8", "int8_float16"])
    parser.add_argument("--asr_cpu_threads", type=int, default=None)
    parser.add_argument("--asr_cache_dir", type=str, default=None)
    parser.add_argument("--whisper_workers", type=int, default=None)
    parser.add_argument("--funasr_workers", type=int, default=None)
    parser.add_argument("--cloud_workers", type=int, default=None)
    parser.add_argument("--language", type=str, default="zh")
    parser.add_argument("--segment_max_duration", type=int, default=1800,
                        help="长音频切分阈值（秒）")
    parser.add_argument("--force_reprocess", action="store_true", default=False,
                        help="忽略幂等索引，强制重处理")
    parser.add_argument("--json_output", type=str, default=None,
                        help="把结果 JSON 写入指定文件")
    return parser


def _build_config(args, hw: dict) -> dict:
    config = _hw_to_flat_config(hw) if hw else {}
    config.update({
        "enable_refine": args.refine,
        "enable_summary": args.summary,
        "force_reprocess": args.force_reprocess,
        "segment_max_duration_sec": args.segment_max_duration,
        "language": args.language,
        "asr_backend": args.asr_backend,
    })
    if args.asr_cache_dir:
        config["asr_cache_dir"] = args.asr_cache_dir

    for key, value in (
        ("asr_model", args.asr_model),
        ("asr_device", args.asr_device),
        ("asr_compute_type", args.asr_compute_type),
        ("asr_cpu_threads", args.asr_cpu_threads),
        ("whisper_workers", args.whisper_workers),
        ("funasr_workers", args.funasr_workers),
        ("cloud_workers", args.cloud_workers),
    ):
        if value is not None:
            config[key] = value

    if args.refine:
        for key, value in (
            ("refine_api_url", args.refine_api),
            ("refine_api_key", args.refine_api_key),
            ("refine_model", args.refine_model),
            ("refine_timeout", args.refine_timeout),
            ("refine_max_output_tokens", args.refine_max_tokens),
        ):
            if value is not None:
                config[key] = value
        if args.refine_enable_thinking:
            config["refine_enable_thinking"] = True

    if args.summary:
        for key, value in (
            ("summary_model", args.summary_model),
            ("summary_timeout", args.summary_timeout),
            ("summary_max_output_tokens", args.summary_max_tokens),
        ):
            if value is not None:
                config[key] = value
    return config


def main():
    args = _build_parser().parse_args()

    source_paths = _collect_sources(args)
    if not source_paths:
        logger.error("未指定源文件（--source 或 --source_dir）")
        sys.exit(1)

    output_dir = args.output or os.path.join("data", "series", args.series_id)
    hw = _load_hardware_config()
    config = _build_config(args, hw)

    logger.info(f"系列 ID: {args.series_id}")
    logger.info(f"源文件数: {len(source_paths)}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"LLM 精修: {'启用' if args.refine else '关闭'} | "
                f"章节摘要: {'启用' if args.summary else '关闭'}")

    from yiren_ingest import ingest_series_material

    result = ingest_series_material(
        series_id=args.series_id,
        upload_paths=source_paths,
        output_dir=output_dir,
        config=config,
    )

    n_total = len(source_paths)
    n_success = len(result["transcripts"])
    n_failed = n_total - n_success
    stats = result.get("stats", {})

    print("\n" + "=" * 60)
    print(f"摄取结果: {result['status']}")
    print(f"系列: {result['series_id']}")
    print(f"成功: {n_success} / 总数: {n_total}")
    print(f"失败: {n_failed}")
    print(f"总时长: {stats.get('total_duration_sec', 0)}s | "
          f"段落: {stats.get('segments_count', 0)} | "
          f"章节: {stats.get('chapters_count', 0)}")
    print(f"精修文件: {stats.get('refined_files', 0)} | "
          f"摘要文件: {stats.get('summarized_files', 0)} | "
          f"est.CER: {stats.get('estimated_cer', 0)}")

    if result["transcripts"]:
        print("\nTranscripts:")
        for path in result["transcripts"]:
            print(f"  - {path}")

    if result.get("warnings"):
        print("\nWarnings:")
        for warning in result["warnings"]:
            print(f"  - {warning}")

    if args.json_output:
        os.makedirs(os.path.dirname(args.json_output) or ".", exist_ok=True)
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n结果已写入: {args.json_output}")

    sys.exit(0 if result["status"] != "error" else 1)


if __name__ == "__main__":
    main()
