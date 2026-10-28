"""LLM 文本精修：把《异人之下》攻略视频的 ASR 转写修正为可读文本。

管线：rules.py（语言层通用清洗）→ cleaner.py（游戏领域 LLM 精修）
- 保留原始口播信息量，不扩写、不总结
- 重点修复同音术语错误（炁/气、险招/显招、垫步/电步等）
- 合并碎片化短句为连贯段落，保留时间戳

统一入口 refine()：读取 transcript JSON → 全量送入 LLM → 覆盖写回。
"""

import json
import logging
import os

from ..domain_terms import LLM_TERM_REFERENCE, TERM_CORRECTIONS
from ..llm_client import call_llm_api, parse_json_output, resolve_config
from ..utils import atomic_write_json
from . import rules

logger = logging.getLogger(__name__)

_CORRECTIONS_TEXT = "；".join(
    f"{correct}（不要写成：{'、'.join(wrong)}）"
    for correct, wrong in TERM_CORRECTIONS.items()
)


def _build_asr_refine_prompt(segments: list[dict], language: str) -> str:
    """构建全文精修 prompt，要求 LLM 输出 JSON 数组。"""
    input_segments = [
        {
            "id": seg["segment_id"],
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
        }
        for seg in segments
    ]
    input_json = json.dumps(input_segments, ensure_ascii=False, indent=2)
    lang_hint = "中文" if language == "zh" else language

    return f"""# Role
你是一位《异人之下》PVP 对战游戏的资深内容编辑，负责把攻略视频的 ASR 转写稿修正为准确、可读的攻略文本。

# Critical: ASR 同音错字修正（最高优先级）
输入来自中文 ASR，必然包含同音/近音错字，尤其是游戏术语和角色名。
你必须依据《异人之下》PVP 领域知识逐句检查并修正。

【高频易错词】
{_CORRECTIONS_TEXT}

【术语参考】
{LLM_TERM_REFERENCE}

# Task
1. **术语纠错（最优先）**：修正同音错字、角色名、技能名、机制名，使其符合《异人之下》游戏语境。
2. **保持原意**：不要扩写、不要删减信息、不要加入视频中没有的内容；不要写成总结或攻略教程，只做纠错与断句整理。
3. **合并碎片**：把相邻的 5-15 个零散短句合并为逻辑连贯的段落；战斗口令（如「垫步」「防反」「平A」）可以保留短句形态。
4. **保留时间戳**：合并后的段落 start 取首段 start，end 取末段 end。
5. 数字、按键（Q/E/R）、连招段数（4A/5A）保持原样。

# Output Format
严格输出 JSON 数组（不要外层对象，不要 markdown 代码块）：
[
  {{
    "id": "seg_0001",
    "start": 0.0,
    "end": 12.5,
    "text": "精修后的{lang_hint}段落..."
  }},
  ...
]

原始 segments（共 {len(segments)} 段）：
{input_json}"""


def _refine_asr(
    transcript_json_path: str,
    output_path: str,
    config: dict | None = None,
) -> dict:
    """对一份 transcript 执行 ASR 文本精修，覆盖写回。"""
    cfg = resolve_config(config)
    if not cfg["api_key"]:
        return {
            "status": "error",
            "message": "未设置 DEEPSEEK_API_KEY 环境变量，且在 config 中未提供 api_key",
        }

    if not os.path.exists(transcript_json_path):
        return {"status": "error", "message": f"transcript 文件不存在: {transcript_json_path}"}

    with open(transcript_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_segments = data.get("segments", [])
    if not raw_segments:
        logger.warning(f"transcript 中没有 segments: {transcript_json_path}")
        return {"status": "error", "message": "transcript 中没有 segments"}

    cleaned_segments = rules.clean_segments(raw_segments)
    logger.info(f"规则清洗完成: {len(raw_segments)} → {len(cleaned_segments)} segments")

    prompt = _build_asr_refine_prompt(cleaned_segments, data.get("language", "zh"))
    logger.info(
        f"开始 ASR LLM 精修: model={cfg['model']}, "
        f"segments={len(cleaned_segments)}, prompt_length={len(prompt)}"
    )

    raw_response = call_llm_api(
        api_key=cfg["api_key"],
        api_url=cfg["api_url"],
        model=cfg["model"],
        prompt=prompt,
        timeout=cfg["timeout"],
        max_tokens=cfg["max_output_tokens"],
        enable_thinking=cfg.get("enable_thinking", False),
    )
    if not raw_response:
        return {"status": "error", "message": "LLM API 调用失败（3 次重试后仍无响应）"}

    parsed = parse_json_output(raw_response)
    if not isinstance(parsed, list):
        logger.error(f"无法解析 LLM 输出为 JSON 数组 (长度={len(raw_response)} 字符)")
        debug_path = output_path.replace(".json", "_refine_llm_raw.txt")
        try:
            os.makedirs(os.path.dirname(debug_path) or ".", exist_ok=True)
            with open(debug_path, "w", encoding="utf-8") as df:
                df.write(raw_response)
            logger.error(f"完整 LLM 原始响应已保存到: {debug_path}")
        except Exception:
            logger.warning("写入 LLM 调试文件失败，继续处理")
        return {
            "status": "error",
            "message": "LLM 输出无法解析为 JSON 数组",
            "debug_file": debug_path,
        }

    output_segments = []
    for i, item in enumerate(parsed):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        output_segments.append({
            "segment_id": item.get("id") or f"seg_{i + 1:04d}",
            "start": round(float(item.get("start", 0)), 2),
            "end": round(float(item.get("end", 0)), 2),
            "text": text,
            "speaker": "narrator",
            "confidence": 0.85,  # LLM 精修文本不可直接量化，使用经验默认值
        })

    if not output_segments:
        return {"status": "error", "message": "LLM 精修后没有有效 segments"}

    avg_conf = sum(s["confidence"] for s in output_segments) / len(output_segments)
    low_count = sum(1 for s in output_segments if s["confidence"] < 0.7)

    output_data = {
        "transcript_id": data.get("transcript_id", "unknown"),
        "series_id": data.get("series_id", ""),
        "source_file": data.get("source_file", ""),
        "source_audio": data.get("source_audio", ""),
        "language": data.get("language", "zh"),
        "asr_backend": data.get("asr_backend", ""),
        "asr_quality": {
            "estimated_cer": round(max(0.0, min(1.0, 1.0 - avg_conf)), 4),
            "low_confidence_segments": low_count,
            "refined": True,
        },
        "ingested_at": data.get("ingested_at", ""),
    }
    # 重复精修时保留已有摘要结果
    if "overview" in data:
        output_data["overview"] = data["overview"]
    if "chapters" in data:
        output_data["chapters"] = data["chapters"]
    output_data["segments"] = output_segments

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    atomic_write_json(output_path, output_data)

    logger.info(f"ASR 精修完成: {len(output_segments)} segments, 输出 {output_path}")
    return {
        "status": "success",
        "refined_path": output_path,
        "segments_count": len(output_segments),
        "original_segments": len(raw_segments),
    }


def refine(input_path: str, output_path: str, config: dict | None = None) -> dict:
    """统一精修入口（当前只有 ASR transcript 一种输入）。"""
    return _refine_asr(input_path, output_path, config)


# 向后兼容别名
refine_transcript = _refine_asr
