"""第二 LLM 阶段：为精修后的 transcript 生成章节摘要与全文总览。

输出直接写回 transcript JSON：
  overview: {title, summary, key_points}
  chapters: [{chapter_id, title, start, end, summary, key_points}]

设计为独立阶段：失败时保留已有 transcript，不影响 ASR/精修结果。
"""

import json
import logging
import os

from ..domain_terms import LLM_TERM_REFERENCE
from ..llm_client import call_llm_api, parse_json_output, resolve_config
from ..utils import atomic_write_json

logger = logging.getLogger(__name__)

# 超长输入保护：超过该字符数时不调用 LLM，保留 transcript 并给出告警
MAX_SUMMARY_INPUT_CHARS = 120_000


def _build_summary_prompt(segments: list[dict], language: str) -> str:
    input_segments = [
        {
            "id": seg.get("segment_id", ""),
            "start": seg.get("start", 0.0),
            "end": seg.get("end", 0.0),
            "text": seg.get("text", ""),
        }
        for seg in segments
    ]
    input_json = json.dumps(input_segments, ensure_ascii=False, indent=2)
    lang_hint = "中文" if language == "zh" else language

    return f"""# Role
你是一位《异人之下》PVP 对战游戏的攻略视频内容编辑。请基于下面的视频转写，输出全文总览和章节摘要。

# 领域术语参考
{LLM_TERM_REFERENCE}

# Task
1. **全文总览**：
   - title：视频标题，不超过 20 字
   - summary：150-300 字，概括视频讲了什么、适合谁看
   - key_points：3-8 条全文要点
2. **章节切分**：按主题把视频切成 2-10 章（视频短于 3 分钟可 1-2 章）。
   - 每章必须有 title、start、end、summary、key_points
   - start/end 必须从输入的 segment 时间戳中选取，按时间升序、首尾相接、覆盖全部内容
   - summary 50-150 字；key_points 2-6 条
3. **只基于转写内容**，不要臆造视频中没有的信息。
4. 术语必须准确（炁、险招、垫步、身外身、强制脱出等），角色名/技能名按领域参考修正。

# Output Format
严格输出 JSON 对象（不要 markdown 代码块）：
{{
  "title": "{lang_hint}标题",
  "summary": "全文摘要...",
  "key_points": ["要点1", "要点2"],
  "chapters": [
    {{
      "title": "章节标题",
      "start": 0.0,
      "end": 60.0,
      "summary": "章节摘要...",
      "key_points": ["要点1", "要点2"]
    }}
  ]
}}

原始 segments（共 {len(segments)} 段）：
{input_json}"""


def _normalize_key_points(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _normalize_chapters(parsed: dict, video_start: float, video_end: float) -> list[dict]:
    raw_chapters = parsed.get("chapters", [])
    if not isinstance(raw_chapters, list):
        return []

    chapters = []
    for item in raw_chapters:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        summary = str(item.get("summary", "")).strip()
        if not title or not summary:
            continue
        try:
            start = round(float(item.get("start", 0.0)), 2)
            end = round(float(item.get("end", 0.0)), 2)
        except (TypeError, ValueError):
            continue
        start = max(video_start, min(start, video_end))
        end = max(start, min(end, video_end))
        if end <= start:
            continue
        chapters.append({
            "title": title,
            "start": start,
            "end": end,
            "summary": summary,
            "key_points": _normalize_key_points(item.get("key_points")),
        })

    chapters.sort(key=lambda c: c["start"])
    for idx, chapter in enumerate(chapters, 1):
        chapter["chapter_id"] = f"ch_{idx:03d}"
    # chapter_id 放最前，保持输出字段顺序稳定
    return [
        {
            "chapter_id": c["chapter_id"],
            "title": c["title"],
            "start": c["start"],
            "end": c["end"],
            "summary": c["summary"],
            "key_points": c["key_points"],
        }
        for c in chapters
    ]


def summarize_transcript(transcript_path: str, config: dict | None = None) -> dict:
    """为 transcript 生成 overview + chapters，成功时覆盖写回原文件。

    Args:
        transcript_path: transcript JSON 路径
        config: LLM 配置（api_key/api_url/model/timeout/enable_thinking/
                max_output_tokens），None 时读 DEEPSEEK_API_KEY

    Returns:
        {"status": "success|error", "chapters_count": N, "message": ...}
    """
    cfg = resolve_config(config)
    if not cfg["api_key"]:
        return {
            "status": "error",
            "message": "未设置 DEEPSEEK_API_KEY，跳过章节摘要",
        }

    if not os.path.exists(transcript_path):
        return {"status": "error", "message": f"transcript 文件不存在: {transcript_path}"}

    with open(transcript_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    if not segments:
        return {"status": "error", "message": "transcript 中没有 segments，跳过摘要"}

    total_chars = sum(len(str(seg.get("text", ""))) for seg in segments)
    if total_chars > MAX_SUMMARY_INPUT_CHARS:
        return {
            "status": "error",
            "message": f"转写文本 {total_chars} 字超过摘要上限 {MAX_SUMMARY_INPUT_CHARS}，已保留转写结果",
        }

    prompt = _build_summary_prompt(segments, data.get("language", "zh"))
    logger.info(
        f"开始章节摘要: model={cfg['model']}, segments={len(segments)}, "
        f"chars={total_chars}, prompt_length={len(prompt)}"
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
        return {"status": "error", "message": "摘要 LLM API 调用失败（3 次重试后仍无响应）"}

    parsed = parse_json_output(raw_response)
    if not isinstance(parsed, dict):
        debug_path = transcript_path.replace(".json", "_summary_llm_raw.txt")
        try:
            with open(debug_path, "w", encoding="utf-8") as df:
                df.write(raw_response)
            logger.error(f"摘要输出无法解析为 JSON，原始响应已保存: {debug_path}")
        except Exception:
            logger.warning("写入摘要调试文件失败")
        return {"status": "error", "message": "摘要 LLM 输出无法解析为 JSON 对象"}

    video_start = min(float(seg.get("start", 0.0)) for seg in segments)
    video_end = max(float(seg.get("end", 0.0)) for seg in segments)
    chapters = _normalize_chapters(parsed, video_start, video_end)

    title = str(parsed.get("title", "")).strip()
    summary = str(parsed.get("summary", "")).strip()
    key_points = _normalize_key_points(parsed.get("key_points"))

    if not title and not summary and not chapters:
        return {"status": "error", "message": "摘要结果为空，保留原 transcript"}

    data["overview"] = {
        "title": title,
        "summary": summary,
        "key_points": key_points,
    }
    data["chapters"] = chapters
    atomic_write_json(transcript_path, data)

    logger.info(f"章节摘要完成: {len(chapters)} 章, 输出 {transcript_path}")
    return {
        "status": "success",
        "chapters_count": len(chapters),
        "transcript_path": transcript_path,
    }
