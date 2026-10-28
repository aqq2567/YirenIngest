"""文本规则清洗：语言层通用过滤，不依赖 LLM，无领域知识。

处理内容：
1. 口语填充词删除
2. 空白/标点规范化
3. 空段过滤

领域术语纠错由 cleaner.py 的 LLM 精修完成。
"""

import logging
import re

logger = logging.getLogger(__name__)

# 口语填充词（语言特征，非领域依赖，可直接删除）
FILLER_WORDS_RE = re.compile(
    r"(那个|那么|就是说|的话呢|所以呢|这个呢|那个呢|然后呢|好吧|对不对|是不是|知道吧|明白吗"
    r"|就是说呢|然后就是说|那么就是说|这个就是说|那个就是说|嗯|啊|呃|嘛)",
)

_MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """对单段文本执行语言层通用清洗（无领域知识）。"""
    text = FILLER_WORDS_RE.sub("", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def clean_segments(segments: list[dict]) -> list[dict]:
    """清洗 segments。

    只丢弃清洗后为空的段：游戏解说里的「平A」「走」「对」等短促口令
    是有效信息，不能按讲义场景的「<4 字」规则过滤。
    """
    cleaned = []
    for seg in segments:
        original = seg.get("text", "")
        cleaned_text = clean_text(original)
        if not cleaned_text:
            logger.debug(f"跳过空段: {seg.get('segment_id')}")
            continue
        cleaned_seg = dict(seg)
        cleaned_seg["text"] = cleaned_text
        cleaned.append(cleaned_seg)

    removed = len(segments) - len(cleaned)
    if removed:
        logger.info(f"规则清洗: {len(segments)} → {len(cleaned)} 段 (过滤 {removed} 段)")
    return cleaned
