"""OpenAI 兼容 LLM 客户端：精修与摘要共用。

职责仅限于 HTTP 调用与 JSON 解析，不包含领域 prompt。
默认 DeepSeek（DEEPSEEK_API_KEY），可通过 config 切换到任意 OpenAI 兼容接口。
"""

import json
import logging
import os
import re
import time

import requests

from .utils import load_project_env

load_project_env()

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_TIMEOUT = 180
DEFAULT_MAX_TOKENS = 32768


def resolve_config(config: dict | None = None) -> dict:
    """合并 LLM 配置，None 值不覆盖默认值。"""
    cfg = {
        "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
        "api_url": DEFAULT_API_URL,
        "model": DEFAULT_MODEL,
        "timeout": DEFAULT_TIMEOUT,
        "enable_thinking": False,
        "max_output_tokens": DEFAULT_MAX_TOKENS,
    }
    if config:
        cfg.update({k: v for k, v in config.items() if v is not None})
    return cfg


def call_llm_api(
    api_key: str,
    api_url: str,
    model: str,
    prompt: str,
    timeout: int = DEFAULT_TIMEOUT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    enable_thinking: bool = False,
) -> str:
    """调用 LLM API，失败重试 3 次（5s/10s 退避）。返回空字符串表示失败。"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if enable_thinking:
        payload["thinking"] = {"type": "enabled"}

    for attempt in range(3):
        try:
            response = requests.post(
                api_url, headers=headers, json=payload, timeout=timeout
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"].strip()
            logger.warning(
                f"LLM API 返回 {response.status_code} (尝试 {attempt + 1}/3): "
                f"{response.text[:300]}"
            )
        except requests.Timeout:
            logger.warning(f"LLM API 超时 (尝试 {attempt + 1}/3)")
        except Exception as e:
            logger.warning(f"LLM API 异常 (尝试 {attempt + 1}/3): {e}")
        if attempt < 2:
            delay = 5 * (2 ** attempt)
            logger.info(f"等待 {delay}s 后重试...")
            time.sleep(delay)
    return ""


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _extract_json_block(text: str) -> str | None:
    """从混有解释文字的响应中提取第一个 JSON 对象/数组。"""
    for pattern in (r"\{[\s\S]*\}", r"\[[\s\S]*\]"):
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return None


def parse_json_output(raw: str):
    """解析 LLM 输出为 dict/list，失败返回 None。

    先按原样解析；失败时尝试修复被 JSON 视为非法转义的反斜杠
    （LLM 偶尔输出 LaTeX 或路径反斜杠）。
    """
    if not raw:
        return None
    text = _strip_code_fence(raw)
    candidates = [text]
    extracted = _extract_json_block(text)
    if extracted and extracted != text:
        candidates.append(extracted)

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    # 修复非 JSON 标准转义的单反斜杠后重试
    fixed = re.sub(r"(?<!\\)\\(?![\\/\"nt])", r"\\\\", text)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        return None
