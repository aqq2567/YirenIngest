"""Yiren 模块通用工具函数。"""
import os
import json
import random
import string
from datetime import datetime, timezone
from pathlib import Path


def load_project_env() -> None:
    """从项目根目录加载 .env 到 os.environ（仅设置尚未存在的变量）。

    所有需要 API Key 的子模块应在模块级调用此函数一次。
    优先使用 python-dotenv，不可用时回退为手动解析。
    """
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).resolve().parent.parent / ".env"
        load_dotenv(env_path)
        return
    except ImportError:
        pass

    # 回退：手动解析 .env（不依赖第三方库）
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.is_file():
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def atomic_write_json(path: str, data: dict, indent: int = 2) -> None:
    """原子写入 JSON：先写 .tmp 文件，完成后 os.replace 到目标路径。

    防止并发读取时读到半截文件。
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    os.replace(tmp_path, path)


def utc_timestamp() -> str:
    """UTC 时间戳字符串 YYYYMMDDHHMMSS。"""
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def random_suffix(length: int = 3) -> str:
    """随机小写字母后缀。"""
    return "".join(random.choices(string.ascii_lowercase, k=length))
