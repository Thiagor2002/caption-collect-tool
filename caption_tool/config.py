"""路径与运行配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VIDEOS_DIR = REPO_ROOT / "test_videos"
OUTPUT_DIR = REPO_ROOT / "outputs"
ENV_FILE = REPO_ROOT / ".env"

VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".flv", ".webm", ".m4v", ".ts", ".mp3", ".wav", ".m4a", ".aac"}

DEFAULT_MODEL_SIZE = "small"
DEFAULT_LANGUAGE = "zh"
DEFAULT_COMPUTE_TYPE = "int8"


def load_env_file(path: Path = ENV_FILE) -> dict[str, str]:
    """读取 .env（不覆盖已存在的真实环境变量）。"""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip().strip("'\"")
        values[key] = value
        os.environ.setdefault(key, value)
    return values


@dataclass
class LLMConfig:
    """OpenAI 兼容接口的配置。"""

    api_key: str = ""
    base_url: str = ""
    model: str = ""
    temperature: float = 0.4
    max_tokens: int = 4096
    timeout: int = 180
    max_retries: int = 2

    @classmethod
    def from_env(cls) -> "LLMConfig":
        load_env_file()
        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        base_url = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        model = os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or ""
        return cls(
            api_key=api_key.strip(),
            base_url=base_url.strip().rstrip("/"),
            model=model.strip(),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.4")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
            timeout=int(os.getenv("LLM_TIMEOUT", "180")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
        )

    @property
    def available(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)

    def missing_fields(self) -> list[str]:
        missing = []
        if not self.api_key:
            missing.append("LLM_API_KEY")
        if not self.model:
            missing.append("LLM_MODEL")
        if not self.base_url:
            missing.append("LLM_BASE_URL")
        return missing
