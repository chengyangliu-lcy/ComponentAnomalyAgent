from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _resolve_path(value: str | Path | int) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return ROOT / path


@dataclass(frozen=True)
class RuntimeConfig:
    raw: Dict[str, Any]

    @property
    def dataset_path(self) -> Path:
        return _resolve_path(self.raw["paths"]["dataset"])

    @property
    def image_root(self) -> Path:
        return _resolve_path(self.raw["paths"]["image_root"])

    @property
    def local_corpus_root(self) -> Path:
        return _resolve_path(self.raw["paths"]["local_corpus_root"])

    @property
    def knowledge_csv(self) -> Path:
        return _resolve_path(self.raw["paths"]["knowledge_csv"])

    @property
    def warc_root(self) -> Path:
        return _resolve_path(self.raw["paths"]["warc_root"])

    @property
    def local_kb_index(self) -> Path:
        return _resolve_path(self.raw["paths"]["local_kb_index"])

    @property
    def outputs_dir(self) -> Path:
        return _resolve_path(self.raw["paths"]["outputs_dir"])

    @property
    def logs_dir(self) -> Path:
        return _resolve_path(self.raw["paths"]["logs_dir"])

    @property
    def api_key(self) -> str | None:
        direct_key = self.raw["model"].get("api_key")
        if direct_key:
            return str(direct_key)
        env_name = self.raw["model"].get("api_key_env", "DASHSCOPE_API_KEY")
        return os.environ.get(env_name)

    @property
    def base_url(self) -> str:
        env_name = self.raw["model"].get("base_url_env", "DASHSCOPE_BASE_URL")
        return os.environ.get(env_name) or self.raw["model"]["default_base_url"]

    @property
    def agent_model(self) -> str:
        env_name = self.raw["model"].get("agent_model_env", "AGENT_MODEL")
        return os.environ.get(env_name) or self.raw["model"]["default_agent_model"]

    @property
    def vision_model(self) -> str:
        env_name = self.raw["model"].get("vision_model_env", "VISION_MODEL")
        return os.environ.get(env_name) or self.raw["model"].get("default_vision_model") or self.agent_model

    @property
    def judge_model(self) -> str:
        env_name = self.raw["model"].get("judge_model_env", "JUDGE_MODEL")
        return os.environ.get(env_name) or self.raw["model"]["default_judge_model"]

    def ensure_dirs(self) -> None:
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(path: str | Path | None = None, overrides: Dict[str, Any] | None = None) -> RuntimeConfig:
    default_path = ROOT / "configs" / "default.yaml"
    raw: Dict[str, Any] = _read_yaml(default_path)
    local_path = ROOT / "configs" / "local.yaml"
    if local_path.exists():
        raw = _deep_update(raw, _read_yaml(local_path))
    if path:
        config_path = Path(path)
        if not config_path.is_absolute():
            config_path = ROOT / config_path
        raw = _deep_update(raw, _read_yaml(config_path))
    if overrides:
        raw = _deep_update(raw, overrides)
    return RuntimeConfig(raw=raw)
