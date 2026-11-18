"""Optional, strictly validated YAML settings; CLI options take precedence."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from .common import DEFAULT_STREAM_LIMIT, MAX_WINDOW, bounded_float, bounded_int, stream_limit
from .detector import DEFAULT_CRITICAL, DEFAULT_CRITICAL_PROBABILITY, DEFAULT_PAIR_SPAN, critical_catalog
from .markov_model import MAX_ORDER, smoothing


@dataclass(frozen=True, slots=True)
class Settings:
    order: int | None = None
    alpha: float | None = None
    window: int | None = None
    percentile: float | None = None
    max_streams: int = DEFAULT_STREAM_LIMIT
    critical_tokens: tuple[str, ...] = DEFAULT_CRITICAL
    critical_probability: float = DEFAULT_CRITICAL_PROBABILITY
    critical_pair_span: int = DEFAULT_PAIR_SPAN


def load_settings(path: Path | None) -> Settings:
    if path is None:
        return Settings()
    try:
        with path.open(encoding="utf-8") as handle:
            text = handle.read((1 << 16) + 1)
        if len(text) > 1 << 16:
            raise ValueError("configuration exceeds 64 KiB")
        data = yaml.safe_load(text)
        if data is None:
            return Settings()
        if not isinstance(data, dict) or not set(data).issubset(Settings.__dataclass_fields__):
            raise ValueError("configuration must be a mapping with documented keys only")
        validators = {
            "order": lambda v: bounded_int(v, "order", 1, MAX_ORDER),
            "alpha": smoothing,
            "window": lambda v: bounded_int(v, "window", 2, MAX_WINDOW),
            "percentile": lambda v: bounded_float(v, "percentile", 0, 100),
            "max_streams": stream_limit,
            "critical_probability": lambda v: bounded_float(v, "critical_probability", 0, 1),
            "critical_pair_span": lambda v: bounded_int(v, "critical_pair_span", 0, MAX_WINDOW),
        }
        validated = {key: validators[key](value) for key, value in data.items() if key != "critical_tokens"}
        if "critical_tokens" in data:
            if not isinstance(data["critical_tokens"], list):
                raise ValueError("critical_tokens must be a YAML list")
            validated["critical_tokens"] = tuple(sorted(critical_catalog(data["critical_tokens"])))
        return Settings(**validated)
    except (ValueError, yaml.YAMLError, RecursionError) as exc:
        raise ValueError(f"{path}: {exc}") from exc
