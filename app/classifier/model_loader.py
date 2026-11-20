"""Lazy, process-wide loader for the existing local XLM-R model."""

from __future__ import annotations

import importlib
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGED_MAPPING_PATH = Path(__file__).with_name("intent_mapping.json")
MIN_LABEL_COUNT = 2


@dataclass(frozen=True)
class ModelBundle:
    tokenizer: Any
    model: Any
    id2label: dict[int, str]
    device: str
    torch: Any
    # Temperature-scaling factor fitted on validation data; 1.0 means uncalibrated.
    temperature: float = 1.0


_MODEL_CACHE: dict[tuple[str, str], ModelBundle] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def _configured_path(env_name: str, default: Path) -> Path:
    raw_value = os.getenv(env_name)
    path = Path(raw_value).expanduser() if raw_value else default
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def resolve_model_path() -> Path:
    default_model = PROJECT_ROOT / "models" / "xlmr_intent_classifier"
    return _configured_path("XLMR_MODEL_PATH", default_model)


def resolve_mapping_path(model_path: Path | None = None) -> Path:
    model_path = model_path or resolve_model_path()
    default_mapping = model_path / "label_mapping.json"
    if not default_mapping.is_file():
        default_mapping = PACKAGED_MAPPING_PATH
    return _configured_path("XLMR_LABEL_MAPPING_PATH", default_mapping)


def load_label_mapping(mapping_path: Path) -> dict[int, str]:
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Intent label mapping not found: {mapping_path}")

    with mapping_path.open("r", encoding="utf-8") as mapping_file:
        mapping = json.load(mapping_file)

    if "id2label" in mapping:
        id2label = {int(index): label for index, label in mapping["id2label"].items()}
    elif "label2id" in mapping:
        id2label = {int(index): label for label, index in mapping["label2id"].items()}
    else:
        raise ValueError("Mapping must contain id2label or label2id.")

    if len(id2label) < MIN_LABEL_COUNT or sorted(id2label) != list(range(len(id2label))):
        raise ValueError("Expected consecutive intent label indices starting at 0.")
    if len(set(id2label.values())) != len(id2label):
        raise ValueError("Intent labels must be unique.")

    return id2label


def _load_temperature(model_path: Path) -> float:
    path = model_path / "calibration.json"
    if not path.is_file():
        return 1.0
    temperature = float(json.loads(path.read_text(encoding="utf-8")).get("temperature", 1.0))
    if not 0.05 <= temperature <= 20:
        raise ValueError(f"Implausible calibration temperature {temperature} in {path}.")
    return temperature


def load_model(
    model_path: str | Path | None = None,
    mapping_path: str | Path | None = None,
) -> ModelBundle:
    """Load the tokenizer and classifier once per model/mapping path."""
    resolved_model = (
        Path(model_path).expanduser().resolve() if model_path else resolve_model_path()
    )
    resolved_mapping = (
        Path(mapping_path).expanduser().resolve()
        if mapping_path
        else resolve_mapping_path(resolved_model)
    )
    cache_key = (str(resolved_model), str(resolved_mapping))

    with _MODEL_CACHE_LOCK:
        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key]

        if not resolved_model.is_dir():
            raise FileNotFoundError(
                f"Bundled XLM-R model directory not found: {resolved_model}. "
                "Restore the model bundle under models/xlmr_intent_classifier."
            )

        required_files = (
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
        )
        missing_files = [name for name in required_files if not (resolved_model / name).is_file()]
        if missing_files:
            missing = ", ".join(missing_files)
            raise FileNotFoundError(
                f"XLM-R inference files are missing from {resolved_model}: {missing}. "
                "Restore the Git LFS model files or configure XLMR_MODEL_PATH."
            )

        id2label = load_label_mapping(resolved_mapping)
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(resolved_model), local_files_only=True
        )
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            str(resolved_model), local_files_only=True
        )
        model.to(device)
        model.eval()

        bundle = ModelBundle(
            tokenizer=tokenizer,
            model=model,
            id2label=id2label,
            device=device,
            torch=torch,
            temperature=_load_temperature(resolved_model),
        )
        _MODEL_CACHE[cache_key] = bundle
        return bundle


def clear_model_cache() -> None:
    """Clear the loader cache; primarily useful for isolated tests."""
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.clear()
