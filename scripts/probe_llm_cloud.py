"""Record bounded Qwen/GGUF compatibility metadata without training weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from warehouse_ai.compatibility import (
    LLM_BASE_MODEL_ID,
    LLM_GGUF_FILENAME,
    LLM_GGUF_REPOSITORY,
    LLM_GGUF_REVISION,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=LLM_BASE_MODEL_ID)
    parser.add_argument("--gguf-repository", default=LLM_GGUF_REPOSITORY)
    parser.add_argument("--gguf-filename", default=LLM_GGUF_FILENAME)
    parser.add_argument("--revision", default=LLM_GGUF_REVISION)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_sha256 = None
    model_present = False
    if args.model_path is not None:
        model_present = args.model_path.is_file()
        if model_present:
            digest = hashlib.sha256()
            with args.model_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            model_sha256 = digest.hexdigest()
    summary = {
        "schema_version": 2,
        "model_id": args.model_id,
        "gguf_repository": args.gguf_repository,
        "gguf_filename": args.gguf_filename,
        "revision": args.revision,
        "model_path_present": model_present,
        "model_file_sha256": model_sha256,
        "runtime_contract": "llama.cpp-openai-json-schema",
        "qwen_transformers_lora_verified": False,
        "reasoning_enabled": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
