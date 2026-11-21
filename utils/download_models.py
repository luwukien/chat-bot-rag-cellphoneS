"""Download the two RAG models before starting the web demo."""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--http", action="store_true", help="Use HTTP instead of Xet for downloads.")
    args = parser.parse_args()

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    if os.environ.get("HF_HUB_OFFLINE", "").upper() in {"1", "ON", "YES", "TRUE"}:
        parser.error("HF_HUB_OFFLINE is enabled. Unset it before downloading models.")
    if args.http:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    if os.name == "nt":
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

    # Import after configuring the environment; Hugging Face reads it at import time.
    from huggingface_hub import constants, snapshot_download
    from huggingface_hub.utils import enable_progress_bars

    enable_progress_bars()
    print(f"Model cache: {constants.HF_HUB_CACHE}", flush=True)
    print("Stop other uvicorn/download processes first to avoid waiting for the same cache lock.", flush=True)
    print("HF_TOKEN is optional for these public models. Keep this process running until Done.", flush=True)
    for index, model in enumerate(("keepitreal/vietnamese-sbert", "itdainb/PhoRanker"), start=1):
        print(f"\n[{index}/2] Downloading {model} ...", flush=True)
        path = snapshot_download(
            model,
            allow_patterns=["*.json", "*.bin", "*.safetensors", "*.txt", "*.codes", "README.md"],
            max_workers=2,
        )
        print(f"Cached: {path}", flush=True)
    print("\nDone. Both model snapshots are cached. You can now start uvicorn.", flush=True)


if __name__ == "__main__":
    main()
