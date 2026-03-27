"""
Download DeBERTa-v3 NLI model weights.

Usage:
    python download_model.py              # Download base model only
    python download_model.py --model large # Download large model only
    python download_model.py --all         # Download both models
"""

import os
import argparse

MODELS = {
    "base": {
        "repo_id": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        "dir": "./best_model/base",
        "size": "~360MB",
        "description": "DeBERTa-v3-base (86M params) — fast, 54.6% on ANLI R2",
    },
    "large": {
        "repo_id": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        "dir": "./best_model/large",
        "size": "~1.2GB",
        "description": "DeBERTa-v3-large (304M params) — higher accuracy, ~58-62% on ANLI R2",
    },
}


def download(model_key):
    info = MODELS[model_key]

    if os.path.exists(os.path.join(info["dir"], "model.safetensors")):
        print(f"[{model_key}] Already exists at {info['dir']}/")
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Installing huggingface_hub...")
        os.system("pip install huggingface_hub")
        from huggingface_hub import snapshot_download

    print(f"\n[{model_key}] Downloading {info['description']}")
    print(f"  Source: {info['repo_id']}")
    print(f"  Size:   {info['size']}")
    print(f"  Dest:   {info['dir']}")
    print()

    snapshot_download(
        repo_id=info["repo_id"],
        local_dir=info["dir"],
    )

    print(f"\n[{model_key}] Downloaded to {info['dir']}/")


def main():
    parser = argparse.ArgumentParser(description="Download NLI model weights")
    parser.add_argument(
        "--model", choices=["base", "large"], default="base",
        help="Which model to download (default: base)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Download both base and large models"
    )
    args = parser.parse_args()

    if args.all:
        for key in MODELS:
            download(key)
    else:
        download(args.model)

    print("\nDone! Start the server with: docker compose up --build")


if __name__ == "__main__":
    main()