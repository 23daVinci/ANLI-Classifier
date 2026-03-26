"""
Download the pre-trained DeBERTa-v3-base NLI model weights.

Run this once after cloning the repository:
    python download_model.py

The model will be saved to ./best_model/ (~360MB).
"""

import os

def main():
    output_dir = "./best_model"

    if os.path.exists(os.path.join(output_dir, "model.safetensors")):
        print(f"Model already exists at {output_dir}/")
        print("Delete the folder and re-run if you want to re-download.")
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Installing huggingface_hub...")
        os.system("pip install huggingface_hub")
        from huggingface_hub import snapshot_download

    print("Downloading DeBERTa-v3-base-mnli-fever-anli...")
    print("This may take a few minutes (~360MB).\n")

    snapshot_download(
        repo_id="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        local_dir=output_dir,
    )

    print(f"\nModel downloaded to {output_dir}/")
    print("You can now run: docker compose up --build")

if __name__ == "__main__":
    main()