#!/usr/bin/env bash
# Deploy the Docker app to a Hugging Face Space and the weights to a model repo.
# Prereqs: pip install huggingface_hub ; huggingface-cli login (token with write access)
# Usage:   HF_USER=your-name bash deploy/push_to_hf_space.sh
set -euo pipefail
HF_USER=${HF_USER:?set HF_USER}
SPACE=${SPACE:-roadvision}
WEIGHTS_REPO=${WEIGHTS_REPO:-roadvision-weights}

python - <<PY
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("$HF_USER/$WEIGHTS_REPO", repo_type="model", exist_ok=True)
for f in ["models/yolo_best.pt", "models/rtdetr_best.pt", "models/vit_best.pt", "models/hybrid_calibration.json"]:
    import os
    if os.path.exists(f):
        api.upload_file(path_or_fileobj=f, path_in_repo=os.path.basename(f), repo_id="$HF_USER/$WEIGHTS_REPO")
        print("uploaded", f)
api.create_repo("$HF_USER/$SPACE", repo_type="space", space_sdk="docker", exist_ok=True)
api.add_space_variable("$HF_USER/$SPACE", "RV_WEIGHTS_REPO", "$HF_USER/$WEIGHTS_REPO")
api.upload_folder(repo_id="$HF_USER/$SPACE", repo_type="space", folder_path=".",
                  allow_patterns=["Dockerfile", "requirements-serve.txt", "configs/**", "src/**", "api/**", "app/**",
                                  "models/.gitkeep"],
                  ignore_patterns=["**/__pycache__/**"])
api.upload_file(path_or_fileobj="deploy/hf_space/README.md", path_in_repo="README.md", repo_id="$HF_USER/$SPACE", repo_type="space")
print("Space: https://huggingface.co/spaces/$HF_USER/$SPACE")
PY
