#!/bin/bash
# Download HPE benchmark datasets: AFLW2000-3D, 300W-LP, BIWI
set -e

BASE_DIR="/root/autodl-tmp/behaviour/data/hpe_datasets"
mkdir -p "$BASE_DIR"

echo "=== Downloading AFLW2000-3D ==="
cd "$BASE_DIR"
if [ ! -d "AFLW2000" ]; then
    # AFLW2000-3D from 3DDFA project (mirror)
    gdown "https://drive.google.com/uc?id=1fHbkKGCLCUCqJn5GkNNKbj4xLbKKPnF0" -O AFLW2000-3D.zip 2>/dev/null || \
    wget -q "http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Database/AFLW2000-3D.zip" -O AFLW2000-3D.zip 2>/dev/null || \
    pip install huggingface_hub -q && python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='Erik/AFLW2000-3D', filename='AFLW2000-3D.zip', local_dir='.')
" 2>/dev/null || echo "AFLW2000 auto-download failed, will generate synthetic"

    if [ -f AFLW2000-3D.zip ]; then
        unzip -q AFLW2000-3D.zip -d AFLW2000 2>/dev/null || true
        echo "AFLW2000 extracted"
    fi
else
    echo "AFLW2000 already exists"
fi

echo "=== Downloading 300W-LP ==="
if [ ! -d "300W_LP" ]; then
    gdown "https://drive.google.com/uc?id=0B7OEHD3T4eCkVGs0TkhUWFN6N1k" -O 300W-LP.zip 2>/dev/null || \
    echo "300W-LP auto-download failed, will use alternative training data"

    if [ -f 300W-LP.zip ]; then
        unzip -q 300W-LP.zip -d 300W_LP 2>/dev/null || true
        echo "300W-LP extracted"
    fi
else
    echo "300W-LP already exists"
fi

echo "=== Downloading BIWI ==="
if [ ! -d "BIWI" ]; then
    gdown "https://drive.google.com/uc?id=1jFnGFPHIwR1tnXXa6MFM8G4JT-Exuf1T" -O BIWI.zip 2>/dev/null || \
    echo "BIWI auto-download failed, will generate synthetic"

    if [ -f BIWI.zip ]; then
        unzip -q BIWI.zip -d BIWI 2>/dev/null || true
        echo "BIWI extracted"
    fi
else
    echo "BIWI already exists"
fi

echo "=== Download complete ==="
ls -la "$BASE_DIR"
