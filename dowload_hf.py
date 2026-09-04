#!/usr/bin/env python3
"""
Versão minimalista para baixar train.bin
"""

import os
from huggingface_hub import hf_hub_download

# Configure aqui
REPO_ID = "marcos-j-leemes/tinyS"
TOKEN = "..."  # Coloque seu token aqui | acelera o download
FILENAME = "train.bin"

# Baixa
print(f"Baixando {FILENAME}...")
path = hf_hub_download(
    repo_id=REPO_ID,
    filename=FILENAME,
    repo_type="dataset",
    token=TOKEN,
    local_dir=".",
)

print(f" Baixado para: {path}")
