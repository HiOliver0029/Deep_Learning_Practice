#!/usr/bin/env bash
set -euo pipefail

# === Configuration ===
FILE_ID="1ZeoiIgGZGwg_J8tb9SdFYxvnd-6Ag13V"   # Replace with your Google Drive file ID
OUTFILE="adapter_checkpoint.zip"         # The name of the zip file to download
DESTDIR="./adapter_checkpoint"              # Directory where the file will be extracted

# === Check if gdown is installed ===
if ! command -v gdown >/dev/null 2>&1; then
  echo "[ERROR] gdown is not installed. Please run: pip install gdown"
  exit 1
fi

mkdir -p "$DESTDIR"

echo "[INFO] Downloading $OUTFILE ..."
gdown --id "$FILE_ID" -O "$OUTFILE"

echo "[INFO] Extracting to $DESTDIR ..."
unzip -o "$OUTFILE" -d "$DESTDIR"

echo "[INFO] Done! Files are extracted to $DESTDIR"
