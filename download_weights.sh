#!/bin/bash
# Download ST-GCN weights for Action Recognition (Kinetics)

set -e

# Google Drive direct download for st_gcn.kinetics.pt
# From official ST-GCN repository: https://github.com/yysijie/st-gcn
GDRIVE_ID="1bMGaV9Fvp8HK5h04BHRvBkmQ8HxjyKyY"
OUTPUT="st_gcn_weights.pt"

echo "Downloading ST-GCN weights from Google Drive..."
echo "Target: $OUTPUT"

if [ -f "$OUTPUT" ]; then
    echo "File already exists. Skipping download."
    exit 0
fi

# Install gdown if needed
pip install -q gdown 2>/dev/null || true

# Download using gdown (handles Google Drive properly)
gdown "https://drive.google.com/uc?id=$GDRIVE_ID" -O "$OUTPUT" || {
    echo "Download failed. Alternative method:"
    echo "Manually download from: https://drive.google.com/drive/folders/1IYKoSrjeI3yYJ9bO0_z_eDo92i7ob_aF"
    echo "Look for: st_gcn.kinetics.pt"
    echo "Then save it as: $OUTPUT"
    exit 1
}

echo ""
echo "✓ Download complete!"
echo "File saved to: $(pwd)/$OUTPUT"
echo "File size: $(du -h $OUTPUT | cut -f1)"
echo ""
echo "Next steps:"
echo "1. Run: docker compose up app-gpu"
echo "   (No rebuild needed - docker-compose.yml already mounts this file)"

