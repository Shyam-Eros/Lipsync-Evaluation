#!/usr/bin/env bash
# Download SyncNet v2 + S3FD face detector weights (Wav2Lip / joonson protocol).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${ROOT}/data" "${ROOT}/detectors/s3fd/weights"

echo "Downloading syncnet_v2.model ..."
wget -c "http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model" \
  -O "${ROOT}/data/syncnet_v2.model"

echo "Downloading sfd_face.pth ..."
wget -c "https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/sfd_face.pth" \
  -O "${ROOT}/detectors/s3fd/weights/sfd_face.pth"

echo "Done."
echo "  SyncNet : ${ROOT}/data/syncnet_v2.model"
echo "  S3FD    : ${ROOT}/detectors/s3fd/weights/sfd_face.pth"
