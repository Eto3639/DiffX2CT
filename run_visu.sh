#!/bin/bash

# visualization.py を実行して、指定したチェックポイントから可視化を行うスクリプト

# --- Docker設定 ---
DOCKER_IMAGE="monai_env:v2.3" # train.py実行時と同じイメージ名

# --- Weights & Biases APIキー設定 ---
# visualize_and_save_mprがWandBに画像をアップロードするため、APIキーが必要です
WANDB_API_KEY="567edc8c3bf8893945a67410945cc6740215ea39"

# コンテナ名 (実行ごとに一意の名前にするためにタイムスタンプを追加)
CONTAINER_NAME="amed-visualization-$(date +%s)"

# ホスト側のプロジェクトルートディレクトリ (このスクリプトがある場所を基準に設定)
HOST_PROJECT_DIR=$(pwd)

# PyTorch Hubのキャッシュを永続化するためのホスト側ディレクトリ
HOST_CACHE_DIR="${HOST_PROJECT_DIR}/.cache"
mkdir -p "${HOST_CACHE_DIR}/torch"
echo "  Using host cache for PyTorch Hub: ${HOST_CACHE_DIR}/torch"

# コンテナ内の作業ディレクトリ
CONTAINER_WORKDIR="/workspace"

# --- 1. 可視化の実行 ---
echo "--- Starting Visualization ---"
echo "  Image: ${DOCKER_IMAGE}"
echo "  Host Dir: ${HOST_PROJECT_DIR}"
echo "  Container Dir: ${CONTAINER_WORKDIR}"
echo "  Arguments: $@"

docker run --rm \
  --gpus all \
  --ipc=host \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  --shm-size="128g" \
  -e WANDB_API_KEY="${WANDB_API_KEY}" \
  -v "${HOST_PROJECT_DIR}:${CONTAINER_WORKDIR}" \
  -v "${HOST_CACHE_DIR}/torch:/root/.cache/torch" \
  --name "${CONTAINER_NAME}" \
  "${DOCKER_IMAGE}" \
  python3 "${CONTAINER_WORKDIR}/DiffX2CT/visualization.py" "$@"

echo "--- Visualization Finished ---"
