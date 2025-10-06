#!/bin/bash

# train.py を実行して学習し、完了後に自動で visualization.py を実行するスクリプト

# --- Docker設定 ---
DOCKER_IMAGE="monai_env:v2.3" # 必要に応じてビルドした最新のイメージ名に変更してください

# --- Weights & Biases APIキー設定 ---
# https://wandb.ai/authorize から取得したご自身のAPIキーを設定してください
WANDB_API_KEY="567edc8c3bf8893945a67410945cc6740215ea39"

# コンテナ名 (実行ごとに一意の名前にするためにタイムスタンプを追加)
CONTAINER_NAME="amed-training-$(date +%s)"

# ホスト側のプロジェクトルートディレクトリ (このスクリプトがある場所を基準に設定)
HOST_PROJECT_DIR=$(pwd)

# ★ 修正点: PyTorch Hubのキャッシュを永続化するためのホスト側ディレクトリ
HOST_CACHE_DIR="${HOST_PROJECT_DIR}/.cache"
mkdir -p "${HOST_CACHE_DIR}/torch"
echo "  Using host cache for PyTorch Hub: ${HOST_CACHE_DIR}/torch"

# コンテナ内の作業ディレクトリ
CONTAINER_WORKDIR="/workspace"

# --- 1. 学習の実行 ---
TRAINING_CONTAINER_NAME="amed-training-$(date +%s)"
echo "--- Starting Training ---"
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
  --name "${TRAINING_CONTAINER_NAME}" \
  "${DOCKER_IMAGE}" \
  python3 "${CONTAINER_WORKDIR}/DiffX2CT/train.py" "$@"

echo "--- Training Finished ---"
