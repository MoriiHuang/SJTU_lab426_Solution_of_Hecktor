#!/usr/bin/env bash
# Stop at first error
set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_IMAGE_TAG="hecktor-task1-baseline"  # 修改为你的现成镜像名称

# 检查是否提供了自定义镜像标签作为参数
if [ "$#" -eq 1 ]; then
    DOCKER_IMAGE_TAG="$1"
fi

DOCKER_NOOP_VOLUME="${DOCKER_IMAGE_TAG}-volume"
INPUT_DIR="${SCRIPT_DIR}/test/input"
OUTPUT_DIR="${SCRIPT_DIR}/test/output"

echo "输入目录: $INPUT_DIR，输出目录: $OUTPUT_DIR"

# 清理函数 - 确保输出文件的权限正确
cleanup() {
    echo "=+= 清理权限..."
    docker run --rm \
      --platform=linux/amd64 \
      --quiet \
      --volume "$OUTPUT_DIR":/output \
      --entrypoint /bin/sh \
      $DOCKER_IMAGE_TAG \
      -c "chmod -R -f o+rwX /output/* || true"
}

# 准备输出目录
if [ -d "$OUTPUT_DIR" ]; then
  chmod -f o+rwx "$OUTPUT_DIR"
  echo "=+= 清理之前的输出"
  docker run --rm \
      --platform=linux/amd64 \
      --quiet \
      --volume "$OUTPUT_DIR":/output \
      --entrypoint /bin/sh \
      $DOCKER_IMAGE_TAG \
      -c "rm -rf /output/* || true"
else
  mkdir -m o+rwx "$OUTPUT_DIR"
fi

# 注册清理函数
trap cleanup EXIT

echo "=+= 执行前向传递"

# 创建临时卷和输出目录
docker volume create "$DOCKER_NOOP_VOLUME" > /dev/null
mkdir -p "$OUTPUT_DIR"
chmod -R 777 "$OUTPUT_DIR"

# 运行容器
docker run --rm \
    --ipc=host \
    --platform=linux/amd64 \
    --gpus all \
    --network none \
    --volume "$INPUT_DIR":/input:ro \
    --volume "$OUTPUT_DIR":/output \
    --volume "$DOCKER_NOOP_VOLUME":/tmp \
    $DOCKER_IMAGE_TAG

# 清理临时卷
docker volume rm "$DOCKER_NOOP_VOLUME" > /dev/null

echo "=+= 结果已写入 ${OUTPUT_DIR}"