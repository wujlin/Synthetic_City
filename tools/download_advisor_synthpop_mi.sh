#!/usr/bin/env bash
# 导师合成人口 Michigan 数据下载（OSF，约 1.8GB）
#
# 用法：
#   本地：bash tools/download_advisor_synthpop_mi.sh
#   wsA 上直接下到数据目录（推荐，省去向日葵传输）：
#     export RAW_ROOT=/home/jinlin/data/geoexplicit_data
#     export DATA_ROOT="$RAW_ROOT/synthetic_city/data"
#     bash tools/download_advisor_synthpop_mi.sh
#   或指定目录：bash tools/download_advisor_synthpop_mi.sh --out-dir /path/to/reference/advisor_synthpop
#
# 建议在终端跑，不要用 IDE Run，避免大文件超时被杀。

set -e
URL="https://osf.io/download/66dd056e896be9e163a3c7fa/"

# 输出目录：优先 --out-dir，其次 DATA_ROOT/reference/advisor_synthpop，最后 repo 下 dataset
if [[ "${1:-}" == "--out-dir" && -n "${2:-}" ]]; then
  OUT_DIR="${2}"
  shift 2
elif [[ -n "${DATA_ROOT:-}" ]]; then
  OUT_DIR="$DATA_ROOT/reference/advisor_synthpop"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
  OUT_DIR="$REPO_ROOT/dataset/advisor_synthpop"
fi

OUT_FILE="$OUT_DIR/mi.zip"
mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

# 删除不完整文件，从头下载
rm -f mi.zip mi.zip.partial

echo "目标目录: $OUT_DIR"
echo "开始下载导师合成人口 Michigan (~1.8GB)，请耐心等待..."
echo "  --progress-bar: 进度条  --max-time 0: 不超时"
echo ""

curl -L -o "$OUT_FILE" "$URL" \
  --progress-bar \
  --connect-timeout 60 \
  --max-time 0

echo ""
echo "下载完成。校验中..."
ls -lh "$OUT_FILE"
if command -v unzip &>/dev/null; then
  unzip -t "$OUT_FILE" | tail -3
  echo "ZIP 校验通过。"
else
  echo "未安装 unzip，请手动检查: unzip -t $OUT_FILE"
fi
