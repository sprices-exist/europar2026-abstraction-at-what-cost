#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLI_DIR="${SCRIPT_DIR}/CLI"
BF16_DIR="${SCRIPT_DIR}/bf16_benchmarks"
FP32_DIR="${SCRIPT_DIR}/fp32_benchmarks"
OUT_DIR="${SCRIPT_DIR}/nsys_reports"
SQLITE_DIR="${SCRIPT_DIR}/nsys_sqlite"

mkdir -p "${OUT_DIR}" "${SQLITE_DIR}"

if [[ ! -d "${CLI_DIR}" ]]; then
  echo "Error: CLI directory not found at ${CLI_DIR}" >&2
  exit 1
fi
if [[ ! -d "${BF16_DIR}" ]]; then
  echo "Error: bf16_benchmarks directory not found at ${BF16_DIR}" >&2
  exit 1
fi
if [[ ! -d "${FP32_DIR}" ]]; then
  echo "Error: fp32_benchmarks directory not found at ${FP32_DIR}" >&2
  exit 1
fi

cd "${CLI_DIR}"

echo "Installing Nsight Systems CLI..."
sudo apt update
sudo apt install -y --no-install-recommends gnupg
echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" | sudo tee /etc/apt/sources.list.d/nvidia-devtools.list > /dev/null
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo apt update
sudo apt install -y nsight-systems-cli

echo "Installing Python requirements..."
python3 -m pip install -r "${CLI_DIR}/requirements.txt"

run_nsys() {
  local output_base="$1"
  shift
  echo "Running: ${output_base}"
  nsys profile \
    -o "${OUT_DIR}/${output_base}" \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    "$@"
}

echo "Running BF16 benchmarks..."
cd "${BF16_DIR}"
# Paged BF16
run_nsys "paged_bf16_ctx512"   python benchmark_paged.py --context-len 512
run_nsys "paged_bf16_ctx1024"  python benchmark_paged.py --context-len 1024
run_nsys "paged_bf16_ctx2048"  python benchmark_paged.py --context-len 2048
run_nsys "paged_bf16_ctx4096"  python benchmark_paged.py --context-len 4096
run_nsys "paged_bf16_ctx8192"  python benchmark_paged.py --context-len 8192
run_nsys "paged_bf16_ctx16384" python benchmark_paged.py --context-len 16384

# Variant BF16
run_nsys "paged_variants_bf16_ctx512"   python benchmark_paged_cutile_variants.py --context-len 512 --dtype bf16
run_nsys "paged_variants_bf16_ctx1024"  python benchmark_paged_cutile_variants.py --context-len 1024 --dtype bf16
run_nsys "paged_variants_bf16_ctx2048"  python benchmark_paged_cutile_variants.py --context-len 2048 --dtype bf16
run_nsys "paged_variants_bf16_ctx4096"  python benchmark_paged_cutile_variants.py --context-len 4096 --dtype bf16
run_nsys "paged_variants_bf16_ctx8192"  python benchmark_paged_cutile_variants.py --context-len 8192 --dtype bf16
run_nsys "paged_variants_bf16_ctx16384" python benchmark_paged_cutile_variants.py --context-len 16384 --dtype bf16

# GEMM BF16
run_nsys "gemm_bf16_512"  python benchmark_matmul.py --size 512
run_nsys "gemm_bf16_1024" python benchmark_matmul.py --size 1024
run_nsys "gemm_bf16_2048" python benchmark_matmul.py --size 2048
run_nsys "gemm_bf16_4096" python benchmark_matmul.py --size 4096
run_nsys "gemm_bf16_8192" python benchmark_matmul.py --size 8192

# Flash BF16
run_nsys "flash_bf16_512"  python benchmark_flash.py --seq-len 512
run_nsys "flash_bf16_1024" python benchmark_flash.py --seq-len 1024
run_nsys "flash_bf16_2048" python benchmark_flash.py --seq-len 2048
run_nsys "flash_bf16_4096" python benchmark_flash.py --seq-len 4096
run_nsys "flash_bf16_8192" python benchmark_flash.py --seq-len 8192

echo "Running FP32 benchmarks..."
cd "${FP32_DIR}"
# Paged FP32
run_nsys "paged_fp32_ctx512"   python benchmark_paged.py --context-len 512 --dtype fp32
run_nsys "paged_fp32_ctx1024"  python benchmark_paged.py --context-len 1024 --dtype fp32
run_nsys "paged_fp32_ctx2048"  python benchmark_paged.py --context-len 2048 --dtype fp32
run_nsys "paged_fp32_ctx4096"  python benchmark_paged.py --context-len 4096 --dtype fp32
run_nsys "paged_fp32_ctx8192"  python benchmark_paged.py --context-len 8192 --dtype fp32
run_nsys "paged_fp32_ctx16384" python benchmark_paged.py --context-len 16384 --dtype fp32

# GEMM FP16
run_nsys "gemm_fp16_512"  python benchmark_matmul.py --size 512 --dtype fp32
run_nsys "gemm_fp16_1024" python benchmark_matmul.py --size 1024 --dtype fp32
run_nsys "gemm_fp16_2048" python benchmark_matmul.py --size 2048 --dtype fp32
run_nsys "gemm_fp16_4096" python benchmark_matmul.py --size 4096 --dtype fp32
run_nsys "gemm_fp16_8192" python benchmark_matmul.py --size 8192 --dtype fp32

# Flash FP32
run_nsys "flash_fp32_512"  python benchmark_flash.py --seq-len 512 --dtype fp32
run_nsys "flash_fp32_1024" python benchmark_flash.py --seq-len 1024 --dtype fp32
run_nsys "flash_fp32_2048" python benchmark_flash.py --seq-len 2048 --dtype fp32
run_nsys "flash_fp32_4096" python benchmark_flash.py --seq-len 4096 --dtype fp32
run_nsys "flash_fp32_8192" python benchmark_flash.py --seq-len 8192 --dtype fp32

echo "Exporting all .nsys-rep files to sqlite..."
shopt -s nullglob
for rep in "${OUT_DIR}"/*.nsys-rep; do
  base="$(basename "${rep}" .nsys-rep)"
  nsys export --type sqlite --force-overwrite=true --output "${SQLITE_DIR}/${base}" "${rep}"
done

echo "Done."
echo "Reports: ${OUT_DIR}"
echo "SQLite:  ${SQLITE_DIR}"
