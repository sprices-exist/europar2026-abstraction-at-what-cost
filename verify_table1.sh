#!/usr/bin/env bash
set -euo pipefail

# Install sqlite3 (Ubuntu/Debian).
sudo apt update
sudo apt install -y sqlite3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SQLITE_DIR="${SCRIPT_DIR}/nsys_sqlite"

for n in 512 1024 2048 4096 8192; do
  db="${SQLITE_DIR}/gemm_fp16_${n}"
  if [[ ! -f "${db}" ]]; then
    echo "Missing SQLite DB: ${db}" >&2
    exit 1
  fi

  echo "== Table 1 (GEMM): gemm_fp16_${n} =="
  sqlite3 "${db}" \
  "SELECT s.value, ROUND(AVG((k.end-k.start)/1000.0),2) AS avg_us
   FROM CUPTI_ACTIVITY_KIND_KERNEL k
   JOIN StringIds s ON k.shortName=s.id
   WHERE s.value IN ('Kernel2','matmul_kernel','triton_matmul_kernel',
                     'create_warp_gemm__locals__gemm_6a1c7341_cuda_kernel_forward')
   GROUP BY s.value
   ORDER BY s.value;"
  echo
done
