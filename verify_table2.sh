#!/usr/bin/env bash
set -euo pipefail

# Install sqlite3 (Ubuntu/Debian).
sudo apt update
sudo apt install -y sqlite3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SQLITE_DIR="${SCRIPT_DIR}/nsys_sqlite"

for s in 512 1024 2048 4096 8192; do
  db="${SQLITE_DIR}/flash_fp32_${s}"
  if [[ ! -f "${db}" ]]; then
    echo "Missing SQLite DB: ${db}" >&2
    exit 1
  fi

  echo "== Table 2 (Flash Attention): flash_fp32_${s} =="
  sqlite3 "${db}" \
  "SELECT s2.value, ROUND(AVG((k.end-k.start)/1000.0),2) AS avg_us
   FROM CUPTI_ACTIVITY_KIND_KERNEL k
   JOIN StringIds s2 ON k.shortName=s2.id
   WHERE s2.value IN ('flash_fwd_kernel','fmha_kernel','_triton_fwd_kernel')
   GROUP BY s2.value
   ORDER BY s2.value;"
  echo
done
