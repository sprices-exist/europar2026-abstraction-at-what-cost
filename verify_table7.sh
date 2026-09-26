#!/usr/bin/env bash
set -euo pipefail

# Install sqlite3 (Ubuntu/Debian).
sudo apt update
sudo apt install -y sqlite3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SQLITE_DIR="${SCRIPT_DIR}/nsys_sqlite"

for c in 512 1024 2048 4096 8192 16384; do
  db="${SQLITE_DIR}/paged_bf16_ctx${c}"
  if [[ ! -f "${db}" ]]; then
    echo "Missing SQLite DB: ${db}" >&2
    exit 1
  fi

  echo "== Table 7 (Paged Attention BF16): paged_bf16_ctx${c} =="
  sqlite3 "${db}" \
  "SELECT s.value, ROUND(AVG((k.end-k.start)/1000.0),2) AS avg_us
   FROM CUPTI_ACTIVITY_KIND_KERNEL k
   JOIN StringIds s ON k.shortName=s.id
   WHERE s.value IN ('paged_attention_kernel','triton_paged_attn_kernel')
   GROUP BY s.value
   ORDER BY s.value;"
  echo
done
