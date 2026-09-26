#!/usr/bin/env bash
set -euo pipefail

# Install sqlite3 (Ubuntu/Debian).
sudo apt update
sudo apt install -y sqlite3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SQLITE_DIR="${SCRIPT_DIR}/nsys_sqlite"
DB="${SQLITE_DIR}/paged_variants_bf16_ctx4096"

if [[ ! -f "${DB}" ]]; then
  echo "Missing SQLite DB: ${DB}" >&2
  exit 1
fi

echo "== Table 4 (Paged BF16 variants at CTX=4096) =="
sqlite3 "${DB}" \
"SELECT s.value, ROUND(AVG((k.end-k.start)/1000.0),2) AS avg_us
 FROM CUPTI_ACTIVITY_KIND_KERNEL k
 JOIN StringIds s ON k.shortName=s.id
 WHERE s.value IN ('paged_attention_kernel_old',
                   'paged_attention_kernel_tuned',
                   'paged_attention_kernel_aggressive',
                   'triton_paged_attn_kernel')
 GROUP BY s.value
 ORDER BY s.value;"
