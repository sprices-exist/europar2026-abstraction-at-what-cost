# Overview Document: Run And Verify Guide

This document is the artifact overview for reproducing the benchmark results from `Euro-Par_2026_paper_254.pdf`.

It explains how to:
- run `run.sh`,
- generate Nsight Systems SQLite outputs,
- verify the generated SQLite data against `data/Euro-Par_2026_paper_254.pdf`.

## Artifact Description

The artifact contains:
- benchmark implementations (`bf16_benchmarks/`, `fp32_benchmarks/`),
- an execution script (`run.sh`) that profiles runs with Nsight Systems,
- generated profiling outputs (`nsys_reports/`) and exported SQLite databases (`nsys_sqlite/`),
- table verification scripts (`verify_table1.sh`, `verify_table2.sh`, `verify_table3.sh`, `verify_table4.sh`, `verify_table7.sh`).

## Hardware And Software Requirements

- GPU: NVIDIA GeForce RTX 5070 Ti (Blackwell, `sm_120`).
- CUDA Toolkit: `13.1` or newer.
- NVIDIA Driver: `590` or newer.
- Python: `3.10+` recommended.
- Nsight Systems CLI: installed by `run.sh` (`nsys` must be available in `PATH`).
- `sqlite3`: required for data verification.

## 1) Make `run.sh` Executable And Run It

From the `scripts` directory:

```bash
cd /Users/tanmaynandanikar/Documents/university/cutile_runs/scripts
chmod +x run.sh
./run.sh
```

What `run.sh` does:
- runs BF16 and FP32 benchmark suites,
- writes `.nsys-rep` files into `nsys_reports/`,
- exports each report to SQLite in `nsys_sqlite/`.

After completion, verify output files exist:

```bash
ls -1 nsys_reports | wc -l
ls -1 nsys_sqlite | wc -l
```

Expected output:
- `run.sh` finishes with `Done.` and prints report/output directories.
- `nsys_reports/` contains `.nsys-rep` files.
- `nsys_sqlite/` contains SQLite files such as `gemm_fp16_4096`, `flash_fp32_8192`, `paged_fp32_ctx4096`.

## 2) Verify Paper Tables Using Shell Scripts

From the `scripts` directory, run:

```bash
cd /Users/tanmaynandanikar/Documents/university/cutile_runs/scripts
chmod +x verify_table1.sh verify_table2.sh verify_table3.sh verify_table4.sh verify_table7.sh
./verify_table1.sh
./verify_table2.sh
./verify_table3.sh
./verify_table4.sh
./verify_table7.sh
```

Script to table mapping:
- `verify_table1.sh` -> Table 1 (GEMM)
- `verify_table2.sh` -> Table 2 (Flash Attention)
- `verify_table3.sh` -> Table 3 (Paged Attention FP32)
- `verify_table4.sh` -> Table 4 (Paged BF16 variants at CTX=4096)
- `verify_table7.sh` -> Table 7 (Paged Attention BF16)

Each script installs `sqlite3` before running queries.

Expected output:
- each script prints per-dataset blocks and kernel `avg_us` values from SQLite,
- printed values should be close to corresponding table values in the paper (small run-to-run variance is expected).

## 3) Optional: One-Command Full Consistency Report

Run this from `scripts/` to print largest relative differences against the paper's table means:

```bash
python3 - <<'PY'
import os, sqlite3
base='nsys_sqlite'
def means(db):
    con=sqlite3.connect(db)
    cur=con.cursor()
    cur.execute('SELECT s.value, AVG((k.end-k.start)/1000.0) FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id GROUP BY s.value')
    out=dict(cur.fetchall())
    con.close()
    return out
rows=[]
gemm={512:(4.5,5.4,64.7,19.6),1024:(26.7,35.1,126.0,106.1),2048:(183,238.7,586,731),4096:(1407,1809,4639,5617),8192:(11171,14132,35077,46070)}
for n,(a,b,c,d) in gemm.items():
    m=means(f'{base}/gemm_fp16_{n}')
    rows += [(f'gemm_fp16_{n}','Kernel2',a,m['Kernel2']),
             (f'gemm_fp16_{n}','matmul_kernel',b,m['matmul_kernel']),
             (f'gemm_fp16_{n}','triton_matmul_kernel',c,m['triton_matmul_kernel']),
             (f'gemm_fp16_{n}','warp_kernel',d,m['create_warp_gemm__locals__gemm_6a1c7341_cuda_kernel_forward'])]
flash={512:(15.8,19.7,21.9),1024:(54.3,69.8,65.4),2048:(204.7,262.0,245.4),4096:(802.0,956.8,944.0),8192:(2990,3775,3489)}
for n,(a,b,c) in flash.items():
    m=means(f'{base}/flash_fp32_{n}')
    rows += [(f'flash_fp32_{n}','flash_fwd_kernel',a,m['flash_fwd_kernel']),
             (f'flash_fp32_{n}','fmha_kernel',b,m['fmha_kernel']),
             (f'flash_fp32_{n}','_triton_fwd_kernel',c,m['_triton_fwd_kernel'])]
p3={512:(852,154),1024:(1717,305),2048:(3372,610),4096:(6665,1212),8192:(13329,2479),16384:(26693,5090)}
for n,(a,b) in p3.items():
    m=means(f'{base}/paged_fp32_ctx{n}')
    rows += [(f'paged_fp32_ctx{n}','paged_attention_kernel',a,m['paged_attention_kernel']),
             (f'paged_fp32_ctx{n}','triton_paged_attn_kernel',b,m['triton_paged_attn_kernel'])]
p7={512:(706,64),1024:(1422,153),2048:(2819,297),4096:(5603,596),8192:(11231,1219),16384:(22590,2488)}
for n,(a,b) in p7.items():
    m=means(f'{base}/paged_bf16_ctx{n}')
    rows += [(f'paged_bf16_ctx{n}','paged_attention_kernel',a,m['paged_attention_kernel']),
             (f'paged_bf16_ctx{n}','triton_paged_attn_kernel',b,m['triton_paged_attn_kernel'])]
v=means(f'{base}/paged_variants_bf16_ctx4096')
rows += [('paged_variants_bf16_ctx4096','paged_attention_kernel_old',5517.06,v['paged_attention_kernel_old']),
         ('paged_variants_bf16_ctx4096','paged_attention_kernel_tuned',1113.90,v['paged_attention_kernel_tuned']),
         ('paged_variants_bf16_ctx4096','paged_attention_kernel_aggressive',609.44,v['paged_attention_kernel_aggressive']),
         ('paged_variants_bf16_ctx4096','triton_paged_attn_kernel',589.84,v['triton_paged_attn_kernel'])]
diffs=[]
for db,k,e,g in rows:
    rel=abs(g-e)/e*100
    diffs.append((rel,db,k,e,g))
diffs.sort(reverse=True)
print(f'Compared rows: {len(diffs)}')
print(f'Max relative deviation: {diffs[0][0]:.2f}%')
for rel,db,k,e,g in diffs[:10]:
    print(f'{db:28s} {k:32s} expected={e:9.2f} got={g:9.2f} rel={rel:5.2f}%')
PY
```

If all deviations are within a small tolerance (for example <=10%), your regenerated SQLite results are consistent with the paper.

## Expected Execution Time

- `run.sh`: under 1 hour on the target hardware, assuming reasonable internet speed (>5 MB/s), since dependencies/tools may need to be installed.
- Each `verify_table*.sh`: typically a few seconds to a couple of minutes.
- Optional full consistency report (Section 3): typically under 1 minute.

## Mapping: Reproduced Results -> Paper Tables

- `verify_table1.sh` -> Table 1 (GEMM kernel latency).
- `verify_table2.sh` -> Table 2 (Flash Attention latency).
- `verify_table3.sh` -> Table 3 (Paged Attention latency, FP32; PyTorch reference is an unfused aggregate baseline).
- `verify_table4.sh` -> Table 4 (BF16 Paged Attention variant study at CTX=4096).
- `verify_table7.sh` -> Table 7 (Paged Attention latency, BF16).
