# Evidence Index

模型權重、生成資料、逐案輸入輸出與 benchmark 原始報告依隱私／大小政策排除於 Git。本索引保存
可公開的本機實測識別碼、hash 與重現入口；所有列為 measured 的項目皆在同一台作品機執行。
不含逐案文字的機器可讀摘要位於 [`docs/measured-evidence.json`](measured-evidence.json)，其資料總數、
稽核值、公平比較欄位與指標差異由測試自動檢查。

## 訓練與資料

| 證據 | 值 |
|---|---|
| training report | `artifacts/qwen3-1.7b-triage-lora-v5/training-report.json` |
| evidence kind | `measured` |
| base revision | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| dataset manifest SHA-256 | `873a72f19398c494539024a2fafb09a7a893e99302c1c4873738b7aa0d67094c` |
| training config SHA-256 | `0a67dff1892ecc33b85f7492537752f0b92a51e0aecc8e53b1d02f4cc543752b` |
| adapter SHA-256 | `07ce2d7ee13a1f8a9cdd3f3c4509caa99e349c502a14449157601cb77ab61a6b` |
| duration | 1,467.509 s |

## 完整評測

| 版本 | report | run ID | max tokens／契約 | gate | input hash |
|---|---|---|---|---|---|
| QLoRA v5 NF4 policy-v2 | `outputs/eval-C-peft-v5-full-policy-v2.json` | `4a5ecea7-5bd0-471f-8b46-b5047939608d` | 192／`prompt-schema+bounded-repair-v1` | pass | `01893121075ce7b2436f18407292e44dccc0e6a13b2299c9929af259e58d7f17` |
| GGUF Q8_0 policy-v2 | `outputs/eval-E5-q8-v5-policy-v2.json` | `b9574978-6f58-413d-98d9-de9f970f6554` | 192／`json-schema-v2` | pass | 同上 |
| GGUF Q4_K_M policy-v2 | `outputs/eval-F3-q4-v5-policy-v2.json` | `dc791d3a-5dcc-4d45-9797-c7b4e0b31eb5` | 192／`json-schema-v2` | fail | 同上 |

三份最新報告的 release gate SHA-256 均為
`2a5b42d1926d457cc54cb48c62fd8514b75f7dd39e0673b686eac6370ea5fc7a`，也都鎖定 192 tokens、
`risk-precedence-v2` 與 `release-metrics-v2`。NF4 使用 prompt schema + bounded repair，Q8／Q4
使用 llama.cpp strict JSON Schema；因此 NF4→Q8 仍只作跨引擎描述性比較，不冒充相同生成契約的
relative gate。

- `outputs/compare-E5-F3-q4-policy-v2.json`：Q8→Q4 完整差異與 Q4 relative gate；資料、gate、prompt、
  max tokens、structured-output contract、safety policy、evaluation contract 七項相容性檢查皆為
  `true`。

## 量化 artifact

| 檔案 | bytes | SHA-256 |
|---|---:|---|
| `modelforge-tw-f16.gguf` | 3,447,349,024 | `1a1cbf88f5575f7fa3377187e2b4916114d69efeed3e46b2c9742c88a7c739fa` |
| `modelforge-tw-q8_0.gguf` | 1,834,426,144 | `d4729ea11ea540d46be45effe7b41ffbe55bc51ce7e42bf7443cd2bbf84ca9ed` |
| `modelforge-tw-q4_k_m.gguf` | 1,107,408,672 | `8745574ab802c7ac799c4d2261fea0e067c638e87ea1f5591e4a9bf2cc49566d` |

來源報告：`artifacts/gguf/quantization-report.json`；llama.cpp build b10665 / commit
`ca3d5a3e10d53f7ea672cb9b6178faca3e2807bc`。

## 效能與發布

- Q8 benchmark：`outputs/benchmark-q8-v5.json`，36/36 requests、0 error。
- Q4 benchmark：`outputs/benchmark-q4-v5.json`，36/36 requests、0 error。
- registry：`outputs/model-registry.json`。事件包含 policy-v2 Q4 blocked、Q8 promoted，以及 NF4／Q8／Q4
  三份 policy-v2 evaluation attested；最終 `released=qwen3-1.7b-qlora-v5-policy-v2`、
  `previous_good=qwen3-1.7b-gguf-q8-v5-policy-v2`。

## 重現與驗證

- `scripts/evaluate.ps1 -Full`：Base／Prompt／QLoRA。
- `scripts/quantize.ps1`、`scripts/serve-gguf.ps1`：GGUF 與 llama.cpp。
- `scripts/benchmark.ps1`：併發與 TTFT。
- `scripts/verify.ps1`：格式、型別、測試、資料 audit、隱私與封裝檢查。

若重新執行產生不同 run ID，必須同時檢查資料 hash、gate hash、模型版本與 artifact hash，不能只
比較檔名。
