# Model Card

## 模型與適配

- Base：`Qwen/Qwen3-1.7B`，Apache-2.0，revision
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`。
- 發布版本：QLoRA v5 / bitsandbytes NF4，prompt `optimized-v2`。
- LoRA：rank 8、alpha 16、dropout 0.05；8,716,288 個可訓練參數，占 1,729,291,264 總參數的
  0.504038%。
- 資料：840 train、140 validation、210 test、14 curated challenge；生成器 1.3.0。
- 訓練：1 epoch、53 optimizer steps、seed 42、最大長度 896、learning rate 2e-4。
- 實測：train loss 0.235826、validation loss 0.017484、訓練時間 1,467.509 秒。
- 訓練 GPU peak allocated 4,523,375,616 bytes；reserved 12,501,123,072 bytes。後者是 Windows
  WDDM／allocator 的虛擬與快取統計，不等同實體獨占顯存。

資料 manifest SHA-256 為
`873a72f19398c494539024a2fafb09a7a893e99302c1c4873738b7aa0d67094c`；訓練設定 SHA-256 為
`0a67dff1892ecc33b85f7492537752f0b92a51e0aecc8e53b1d02f4cc543752b`；adapter safetensors SHA-256
為 `07ce2d7ee13a1f8a9cdd3f3c4509caa99e349c502a14449157601cb77ab61a6b`。

這份歷史 training report schema 1.0 保存 Python、Torch、CUDA、Transformers 與 GPU。程式目前的
training report schema 1.2 會在下一次真實訓練另外保存 Accelerate、bitsandbytes、Datasets、PEFT、
safetensors、sentencepiece、cuDNN、GPU compute capability、實體顯存、resolved tokenizer revision
與 training run contract hash；後續 merge 與新 registry 候選會驗證這些 lineage，不以目前環境版本
回填過去未記錄的欄位。

## 完整評測

2026-08-29 本機 210 test + 14 challenge 的真實執行結果：

| 指標 | QLoRA v5 / NF4 | merged v5 / llama.cpp Q8 |
|---|---:|---:|
| raw JSON schema | 99.55% | 100.00% |
| required fields | 100.00% | 100.00% |
| event macro-F1 | 86.24% | 85.16% |
| severity macro-F1 | 95.35% | 85.34% |
| high/critical review recall | 100.00% | 100.00% |
| unknown accuracy | 100.00% | 96.77% |
| evidence precision | 100.00% | 100.00% |
| injection resistance | 100.00% | 100.00% |
| release gate | pass | pass |

NF4 v5 policy-v2 是正式 released 版本。Q8 通過絕對門檻並顯著降低延遲，但 severity macro-F1 與
NF4 的描述性差距為 10.01 個百分點；Q8 保留為 previous-good。相同生成契約下，Q4_K_M
的 event macro-F1 為 74.43%，相對 Q8 退化 10.73 個百分點，已在 registry 封鎖。

這些結果來自原創合成測試與小型人工 challenge，只能證明工程流程、已定義分布與安全 gate，
不能外推為真實工廠準確率或安全認證。

## 外部校準診斷

UCI AI4I 第三方合成資料另形成 98 筆 calibration 與 98 筆來源列不重疊 holdout，不參與訓練。
現行 Q8 prompt v2 在內部新版診斷 gate 通過，但 external calibration event F1 只有 40.82%。實驗
prompt v5 在 calibration 與首次 holdout 的政策後結果皆達 100%，可是 holdout raw evidence
precision 只有 84.85%、16.33% 案例需要安全層介入；同一 prompt 回到內部資料時 event F1 降到
69.72%、unknown accuracy 降到 45.16%。因此 v5 不發布，正式模型與 registry 狀態不變。

這項 holdout 只能稱為「同一第三方合成來源內的未觸碰列級驗證」，不是獨立工廠或真實設備驗證。
完整方法見 [`EXTERNAL_CALIBRATION.md`](EXTERNAL_CALIBRATION.md)。

## Intended use

將繁體中文設備事件正規化為事件類型、嚴重度、摘要、原文證據、缺失資訊、允許清單動作與人工
複核旗標，供工單分流、人工看板或離線分析使用。

## Out of scope

模型不得控制設備、下達維修程序、取代安全工程師、診斷人員傷害，或處理未授權真實客戶資料。
critical、high、unknown、提示注入、無效 schema 與無 grounding 證據都會轉人工或保守 fallback。

## 已知行為

- NF4 policy-v2 完整集有 31 筆 event 與 12 筆 severity mismatch；唯一 raw JSON 損壞會 fail closed，不會被
  fallback 冒充原始格式正確。
- Q8 使用嚴格 JSON Schema 與 reason-code 白名單後格式達 100%，但有 33 筆 event 與 37 筆
  severity mismatch。
- Q4 的主要退化是事件類型：57 筆 mismatch，顯示 4-bit 檔案雖小且快，不能只憑安全硬門檻發布。
- 人工 challenge 已用於迭代診斷，屬於開發回歸集，不是未觸碰的外部驗證集。
- AI4I v5 的外部高分無法泛化回內部資料；安全層介入後的正確率不得當成模型原生正確率。
