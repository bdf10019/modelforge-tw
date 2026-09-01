# Evaluation

## 評測契約

正式門檻使用 210 筆 test + 14 筆 curated challenge，共 224 筆；不以 train 或 validation 計分。
每份報告保存資料內容 hash、release-gate hash、模型／revision／版本、prompt、量化、最大輸出 token、
每 slot context、結構化輸出契約、安全政策版本、評分契約版本、逐案原文、延遲與 `evidence_kind`。
評分合約 v4 另保存模型原生 evidence precision、後處理介入率、強制 unknown 比率、警告原因計數與
逐案 `validation_warnings`，避免把安全層修正誤報為模型原生能力。長時間執行逐筆
寫入續跑快取，只有資料紀錄 SHA-256、adapter 檔案 hash 與所有推論設定完全一致時才能沿用。

絕對門檻：

| 指標 | 門檻 | hard gate |
|---|---:|---|
| raw JSON schema | ≥ 99% | 否 |
| required fields | 100% | 是 |
| event macro-F1 | ≥ 85% | 否 |
| severity macro-F1 | ≥ 85% | 否 |
| high/critical review recall | ≥ 98% | 是 |
| unknown accuracy | ≥ 95% | 否 |
| evidence span precision | 100% | 是 |
| injection resistance | 100% | 是 |

Q4 另要求相對 Q8 的 event／severity macro-F1 退化各不超過 0.015、raw schema 退化不超過
0.005，且不能新增 hard-gate 失敗。比較前必須先證明資料 hash、gate hash、prompt、最大輸出 token
與結構化輸出契約、安全政策版本、評分契約版本完全相同；缺值也視為不相容。安全政策套用後的
輸出與模型原始 JSON 分開計分，因此保守 fallback 不會掩蓋原始格式錯誤。

## Base、Prompt 與早期適配

下表是同一組 14 筆 challenge 的 2026-08-28 本機真實推論，用於分離 prompt 與權重適配效果；
小樣本通過不等於正式發布。

| 指標 | A Base minimal | B Base optimized-v1 | C QLoRA v1 |
|---|---:|---:|---:|
| raw JSON schema | 0.0% | 7.1% | 100.0% |
| event macro-F1 | 1.9% | 78.1% | 100.0% |
| severity macro-F1 | 2.7% | 73.3% | 85.0% |
| high/critical review recall | 100.0% | 100.0% | 100.0% |
| evidence precision | 100.0% | 100.0% | 100.0% |
| injection resistance | 0.0% | 100.0% | 100.0% |

v1 隨後在 224 筆完整集因資料標籤與風險下限衝突而失敗。v2 因跨切分情境重用在 checkpoint-10
主動中止；v3／v4 在 challenge 顯示嚴重度校準不足。所有失敗報告與 artifact 均保留，不調低門檻。

## v5 與量化完整結果

| 指標 | C：QLoRA NF4 | E：GGUF Q8_0 | F：GGUF Q4_K_M |
|---|---:|---:|---:|
| raw JSON schema | 99.55% | 100.00% | 100.00% |
| required fields | 100.00% | 100.00% | 100.00% |
| event accuracy | 86.16% | 85.27% | 74.55% |
| event macro-F1 | 86.24% | 85.16% | 74.43% |
| severity accuracy | 94.64% | 83.48% | 90.63% |
| severity macro-F1 | 95.35% | 85.34% | 91.04% |
| high/critical review recall | 100.00% | 100.00% | 100.00% |
| unknown accuracy | 100.00% | 96.77% | 83.87% |
| evidence precision | 100.00% | 100.00% | 100.00% |
| injection resistance | 100.00% | 100.00% | 100.00% |
| case pass rate | 82.59% | 75.45% | 71.43% |
| absolute gate | pass | pass | fail |

NF4 與 llama.cpp 使用不同的生成契約，兩欄只作跨引擎描述；Q8 的 event／severity macro-F1 分別
比 NF4 低 1.08／10.01 個百分點，安全硬門檻沒有退化。Q8→Q4 則是機器驗證的公平比較：七項
相容性檢查全過，event macro-F1 下降 10.73 個百分點，遠超 1.5 個百分點的相對上限，且 Q4 本身
未過絕對門檻，所以 quantization gate 明確失敗。

## 結構化輸出契約迭代

merged BF16 challenge 與第一輪 Q8 只使用 `json_object`，各有 2/14 筆原始 JSON 損壞。改成服務端
嚴格 JSON Schema 後，Q8 challenge 達 14/14；但 224 筆完整集仍有 5 筆在自由文字
`reason_codes` 內生成未閉合字串直到 token 上限。最終契約將所有欄位設為 required，並只在新版
出站 schema 對 reason code 套用版本化白名單；Q8 完整集遂達 224/224 raw schema，同時保留歷史
報告的讀取相容性。

這項改善發生在生成前的解碼約束，不是事後修補。每次契約變更都提升評測快取指紋版本，舊輸出
不能被新版評測誤用。

## 外部校準與未觸碰列級 holdout

UCI AI4I 2020 經版本化轉換後形成 98 筆 calibration 與 98 筆 source-row-disjoint holdout。v5 在
calibration 上完成調整、程式提交後，holdout 首次執行即 98/98 系統結果通過；但 raw evidence
precision 只有 84.85%，16.33% 案例需安全層介入。v5 回到內部 224 筆時 event macro-F1 只有
69.72%、unknown accuracy 45.16%，因此外部高分不能作為發布依據。

現行 v2 在同一份新版合約的內部 224 筆仍通過（event F1 85.54%、severity F1 85.64%），在外部
calibration 則只有 event F1 40.82% 且 raw schema 85.71%，明確顯示跨來源泛化缺口。v5 與 v6
均不發布，registry 狀態不變。完整來源、hash、run ID、容量事故與介入原因見
[`EXTERNAL_CALIBRATION.md`](EXTERNAL_CALIBRATION.md)。

## Registry 結果

- `qwen3-1.7b-qlora-v5-policy-v2`：通過完整門檻與 policy-v2 attestation，目前 released。
- `qwen3-1.7b-gguf-q8-v5-policy-v2`：通過絕對門檻；本機 promote 後完成受控 rollback，目前
  retired 並作為 previous-good。
- `qwen3-1.7b-gguf-q4-v5-policy-v2`：完整門檻與 Q4 相對門檻失敗，blocked。

正式 registry 分數來自原創合成分布與 14 筆人工回歸案例；新增的外部結果來自第三方合成 AI4I，
仍不能視為真實工廠準確率、設備安全認證或真實場域泛化證據。
