# External Calibration and Holdout

## 目的與資料邊界

本階段加入 UCI 的 [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i)
（DOI `10.24432/C5HS5C`、CC BY 4.0）作為第三方來源。原始資料有 10,000 列；本專案鎖定下載
ZIP 與 CSV 的 SHA-256、欄位集合、列數、轉換版本及抽樣 seed，產生 98 筆校準集與 98 筆列級
不重疊 holdout。

這仍是第三方合成的加工設備資料，不是真實工廠量測、不是人工標註語料，也沒有 pressure／
communication 事件。它能檢查來源與文字型態改變後的行為，但不能支持真實場域準確率宣稱。
外部資料只用於評測，不進入 840 筆訓練集。

| 項目 | 固定值 |
|---|---|
| source CSV SHA-256 | `dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e` |
| transform | `ai4i-calibration-v2` |
| calibration rows hash | `afbf2bc44364db46ad175710a8758a852bf638357a2de6325ef970b43d2260e5` |
| holdout rows hash | `1820378b4ff4c6a5737aac5aaec20217a13b7c27f5e27dec309a73b2c2229ed7` |
| calibration JSONL SHA-256 | `cfa2b95d127a52ad3b5e6692394f7ffaec96ae654ceccf0d0c26429f74f005bd` |
| holdout JSONL SHA-256 | `e1e2b4b1f45ba23b513636c0c5d9091b7d1256fee99ffb84aad7bdae0e1aab57` |
| source-row overlap | 0 |
| combined audit | 196 records，所有 leakage／evidence／policy errors 為 0 |

每個 split 各含 thermal、electrical、mechanical、process 的 high／medium／low 各 7 筆，以及
unknown 14 筆。高嚴重度來自單一失效旗標；medium／low 是未失效且靠近或仍在衍生門檻的列；
unknown 刻意只提供單點讀值並拿掉基準、門檻與趨勢。這些標籤是版本化轉換規則，不是 UCI
原始事件標籤，也不是領域專家診斷。

## 防止調參污染

1. `d74b64d` 先提交兩組不同 seed、共享排除集合與 row-overlap 稽核。
2. 校準發現 v1 把所有感測欄位同時標成異常，造成互相矛盾的輸入；`4c74b0c` 將轉換升為 v2，
   只標示與事件相關的衍生訊號，來源列不變。
3. prompt v5、安全政策 v3 與評估合約 v4 在 `61ce586` 固定後，才首次執行 holdout。
4. holdout 只執行一次；之後的 v6 實驗不再把 holdout 當成未觸碰資料，也不以其結果調參。

因此 holdout 是「同一第三方合成來源內的未觸碰列級驗證」，不是另一個獨立機構、不同設備或真實
工廠資料集。這個限定必須和分數一起說明。

## 容量事故與修正

第一輪 Q8 使用 llama-server 總 context 4096、parallel 4，等於每 slot 1024 tokens。外部 prompt
約 967 tokens，再保留 192 output tokens 後超出容量；server log 顯示 `truncated=1`，98 筆 raw
schema 全部失敗。這是基礎設施容量錯誤，不是模型品質分數。

改成總 context 8192、parallel 4（每 slot 2048）後不再截斷；transform-v1 的 raw schema 回升至
77.55%，才看見真正的分類與輸出長度問題。評估報告 schema 因此新增 `context_window_tokens`，Q4
公平比較也把 context 納入相容性檢查；啟動腳本與 Kubernetes baseline 現在預設每 slot 2048，
低於 1536 時本機啟動腳本直接拒絕。

## 最終測量與發布決策

所有下列 Q8 報告使用 192 output tokens、每 slot 2048、strict JSON Schema。`system evidence` 是
安全政策後結果；`raw evidence` 是模型原始 evidence span 的逐字精度。

| 資料／prompt | raw schema | event F1 | severity F1 | unknown | raw evidence | 介入率 | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| 內部 224／v2 | 100.00% | 85.54% | 85.64% | 96.77% | 99.48% | 24.11% | pass |
| 外部 calibration 98／v2 | 85.71% | 40.82% | 100.00% | 100.00% | 92.05% | 19.39% | fail |
| 外部 calibration 98／v5 | 100.00% | 100.00% | 100.00% | 100.00% | 84.85% | 17.35% | pass |
| 外部 holdout 98／v5 | 100.00% | 100.00% | 100.00% | 100.00% | 84.85% | 16.33% | pass |
| 內部 224／v5 | 100.00% | 69.72% | 84.97% | 45.16% | 99.51% | 34.38% | fail |
| 外部 calibration 98／v6 | 87.76% | 36.85% | 96.41% | 100.00% | 66.28% | 41.84% | fail |

v5 的 holdout 系統結果為 98/98，但不能解讀成模型原生完美：16/98 由安全層介入，其中 15 筆
移除非逐字 evidence、1 筆把明確「無法判定事件類型」的猜測強制改成 unknown。更重要的是，v5
回到內部 224 筆後 event F1 降至 69.72%、unknown accuracy 降至 45.16%，所以它是被拒絕的
外部校準 profile，不是發布候選。v6 也未形成折衷。

目前 registry 不變：NF4 policy-v2 維持 released，Q8 policy-v2 維持 previous-good，Q4 維持
blocked。下一個可發布版本需要新的訓練資料或 adapter，使同一 prompt 同時通過內部 224 與新的
跨來源 holdout；不能靠資料集專用 prompt 或安全層介入率掩蓋模型能力缺口。

## 重現

```powershell
.\scripts\prepare-external-calibration.ps1
.\.venv\Scripts\modelforge.exe evaluate `
  --data data\external\generated --splits external `
  --engine llama-http --base-url http://127.0.0.1:8081 `
  --model-id modelforge-tw-q8_0 `
  --model-version qwen3-1.7b-merged-v5-q8_0-policy-v2 --quantization q8_0 `
  --prompt-version optimized-v2 `
  --max-new-tokens 192 --context-window-tokens 2048 `
  --safety-policy-version risk-precedence-v3
```

原始 ZIP、CSV、生成 JSONL 與逐案報告都位於 Git 忽略目錄；公開 Git 只保留來源設定、轉換程式、
測試、hash、run ID 與去識別統計。

服務與部署預設仍是 registry 已 attested 的 `risk-precedence-v2`。v3 只為上述外部實驗顯式啟用；
沒有 `--safety-policy-version risk-precedence-v3` 時，不會靜默套用實驗用 unknown override。
