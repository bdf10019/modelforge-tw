# Demo

## 零模型模式

```powershell
.\scripts\bootstrap.ps1 -Profile core
.\scripts\run-demo.ps1 -Engine fixture
```

開啟 `http://127.0.0.1:8000`，依序試用一般事件、資訊不足與提示注入範例。fixture 頁面會明確標示
`evidence_kind=fixture`，不可截圖宣稱為模型效能。

## 真實 adapter

```powershell
.\scripts\bootstrap.ps1 -Profile ml -DownloadModel
.\scripts\run-demo.ps1 -Engine transformers
```

模型與 adapter 必須已存在於 Git 忽略目錄。啟動時會載入約數 GB 權重；`/readyz` 在模型完成載入前
不會 ready。

```powershell
$body = @{ incident_text = "控制櫃突然出現火花並伴隨焦味。" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/triage -ContentType application/json -Body $body
```

展示時說明三點：原始模型輸出不可信、證據必須落在輸入、high/critical/unknown 一律人工複核。
`/api/config` 只顯示非敏感 runtime 契約，例如並發、queue 與 bytes 上限、是否需要認證及綁定的
registry manifest ID；不回傳 bearer key、事件原文或本機路徑。

## GGUF + llama.cpp

完成 merge 與量化後，先啟動本機 OpenAI-compatible 模型服務，再讓 ModelForge-TW API 連線：

```powershell
.\scripts\serve-gguf.ps1 -Quantization q8_0 -Port 8081
# 另開終端機
.\scripts\run-demo.ps1 -Engine llama-http -ModelId modelforge-tw-q8_0 `
  -ModelVersion qwen3-1.7b-merged-v5-q8_0-policy-v2 -Quantization q8_0 `
  -LlamaBaseUrl http://127.0.0.1:8081
```

`llama-server` 預設使用 8192 total context、4 個 parallel slots，即每 slot 2048；啟動腳本會拒絕
低於 1536 的 slot。若硬體不同，應調整 parallel／context 或 GPU layers，再重新執行 benchmark，
不可沿用本機數字。Q8
通過完整絕對門檻；Q4_K_M 僅供展示量化退化與 blocked 流程，不應作為預設 demo。

現有正式 registry 是保留真實操作歷史的 schema 1.0，可由新版程式唯讀驗證；不要為了展示而改寫。
未來用新版完整 training／evaluation attestation 加入候選時，mutation 會升級事件 hash chain，且可
用 `MODELFORGE_REGISTRY_DEPLOYMENT_ROLE=released` 或 `previous-good` 讓服務啟動時 fail closed 核對
runtime identity。fixture 或缺少新版欄位的舊 manifest 不會通過這項綁定。
若 registry 的報告與模型 artifact 也已唯讀掛載，可再設定
`MODELFORGE_REGISTRY_ARTIFACT_ROOT=<掛載根目錄>`，讓每個服務實例在啟動時重算檔案大小與 SHA-256；
這驗證的是掛載證據副本，不宣稱能遠端證明另一個模型 server 已載入同一份實體檔案。
