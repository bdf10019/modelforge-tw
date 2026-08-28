# Security

## Threat model

- 事件文字可能夾帶 prompt injection 或聲稱自己是 system instruction。
- 模型可能產生不存在的證據、錯誤 enum、不安全動作或低估危險程度。
- 呼叫端可能送入超長文字、額外欄位、控制字元或偽造 request ID。
- 模型 server 可能 timeout、離線或回傳非 JSON。

## Controls

- 輸入／輸出均 `extra=forbid`，限制長度、陣列大小與控制字元。
- OpenAI-compatible 路由只接受已文件化的參數子集；未部署模型、非零 temperature、過小生成預算
  與未知參數會拒絕，不會靜默忽略。
- llama.cpp 出站請求使用 strict JSON Schema，所有欄位 required，enum 與 reason code 採允許清單。
- prompt 明確把事件視為不可信資料；server 另做注入 regex，不依賴模型服從。
- 證據必須逐字存在輸入；否則移除並強制人工複核。
- critical/high 關鍵風險只可向上升級；動作來自四項 allowlist，沒有設備控制工具。
- 解析失敗、timeout、engine exception 都回 unknown + collect evidence + review required。
- semaphore、timeout、request ID 驗證、Prometheus outcome 與 readiness 檢查。
- 預設 `MODELFORGE_LOG_INPUTS=false`；repository 忽略模型、資料生成物與輸出報告。

## Secret and privacy checks

`scripts/privacy-check.ps1` 排除大型生成物後掃描常見 key、Bearer token、密碼賦值與使用者私人路徑。
正式提交前同時檢查 Git staged diff。不要把 `.env`、模型權重、客戶事件或任何 token 放入 Git。

## Residual risk

regex 不是完整語意安全分類器；本機 llama-server 只綁定 `127.0.0.1` 且沒有 API key。真實部署
需要領域規則、身分驗證、TLS、rate limit、網路隔離、集中 audit 與事故應變流程。本 demo 沒有
宣稱具備這些外部基礎設施。
