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
- 評估分開記錄模型原生 evidence precision、政策後 evidence precision、後處理介入率與逐案警告，
  防止把安全層修正包裝成模型能力。
- critical/high 關鍵風險只可向上升級；動作來自四項 allowlist，沒有設備控制工具。
- 危險語句的 stop/isolate-and-escalate 優先於缺證據的 collect-evidence 規則；缺證據只能增加人工
  複核與缺失資訊，不能把既有升級動作降級。
- 解析失敗、timeout、engine exception 都回 unknown + collect evidence + review required。
- semaphore、timeout、request ID 驗證、Prometheus outcome 與 readiness 檢查。
- timeout 後仍保留同步模型的容量槽，直到不可取消的背景推論真正結束，避免資源耗盡。
- `MODELFORGE_LOG_INPUTS` 唯一合法值為 `false`；誤設 `true` 時服務拒絕啟動，API 不提供事件原文
  logging。repository 也忽略模型、資料生成物與輸出報告。
- llama.cpp 每-slot context 預設 2048；本機啟動腳本低於 1536 時 fail fast，避免輸出被靜默截斷。
- 安全政策不可因程式更新靜默漂移；部署預設鎖定 registry 已 attested 的 v2，實驗 v3 必須顯式選擇。

## Secret and privacy checks

`scripts/privacy-check.ps1` 排除大型生成物後掃描常見 key、Bearer token、密碼賦值與使用者私人路徑。
完整驗收另執行 `pip check`、wheel 建置與 Web UI 封裝驗證；`pip check` 只驗依賴一致性，不等同
第三方套件 CVE 掃描。wheel 會無依賴安裝到一次性暫存目錄，再確認套件來源、FastAPI app 與公開
路由，最後只清理已驗證位於系統暫存根目錄下的資料夾。正式提交前同時檢查 Git staged diff。
不要把 `.env`、模型權重、客戶事件或任何 token 放入 Git。

## Residual risk

regex 不是完整語意安全分類器；本機 llama-server 只綁定 `127.0.0.1` 且沒有 API key。真實部署
需要領域規則、身分驗證、TLS、rate limit、網路隔離、集中 audit 與事故應變流程。本 demo 沒有
宣稱具備這些外部基礎設施。
