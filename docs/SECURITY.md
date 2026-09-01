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
- 所有引擎解析後都拒絕 allowlist 以外的 event、severity、safe action 與 reason code；llama.cpp
  路徑還會在生成前用 strict JSON Schema 約束同一份契約。
- `incident_text`、asset 與 signal 文字都視為不可信資料並納入注入偵測；signal 字串限制 128 字，
  數字必須是有限值且在明確範圍內，控制字元會在進入 prompt 前拒絕。
- HTTP body 預設上限 32 KiB；有無 `Content-Length` 都會依實收 bytes 計數，超限先回 413，
  不讓大型 payload 進入 JSON／Pydantic 解析。上限可設定但限制在 1 KiB 至 1 MiB。
- llama.cpp 一般與 SSE 回應都以解碼後 bytes 計數，預設上限 256 KiB；`Content-Length` 超限會
  在讀 body 前拒絕，chunked／壓縮回應則依實收解碼內容停止，無換行事件也不能無界累積。
- 設定 `MODELFORGE_API_KEY` 後，兩個推論路由都要求標準 `Authorization: Bearer ...`；key 至少
  32 字元、以 constant-time 比對、不出現在 config response。Kubernetes 由可選 Secret 注入，
  repository 不保存值；未設定時保留 loopback 本機 demo 相容性。
- 推論／config 回應與認證錯誤都帶 `Cache-Control: no-store`；展示頁限制同源 script／style／連線，
  禁止 object、base URI 與 frame ancestor，所有 HTTP 回應另加 nosniff、deny framing、
  no-referrer 與 camera／microphone／geolocation 權限限制。
- `X-Request-ID` 在 JSON 解析前限制為 1–96 字元安全字元；重複或非法值回 400 且不回顯。所有 HTTP
  response 都由 server 附追蹤 ID，便於關聯失敗而不需要記錄事件原文。
- semaphore、timeout、request ID 驗證、Prometheus outcome 與 readiness 檢查。
- timeout 後仍保留同步模型的容量槽，直到不可取消的背景推論真正結束，避免資源耗盡。
- 等待容量另有獨立 queue timeout；超時回 503／`Retry-After` 並記錄非原文 outcome，防止卡住的
  模型讓未執行請求在服務內無限累積。每 replica 的等待區另有可設定筆數上限，滿載立即拒絕；
  兩者仍不是跨 replica 的完整 rate limiter。
- 關閉流程先撤除 readiness，追蹤並在有界寬限期內排空 timeout 後仍執行的模型工作；同步執行緒
  無法安全強制取消，超過應用寬限後由較長的容器終止期限提供最後邊界。
- CLI 與 registry 對設定、manifest、評測與訓練 JSON 採 16 MiB byte 上限並要求合法 UTF-8；
  異常大的控制面檔案不會先被無界載入記憶體。大型模型 artifact 仍以串流方式計算 SHA-256。
- 資料與續跑快取 JSONL 的單行上限為 1 MiB；正式資料超限或非 UTF-8 時 fail closed，快取則只
  丟棄損壞行並重新產生受影響案例，不讓一行殘片破壞後續可恢復結果。
- `MODELFORGE_LOG_INPUTS` 唯一合法值為 `false`；誤設 `true` 時服務拒絕啟動，API 不提供事件原文
  logging。repository 也忽略模型、資料生成物與輸出報告。
- llama.cpp 每-slot context 預設 2048；本機啟動腳本低於 1536 時 fail fast，避免輸出被靜默截斷。
- 安全政策不可因程式更新靜默漂移；部署預設鎖定 registry 已 attested 的 v2，實驗 v3 必須顯式選擇。
- 掛載新版 registry 後可設定 `MODELFORGE_REGISTRY_DEPLOYMENT_ROLE`，將 runtime metadata 綁定
  `released` 或 `previous-good`；缺少完整 measured attestation 或任一契約欄位不符即拒絕啟動。

## Secret and privacy checks

`scripts/privacy-check.ps1` 排除大型生成物後掃描常見 key、Bearer token、密碼賦值與使用者私人路徑。
完整驗收另執行 `pip check`、wheel 建置與 Web UI 封裝驗證；`pip check` 只驗依賴一致性，不等同
第三方套件 CVE 掃描。wheel 會無依賴安裝到一次性暫存目錄，再確認套件來源、FastAPI app 與公開
路由，最後只清理已驗證位於系統暫存根目錄下的資料夾。正式提交前同時檢查 Git staged diff。
不要把 `.env`、模型權重、客戶事件或任何 token 放入 Git。

## Residual risk

regex 不是完整語意安全分類器；本機 llama-server 只綁定 `127.0.0.1` 且沒有 API key。統一 API
雖提供 shared bearer key 基線，但不等同使用者身分、細粒度授權或 key rotation。真實部署仍需要
領域規則、TLS、rate limit、網路隔離、集中 audit 與事故應變流程。本 demo 沒有宣稱具備這些
外部基礎設施。
