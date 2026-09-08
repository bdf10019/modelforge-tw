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
- 解析失敗、timeout、engine exception 都回 unknown + collect evidence + review required；服務端重建的
  fallback metadata 仍保留已部署模型的 immutable revision 與 context-window 身分，不信任失敗輸出
  自報的 metadata，也不讓保守回應失去部署可追溯性。
- 所有引擎解析後都拒絕 allowlist 以外的 event、severity、safe action 與 reason code；llama.cpp
  路徑還會在生成前用 strict JSON Schema 約束同一份契約。
- `incident_text`、asset 與 signal 文字都視為不可信資料並納入注入偵測；signal 字串限制 128 字，
  數字必須是有限值且在明確範圍內，控制字元會在進入 prompt 前拒絕。
- HTTP body 預設上限 32 KiB；有無 `Content-Length` 都會依實收 bytes 計數，超限先回 413，
  不讓大型 payload 進入 JSON／Pydantic 解析。重複 Content-Length 或同時宣告 Content-Length 與
  Transfer-Encoding 的歧義 framing 會在讀取前回 400；上限可設定但限制在 1 KiB 至 1 MiB。
- 推論路由只接受單一 JSON `Content-Type`（`application/json` 或 `application/*+json`）；缺少、重複、
  非 JSON、畸形參數或非 UTF-8 charset 都在 body 解析前拒絕。啟用 bearer key 時仍先認證，避免未授權
  呼叫端利用格式錯誤探測受保護路由。Request `Content-Encoding` 只接受省略或 `identity`；gzip 等壓縮
  payload 在讀取前回 415，避免應用與前置代理對解壓前後 byte 上限有不同解讀。
- llama.cpp 一般與 SSE 回應都以解碼後 bytes 計數，預設上限 256 KiB；`Content-Length` 超限會
  在讀 body 前拒絕，chunked／壓縮回應則依實收解碼內容停止，無換行事件也不能無界累積。
- 模型 server 只接受具有 host、無內嵌帳密／query／fragment 的 HTTP(S) base URL；client 固定
  `trust_env=false`，事件原文不會因程序環境中的 `HTTP_PROXY`／`HTTPS_PROXY` 意外繞送代理。
- 設定 `MODELFORGE_API_KEY` 後，兩個推論路由都要求標準 `Authorization: Bearer ...`；key 至少
  32 字元、無空白的可列印 ASCII，以 constant-time 比對，不出現在 config response；
  文字設定錯誤亦隱藏原始輸入。Kubernetes baseline 要求 Secret
  存在，遺漏時 Pod 不啟動，repository 不保存值；未設定模式只保留 loopback／Compose demo 相容性，
  Compose 的 published port 明確綁定 `127.0.0.1`，並移除全部 Linux capabilities。
- 展示頁只在 config 宣告需要認證時顯示 password 欄位，將 bearer key 放進當次 request header；不寫入
  URL、localStorage、sessionStorage、結果 metadata 或事件紀錄。正式環境仍應由短效身分憑證取代共用 key。
- 推論／config 回應與認證錯誤都帶 `Cache-Control: no-store`；展示頁限制同源 script／style／連線，
  禁止 object、base URI 與 frame ancestor，所有 HTTP 回應另加 nosniff、deny framing、
  no-referrer 與 camera／microphone／geolocation 權限限制。
- `X-Request-ID` 在 JSON 解析前限制為 1–96 字元安全字元；重複或非法值回 400 且不回顯。所有 HTTP
  response 都由 server 附追蹤 ID，便於關聯失敗而不需要記錄事件原文。
- semaphore、timeout、request ID 驗證、Prometheus outcome 與 readiness 檢查；readiness 有獨立 4 秒
  總上限，registry metadata 與引擎健康都在同一背景檢查內；卡住時後續探針共用該工作，不阻塞
  event loop 或重複堆積執行緒。
- timeout 後仍保留同步模型的容量槽，直到不可取消的背景推論真正結束，避免資源耗盡。
- 等待容量另有獨立 queue timeout；超時回 503／`Retry-After` 並記錄非原文 outcome，防止卡住的
  模型讓未執行請求在服務內無限累積。每 replica 的等待區另有可設定筆數上限，滿載立即拒絕；
  兩者仍不是跨 replica 的完整 rate limiter。
- Client 取消排隊會釋放 queue 計數；若同步引擎已開始，仍保留 slot 到背景工作完成，避免斷線或取消
  被用來繞過 concurrency 上限。兩種情況都記錄 `client_cancelled`，不記事件原文。
- 關閉流程先撤除 readiness，追蹤並在有界寬限期內排空 timeout 後仍執行的模型工作；同步執行緒
  無法安全強制取消，超過應用寬限後由較長的容器終止期限提供最後邊界。
- CLI 與 registry 對設定、manifest、評測與訓練 JSON 採 16 MiB byte 上限並要求合法 UTF-8；
  異常大的控制面檔案不會先被無界載入記憶體。Registry、adapter lineage 與 training run contract
  驗證都只讀取一次，同一份 bounded bytes 同時計算 SHA-256 與解析，避免驗 hash 後重新開檔造成
  內容替換競態；大型模型 artifact 仍以串流方式計算 SHA-256。
- 訓練在匯入 ML runtime 前 fail closed 驗證 dataset manifest：audit 必須通過、combined hash 與
  split counts 必須一致，所有 JSONL 必須是具名 regular file 且大小、SHA-256、筆數完全相符。
- Checkpoint resume 由原子 run contract 綁定 model、dataset manifest 與完整 training config hash；
  contract 缺失、被改寫、lineage 不符、checkpoint 名稱異常或明確停用 resume 都會在 ML 載入前拒絕。
  Training report 1.2、adapter runtime 與 registry 會再驗 contract 檔案 hash，不接受事後自稱的續跑來源。
  新 registry 候選必須帶 runtime identity contract，並拒絕缺 tokenizer revision／run contract 的 1.0／1.1
  training report；舊 registry 仍可依原 schema 唯讀驗證。
- 訓練、評測、合併、probe 與 Transformers serving 的 base revision 必須是 40–64 位 lowercase
  commit hash；預設統一鎖定已驗證 commit，浮動的 `main`／tag 在載入模型或 ML runtime 前拒絕。
- 訓練與 merge 載入模型後會把 loader 回報的 resolved commit 與要求值逐字比對；缺失、格式錯誤或
  不相等都停止，training attestation 載入時也會再次核對，避免快取或下載層漂移被報告掩蓋。
- Merge 必須顯式提供 training report；載入 ML runtime 前逐項核對 base identity 與 adapter regular
  files 的集合、大小及 SHA-256，資料、report、run contract 與 adapter 路徑的任一層 symlink／
  Windows reparse point 都拒絕，也不接受額外檔、缺檔或被替換的 adapter。
- Merge output 必須是全新或空的 regular directory；既有檔案會在 ML runtime 載入前拒絕，避免舊
  safetensors shard、tokenizer 或 config 殘留並混入新 artifact／後續量化。
- 本機 Transformers 的 tokenizer assets 必須全部來自指定 Hugging Face snapshot commit；訓練、merge
  與 serving 都執行此檢查，serving 並核對載入後模型 config 的 `_commit_hash`，防止權重／tokenizer
  或服務快取來源分裂。
- 直接 PEFT serving、evaluate 與 benchmark 不接受單獨 adapter 路徑；training report 必須成對提供並
  在載入 ML runtime 前驗證，續跑 cache 另綁定該報告 SHA-256，不能沿用不同訓練 lineage 的結果。
- 新 training／merge report 只存以報告所在目錄為基準的 POSIX-style 運行路徑；不寫入使用者名稱、
  磁碟根目錄或專案絕對路徑，跨磁碟時只保留檔名並以 SHA-256 作內容權威。
- Measured evaluate／benchmark 必須顯式提供 `--model-version`；`unversioned`、控制字元與不安全識別字
  不能進入引擎、推論 metadata、評測報告或 registry，避免不同 artifact 共用模糊身份。
- llama-http measured 流程另強制顯式量化格式；`unknown` 或不安全識別字不能進入 runtime metadata、
  report 或 registry。Transformers 量化則由實際 `load_in_4bit` 選項產生，不信任外部標籤。
- 評測中的非空 model revision 必須是 immutable commit，registry 會把它和 training base revision
  逐值比對；遠端模型 API 未揭露可驗證 commit 時保存 `null`，不以版本字串假裝來源證明。
- 正式評測只讀取 release-gate 設定一次，同一份 bounded UTF-8 snapshot 同時套用門檻並計算寫入報告的
  SHA-256，避免執行期間換檔讓規則與 attestation hash 分離。
- Prompt 與 safety policy 版本是封閉且共用的實作清單；runtime metadata、評測報告與 registry
  manifest 只接受已支援值，不讓拼字錯誤或未實作政策進入發布契約。
- Evaluation schema 與 scoring contract 同樣 fail closed；未知 schema 或 release-metrics 版本不能
  被目前程式解析成已理解的發布證據，舊版 1.0–1.5 報告仍可按已知欄位唯讀驗證。
- Evaluation report 會從 cases 重算可獨立驗證的 accuracy、macro-F1、case pass rate 與完整延遲摘要，
  並要求 `passed` 與 errors 一致；發布通過時不接受缺少 actual output 的案例。
- 新量化流程在轉檔前會重驗 merge report 與其每個 regular source file，核對 llama.cpp commit、乾淨的
  source tree 與 quantizer executable SHA-256，並把 lineage 寫入報告；來源或工具漂移會在產生 GGUF 前停止。
- 量化 source 與 output 必須位於專案目錄內，output 必須是全新或空的 regular directory，兩種路徑都不
  接受 symlink／Windows reparse point；Q8／Q4 移入正式名稱時
  不使用強制覆寫，既有實測 GGUF 不會被一次重跑悄悄取代。
- llama.cpp 工具準備會核對壓縮檔 hash、解壓後完整檔案清單與 quantizer／server executable hash；
  啟動 server 不遞迴搜尋同名檔，還會重驗 GGUF 與 quantization report，工具或模型漂移均先停止。
- Kubernetes 線上 Pod 與 training Job 都以 UID 10001 強制 non-root、不掛 service-account token、使用
  RuntimeDefault seccomp、禁止提權並移除全部 capabilities；training root filesystem 唯讀，只有
  artifact、可重建資料、cache 與暫存目錄可寫。
- 資料與續跑快取 JSONL 的單行上限為 1 MiB；正式資料超限或非 UTF-8 時 fail closed，快取則只
  丟棄損壞行並重新產生受影響案例，不讓一行殘片破壞後續可恢復結果。
- `MODELFORGE_LOG_INPUTS` 唯一合法值為 `false`；誤設 `true` 時服務拒絕啟動，API 不提供事件原文
  logging。repository 也忽略模型、資料生成物與輸出報告。
- llama.cpp 每-slot context 預設 2048；本機啟動腳本低於 1536 時 fail fast，避免輸出被靜默截斷。
- 安全政策不可因程式更新靜默漂移；部署預設鎖定 registry 已 attested 的 v2，實驗 v3 必須顯式選擇。
- 掛載新版 registry 後可設定 `MODELFORGE_REGISTRY_DEPLOYMENT_ROLE`，將 runtime metadata 綁定
  `released` 或 `previous-good`；缺少完整 measured attestation 或任一契約欄位不符即拒絕啟動。
  另設定 `MODELFORGE_REGISTRY_ARTIFACT_ROOT` 時，啟動會串流重算掛載的 evaluation／training report
  與所有模型 artifact 大小及 SHA-256，而且在引擎初始化前完成；初始化後會重驗 pointer／metadata，
  防止兩階段之間切版。readiness 只重驗 metadata，避免探針反覆讀取 GB 級檔案。

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
