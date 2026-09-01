# Architecture

ModelForge-TW 將模型生成視為不可信輸入。請求先通過嚴格 Pydantic schema，再由選定引擎生成；
輸出經 bounded JSON repair、schema 驗證、證據逐字對齊與確定性風險政策後，才可回給呼叫端。

```text
Web / REST / OpenAI-compatible client
                 │
        FastAPI + request ID
                 │
       timeout + concurrency gate
                 │
  Transformers / llama.cpp HTTP / fixture
                 │
      parse → schema → safety policy
                 │
 structured response + Prometheus metrics
```

核心模組：

- `schemas.py`：輸入、輸出、資料與報告契約。
- `prompts.py`：minimal、optimized 與 SFT message contract。
- `safety.py`：不可信指令偵測、證據 grounding、風險向上升級與 fallback。
- `engines/`：相同 protocol 下的 fixture、Transformers 與 OpenAI-compatible HTTP adapter。
- `evaluation.py`：同資料 hash、release-gate SHA-256 的品質指標與 hard gate；Q4 另檢查相對 Q8
  的 macro-F1、raw schema 與 hard-gate regression。長時間評估逐筆寫入續跑快取；只有資料紀錄
  SHA-256 與模型、adapter、prompt、解碼、安全政策、評分契約指紋完全一致時才可沿用；評分器
  也拒絕重複 record ID，並檢查每筆結果的模型與推論契約 metadata 一致後才寫入快取及計分。
  報告 schema 另核對 SHA-256 格式、有限範圍 metric、case count／唯一 ID、gate value／判定與
  release 總結果；零案例 measured report 或缺少必要 gate 不得宣稱通過。
- `external_calibration.py`：驗證第三方來源 SHA-256、固定兩組不重疊抽樣、轉換 AI4I 衍生事件，
  並輸出 calibration／holdout manifest 與 combined leakage audit。
- `benchmark.py`：併發、P50/P95/P99、吞吐、RAM／整卡 VRAM 峰值取樣；HTTP streaming 可量
  TTFT，不支援時明確記 `null`。新版報告只有在每一筆量測請求都成功時才標為 passed；任一錯誤
  會保留樣本並讓 CLI 回傳非零狀態，不能把部分或全數失敗的壓測當作成功產物。
- `training.py`：固定 seed 的 completion-only QLoRA 與 merge；新版報告保存資料／設定／adapter
  hash、Python、Torch、CUDA、cuDNN、GPU capability、實體顯存與完整 ML 依賴版本，避免只鎖
  模型卻漏掉訓練 runtime 漂移。候選加入 registry 前會驗證報告 schema、measured evidence、
  immutable resolved revision、內嵌 config 雜湊、模型 identity、prompt 契約及本機資料 manifest；
  正式發布另要求至少一個已雜湊模型 artifact，不能發布只有 metadata 的空候選。
  訓練、評測、合併、載入探針與 Transformers 服務共用同一個已驗證 base-model commit 預設；
  revision 必須是 40–64 位 lowercase commit hash，`main`／tag 會在匯入大型 ML runtime 前 fail fast。
  未指定時的實測評估與訓練也統一使用現行 attestation 的 `optimized-v2`／192-token 生成契約；歷史
  A/B 實驗的 v1 標記只保留在既有證據與顯式選擇中。
  Transformers／llama-http 評測另強制提供安全格式的 `--model-version`；保留值 `unversioned` 不能
  進入引擎、評測報告或 registry。llama-http 也必須明確提供量化格式，`unknown` 不能進入 measured
  evidence；Transformers 則依實際 NF4／BF16 載入方式設定。載入 probe 由 immutable base commit
  推導可重現版本標記。
  Transformers inference metadata／evaluation 內的非空 model revision 同樣只接受 commit hash，且
  registry 必須確認它等於 training manifest 的 base revision；遠端 llama-http 無法證明時保留 `null`。
- `registry.py`：candidate、evaluation attestation、promote、previous-good 與 rollback 的原子檔案
  registry；切換前重新驗證 artifact 大小與 SHA-256，attestation 會鎖住評測資料 hash、引擎、
  模型 identity、evidence kind、生成、安全與評分契約。新候選缺少任一 attestation 欄位都不能
  加入；舊 manifest 缺少 identity 時只能讀取或沿用既有狀態，不能拿新版報告重新 attestation。
  評測／訓練報告與 artifact 只接受 canonical POSIX-style 相對路徑，絕對路徑、反斜線及 `..`
  穿越會在讀取或寫入 registry 時拒絕；新版 attestation 也保存兩份報告本身的 SHA-256。新 promote
  會重新要求完整 attestation 且只接受 measured evidence，並重新雜湊、解析評測與訓練報告、
  逐項比對 manifest identity／release 結果及模型 artifact，確認路徑未逃出 project root。
  候選加入也執行同一內容驗證，重新 evaluation attestation 時則立即核對新報告；fixture 可驗證
  機制但不能成為正式發布。rollback
  同樣採此規則；唯一相容例外是 schema 1.0 中八個新版 identity／報告雜湊欄位全部不存在、
  且早已被指定為 previous-good 的既有狀態，避免安全升級反而切斷已驗證的舊回退路徑；
  部分遷移的資料不放行。Registry 載入時另驗證 manifest key／ID、唯一 released status、released
  pointer、通過結果與 retired previous-good pointer 一致；人工改壞的狀態不能進入後續操作。
  線上服務可選擇綁定 `released` 或 `previous-good` 指標；啟用後會在啟動時逐欄比對實際引擎的
  measured identity、模型版本、量化、prompt、生成上限、輸出契約與安全政策，漂移即拒絕啟動；
  `/readyz` 也會重讀 pointer 與相同契約，運行中 registry 改版會先撤除此 Pod 流量，等待明確重啟。
  若另設定 artifact root，啟動還會重算評測／訓練報告與模型 artifact 的大小及 SHA-256；高頻 readiness
  不重讀大型權重。這能驗證 API Pod 看見的唯讀掛載副本，不等同遠端模型 server 的記憶體證明。
  Registry 1.4 將操作事件限制為 typed schema，並以 canonical JSON SHA-256 串接前一筆事件；
  1.0–1.3 可唯讀相容，下一次 mutation 才封入新鏈。這可偵測未同步重算的封存後改寫，但不是
  外部數位簽章或不可變 audit storage；能改寫檔案的攻擊者也能重算整條鏈。正式環境必須把 head digest／registry
  副本送往獨立受控稽核系統，才能對抗事後重算。`modelforge registry anchor` 會先完整驗證
  registry，再從同一次讀取匯出整份檔案 SHA-256、事件數、鏈驗證狀態與 head digest；舊版未封鏈
  registry 會明確回報 `event_chain_verified: false`，不會把相容讀取誤稱為已驗證。
  Manifest 的 base revision 必須是 40–64 位 lowercase commit hash；事件 action 也限制可帶的欄位，
  例如 evaluation attestation 缺 report，或 promoted 事件夾帶 report，都會在載入時拒絕。

資料 split／manifest、評測、比較、benchmark、訓練、merge、量化與 registry JSON 都先寫入同目錄
的唯一暫存檔，完成後再原子替換；Python 路徑會先 flush + fsync。中斷或 replace 失敗會保留上一份
完整檔案並清掉暫存檔。Registry 的 read-modify-write 另外持有跨程序單寫者檔案鎖，避免並行的
候選加入、attestation、promote、block 或 rollback 互相覆蓋；等候逾時會失敗而非冒險寫入，CLI
會以簡短錯誤與非零結束碼回報，不輸出內部 traceback，方便自動化流程明確重試或告警。
POSIX 在 replace 後另 fsync 父目錄，讓新 directory entry 也進入耐久邊界；Windows 沒有使用
Python 不可攜的目錄 handle 技巧，仍採同目錄暫存檔加原子 replace。

所有引擎的解析後 schema 都要求八個輸出欄位，並限制分類、動作與 reason code 為有限允許清單；
OpenAI-compatible HTTP adapter 另用 strict JSON Schema 在生成前套用相同約束。線上 bounded repair
與 fail-closed fallback 仍保留，但 raw 評測不把修補後結果算成原始格式成功。

對外 `/v1/chat/completions` 是刻意縮小的 OpenAI-compatible 子集：接受 `model`、恰好一筆
`user` message、`temperature=0`、`max_tokens` 與 `stream`。多輪或 system／assistant／tool 訊息、
未知欄位、未部署模型、非零 temperature，以及小於部署結構化輸出預算的 `max_tokens` 都明確
拒絕，避免表面接受卻忽略參數。推論路由在認證後只接受單一合法 JSON media type；所有寫入型
HTTP request 會先經過 byte-level body 上限，包含
沒有 `Content-Length` 的 chunked 傳輸；串流在完整 schema 與安全驗證後送出，並不宣稱是
Transformers token-level streaming。

服務不執行設備控制、不讀任意 URL 或用戶指定模型路徑，也預設不記錄事件原文。Kubernetes
範例把 API 與模型伺服器分開，讓 GPU server 可獨立縮放；`/readyz` 會檢查下游模型健康狀態，
下游回報未就緒、健康檢查拋錯或超過預設 4 秒都會轉成不含內部細節的 503。同一個尚未完成的
registry metadata／引擎健康工作會由後續探針共用，且兩段合計使用同一個 4 秒預算；慢速唯讀卷
不會同步阻塞 API event loop，也不因連續 probe 無界建立背景執行緒。Kubernetes timeout 略長於應用上限。
所有 HTTP 請求在 body 解析前取得或驗證 `X-Request-ID`；重複、非 ASCII 或格式錯誤的值不回顯，
而以新 server ID 回覆 400。成功、認證失敗、body 超限與 schema 錯誤都在 response header 帶同一
追蹤 ID，推論 body 內的 ID 也與 header 一致。
推論路由可由至少 32 字元的 shared bearer key 保護，Kubernetes 透過 Secret 注入且 config endpoint
只揭露是否啟用，不回傳 secret；這是內部服務基線，不宣稱取代正式 IAM 或細粒度授權。
同步模型推論逾時時會立即回傳保守 fallback，但底層執行緒無法強制取消；該工作真正結束以前仍
占用 concurrency slot，避免連續 timeout 繞過容量上限並堆積 GPU 工作。Prometheus 分開呈現
尚未回應的 HTTP request 與仍在執行的 engine job，因此 timeout 後的背景負載仍可觀測。等待
semaphore 另有預設 5 秒 queue timeout；容量未釋放時回 503 + `Retry-After` 並記
`capacity_rejected`。等待區另有預設 8 筆的硬上限，滿載時不等待 timeout 就立即拒絕，並以
`modelforge_queued_requests` 呈現當前排隊深度，避免請求在應用層無限堆積。
呼叫端取消等待中的 request 時會立即歸還 queue reservation；若取消發生在同步引擎已開始後，
HTTP request 雖結束，slot 與 engine-job gauge 仍保留到背景工作真正完成，並記為 `client_cancelled`。
服務關閉時會先讓 readiness 失敗，再在可設定的寬限期內等待已追蹤的背景推論；等待有上限，
且不宣稱能強制取消同步模型執行緒。Kubernetes 範例給 API 65 秒應用排空與 75 秒 Pod 終止寬限，
讓一般 60 秒下游 timeout 有完成空間；若底層工作仍不返回，最終由容器平台強制終止程序。

llama.cpp 的 `--ctx-size` 是所有 parallel slots 的總容量。部署設定把每-slot context 明確寫入 API
metadata 與評估報告；本機預設 8192／4＝2048 tokens，低於實測最低值時啟動失敗，避免容量截斷
被誤判為模型品質退化。

安全政策同樣由版本明確選擇：服務、`.env` 與 Kubernetes 預設 `risk-precedence-v2`，和 registry
attestation 一致；外部實驗的 v3 必須由 CLI／環境變數顯式指定，並進入 metadata 與快取指紋。
