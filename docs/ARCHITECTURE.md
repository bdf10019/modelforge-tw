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
  SHA-256 與模型、adapter、prompt、解碼、安全政策、評分契約指紋完全一致時才可沿用。
- `external_calibration.py`：驗證第三方來源 SHA-256、固定兩組不重疊抽樣、轉換 AI4I 衍生事件，
  並輸出 calibration／holdout manifest 與 combined leakage audit。
- `benchmark.py`：併發、P50/P95/P99、吞吐、RAM／整卡 VRAM 峰值取樣；HTTP streaming 可量
  TTFT，不支援時明確記 `null`。
- `training.py`：固定 seed 的 completion-only QLoRA 與 merge；新版報告保存資料／設定／adapter
  hash、Python、Torch、CUDA、cuDNN、GPU capability、實體顯存與完整 ML 依賴版本，避免只鎖
  模型卻漏掉訓練 runtime 漂移。
- `registry.py`：candidate、evaluation attestation、promote、previous-good 與 rollback 的原子檔案
  registry；切換前重新驗證 artifact 大小與 SHA-256，attestation 會鎖住生成、安全與評分契約。

OpenAI-compatible HTTP adapter 使用 strict JSON Schema：八個輸出欄位皆 required，分類、動作與
reason code 使用有限允許清單。此約束在生成前套用；線上 bounded repair 與 fail-closed fallback
仍保留，但 raw 評測不把修補後結果算成原始格式成功。

對外 `/v1/chat/completions` 是刻意縮小的 OpenAI-compatible 子集：接受 `model`、`messages`、
`temperature=0`、`max_tokens` 與 `stream`。未知欄位、未部署模型、非零 temperature，以及小於部署
結構化輸出預算的 `max_tokens` 都明確拒絕，避免表面接受卻忽略參數。串流在完整 schema 與安全
驗證後送出，並不宣稱是 Transformers token-level streaming。

服務不執行設備控制、不讀任意 URL 或用戶指定模型路徑，也預設不記錄事件原文。Kubernetes
範例把 API 與模型伺服器分開，讓 GPU server 可獨立縮放；`/readyz` 會檢查下游模型健康狀態。
同步模型推論逾時時會立即回傳保守 fallback，但底層執行緒無法強制取消；該工作真正結束以前仍
占用 concurrency slot，避免連續 timeout 繞過容量上限並堆積 GPU 工作。Prometheus 分開呈現
尚未回應的 HTTP request 與仍在執行的 engine job，因此 timeout 後的背景負載仍可觀測。

llama.cpp 的 `--ctx-size` 是所有 parallel slots 的總容量。部署設定把每-slot context 明確寫入 API
metadata 與評估報告；本機預設 8192／4＝2048 tokens，低於實測最低值時啟動失敗，避免容量截斷
被誤判為模型品質退化。

安全政策同樣由版本明確選擇：服務、`.env` 與 Kubernetes 預設 `risk-precedence-v2`，和 registry
attestation 一致；外部實驗的 v3 必須由 CLI／環境變數顯式指定，並進入 metadata 與快取指紋。
