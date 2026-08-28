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
  SHA-256 與模型、adapter、prompt、解碼設定指紋完全一致時才可沿用。
- `benchmark.py`：併發、P50/P95/P99、吞吐、RAM／整卡 VRAM 峰值取樣；HTTP streaming 可量
  TTFT，不支援時明確記 `null`。
- `registry.py`：candidate、promote、previous-good 與 rollback 的原子檔案 registry。

OpenAI-compatible HTTP adapter 使用 strict JSON Schema：八個輸出欄位皆 required，分類、動作與
reason code 使用有限允許清單。此約束在生成前套用；線上 bounded repair 與 fail-closed fallback
仍保留，但 raw 評測不把修補後結果算成原始格式成功。

服務不執行設備控制、不讀任意 URL 或用戶指定模型路徑，也預設不記錄事件原文。Kubernetes
範例把 API 與模型伺服器分開，讓 GPU server 可獨立縮放；`/readyz` 會檢查下游模型健康狀態。
