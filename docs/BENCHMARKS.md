# Benchmarks

## 本機環境

- Windows 11、Python 3.11.9、12 logical CPUs、RAM 16,755,937,280 bytes。
- NVIDIA GeForce RTX 5060 Laptop GPU，8,151 MiB，compute capability 12.0；歷史 driver 610.88，
  2026-09-08 壓測 driver 616.64。
- PyTorch 2.13.0 + CUDA 13.0、Transformers 5.16.1。
- llama.cpp official Windows CUDA release b10665，commit
  `ca3d5a3e10d53f7ea672cb9b6178faca3e2807bc`。

歷史數字來自 2026-08-28／29，新預設設定另於 2026-09-08 本機執行。GPU 記憶體由
`nvidia-smi` 取樣時，報告明確標示
為整張卡的 host-wide reading，不宣稱是單一程序的精確歸因。
原有公開 Q8／Q4 報告是歷史 schema 1.0；新版 1.4 逐請求核對推論身份，並保存
prompt-token／output-token／TTFT 樣本覆蓋數、run ID 與有序輸入工作負載 hash。
這項 schema 改進不回填或改寫既有實測數字。

## 訓練

- QLoRA v5：1 epoch、53 optimizer steps、1,467.509 秒。
- train loss 0.235826；validation loss 0.017484。
- peak allocated 4,523,375,616 bytes。
- peak reserved 12,501,123,072 bytes；這是 WDDM／allocator 虛擬與快取統計，可能高於專用 VRAM，
  不解讀為實體使用量。

## 模型大小

| 檔案 | bytes | 相對 F16 |
|---|---:|---:|
| merged BF16 safetensors | 3,441,185,608 | — |
| GGUF F16 | 3,447,349,024 | 100.0% |
| GGUF Q8_0 | 1,834,426,144 | 53.2% |
| GGUF Q4_K_M | 1,107,408,672 | 32.1% |

三個 GGUF 與 merged safetensors 都保存 SHA-256；量化腳本使用同磁碟 ASCII hard-link bridge，避開
official Windows binary 無法解析中文 argv 路徑的限制，正式 artifact 仍保留在專案目錄。

## 224 筆完整評測延遲

| 版本 | mean | P50 | P95 | max | 品質 gate |
|---|---:|---:|---:|---:|---|
| QLoRA v5 / Transformers NF4 | 8.333 s | 8.036 s | 10.823 s | 12.704 s | pass |
| merged v5 / llama.cpp Q8_0 | 1.060 s | 1.012 s | 1.379 s | 1.474 s | pass |
| merged v5 / llama.cpp Q4_K_M | 0.825 s | 0.811 s | 0.984 s | 1.391 s | fail |

Q8 mean latency 是 NF4 路徑的 12.7%，描述上約 7.86 倍加速。相同契約下，Q4 比 Q8 再降低
22.2% mean latency，但 event macro-F1 只有 74.43%，所以不能以速度或檔案大小抵銷品質失敗。

## llama.cpp 併發基準

每個量化版本先 warm up 2 次，再於 challenge 請求池對 concurrency 1／2／4 各執行 12 次。
36/36 請求皆完成，錯誤率 0%。TTFT 由真正 streaming response 的第一個內容 chunk 量測。

| 版本 | concurrency | req/s | output tok/s | latency P50 | latency P95 | TTFT P50 | host GPU peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q8_0 | 1 | 0.808 | 78.7 | 1.186 s | 1.521 s | 210 ms | 3,134 MiB |
| Q8_0 | 2 | 1.417 | 139.7 | 1.421 s | 1.751 s | 236 ms | 3,134 MiB |
| Q8_0 | 4 | 2.116 | 201.9 | 1.649 s | 2.434 s | 404 ms | 3,134 MiB |
| Q4_K_M | 1 | 1.017 | 84.1 | 0.980 s | 1.072 s | 209 ms | 2,443 MiB |
| Q4_K_M | 2 | 1.566 | 130.7 | 1.269 s | 1.386 s | 292 ms | 2,443 MiB |
| Q4_K_M | 4 | 2.692 | 226.6 | 1.441 s | 1.542 s | 374 ms | 2,443 MiB |

上述壓測使用 llama-server 的 4 個 parallel slots、4096 總 context、全部 GPU layers 與 flash
attention。每 slot context 為 1024，當時足以容納內部資料的約 700–900 token prompt；這份吞吐量
結果仍有效，但後續外部校準證明它不是安全的通用預設。

## Context 容量回歸

AI4I 外部輸入的 prompt 約 967 tokens；再加 192-token output budget 後，4096 total context／4 slots
會在 1024-token 邊界截斷。第一輪 Q8 的 98/98 raw schema 因此失敗，server log 顯示
`truncated=1`。改為 8192／4（每 slot 2048）後沒有截斷，transform-v1 raw schema 回升至 77.55%，
確認剩餘問題才是模型輸出與分類，而非容量。

目前 `serve-gguf.ps1` 與 Kubernetes baseline 改用 8192／4，API／報告宣告每 slot 2048；本機腳本
拒絕低於 1536 的 slot。這項變更沒有回寫上方既有 4096-context 併發數字；若要宣稱新設定的吞吐量，
必須使用獨立的 8192-context benchmark。2026-09-08 已完成這項獨立壓測，見下一節。

## 8192-context 暖機後壓測（2026-09-08）

Q8／Q4 各使用相同 test pool 的前 100 個輸入，在 client concurrency 1／2／4／8 各跑 100 次；
每個版本共 400/400 完成、零請求錯誤，prompt／output token 與 TTFT 樣本覆蓋皆完整。
服務固定為 4 slots、每 slot 2048，client concurrency 8 是排隊負載，不是 8 個模型 slots。

| 版本 | client concurrency | req/s | output tok/s | latency P95 | TTFT P50 |
|---|---:|---:|---:|---:|---:|
| Q8_0 | 4 | 2.3000 | 201.1817 | 2.274 s | 343 ms |
| Q8_0 | 8 | 2.9179 | 261.8781 | 3.308 s | 1,326 ms |
| Q4_K_M | 4 | 2.5671 | 212.5322 | 1.820 s | 321 ms |
| Q4_K_M | 8 | 3.3656 | 279.5495 | 2.624 s | 1,213 ms |

完整四級結果、輸入指紋、run ID、環境與原始報告 SHA-256 見
[新設定壓測證據](current-context-benchmarks.json)。每級重複同一批 100 個輸入（機械 30、電氣 30、
溫度 30、壓力 10），不是完整 210 筆 test，更不是 400 個獨立案例。各級依序在同一暖機 server
執行，未重置預設 prefix cache；新舊測試的資料與 driver 不同，不把差異單獨歸因於 context。
Q4 的品質 blocked 決策與正式 registry 均未變更。

## 未執行邊界

- benchmark schema 1.4 延續 `passed` 與 `failure_reason`；只有所有量測請求零錯誤才成功，
  否則腳本回傳非零狀態。`error_samples` 只保存穩定錯誤代碼與例外類型，不保存可能含私有端點、
  路徑或憑證片段的原始例外訊息。既有 Q8／Q4 報告為 schema 1.0，但公開摘要已核對皆為
  36/36、0 error。
- `passed` 是推論完成／身份檢查結果，不是 JSON 原始品質或分類 gate；政策 fallback 仍可能算
  完成。效能報告不授權發布，也不取代 224 筆品質評測。
- 這些是暖機後、封閉式固定併發 client 對模型服務的負載測試；每個 worker 收到回應後才送
  下一筆。不含統一 API 的認證／限流成本、冷啟動、固定到達率或長時間 soak 測試。
- Transformers adapter 沒有 token-level streaming，因此其 TTFT 不可用，不填入誤導性的 0。
- Docker／Compose／Kubernetes 因本機沒有 runtime，只做靜態設定、YAML 解析與程式測試。
- vLLM、SGLang、ONNX Runtime、TensorRT-LLM 未在本機執行，不列效能數字。
- 首次 Qwen 載入曾包含模型下載時間，不能與純生成延遲混為一談。
