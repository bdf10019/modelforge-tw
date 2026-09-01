# ModelForge-TW

繁體中文設備事件模型適配、量化與推論發布平台。它以同一封存評測集比較 Base、prompt-only、
QLoRA、合併權重與 GGUF 量化版本，並用實測報告回答品質、延遲、吞吐、VRAM、檔案大小與安全
發布取捨。

> 專案狀態：公司實務版已完成本機端到端驗證。正式 registry 目前發布完成 policy-v2 attestation
> 的 Qwen3-1.7B QLoRA v5（Transformers NF4）；llama.cpp Q8 通過絕對門檻並保留為 previous-good，
> Q4_K_M 因完整品質門檻失敗而封鎖。訓練／正式 registry 資料為原創合成內容；另有授權清楚的
> UCI AI4I 第三方合成外部校準，不包含公司、客戶或私人資料。

## 核心成果

本作品不是 RAG 聊天機器人。輸入是繁體中文設備事件，輸出是經 JSON Schema、原文證據對齊與
確定性安全政策驗證的事件分流結果。模型不控制設備；高風險、資訊不足、輸出無效或逾時都會
保守轉人工。

| 版本（224 筆 test + challenge） | raw JSON | event macro-F1 | severity macro-F1 | 安全硬門檻 | mean latency | 發布結果 |
|---|---:|---:|---:|---:|---:|---|
| QLoRA v5 / Transformers NF4 | 99.55% | 86.24% | 95.35% | 全過 | 8.333 s | released |
| merged v5 / llama.cpp Q8_0 | 100.00% | 85.16% | 85.34% | 全過 | 1.060 s | 通過；rollback 演練後 retired |
| merged v5 / llama.cpp Q4_K_M | 100.00% | 74.43% | 91.04% | 全過 | 0.825 s | blocked |

安全硬門檻為必要欄位 100%、high/critical 人工複核召回 100%、證據精度 100% 與注入抵抗
100%。Q8 的平均延遲是 NF4 路徑的約 12.7%，但 severity macro-F1 描述性差距為 10.01 個百分點，
因此保留較保守的 NF4 v5 為正式發布。Q8 與 Q4 的正式公平比較鎖定相同資料、門檻、prompt、192-token
上限、`json-schema-v2`、`risk-precedence-v2` 與 `release-metrics-v2`；Q4 雖更快，event macro-F1
相對 Q8 退化 10.73 個百分點，不能發布。
完整條件與逐階段失敗紀錄見 [評測文件](docs/EVALUATION.md)、[錯誤分析](docs/ERROR_ANALYSIS.md)與
[證據索引](docs/EVIDENCE_INDEX.md)。

外部校準另外揭露現行 Q8 v2 的跨來源缺口（external event F1 40.82%）。專用 v5 雖在首次
row-disjoint holdout 得到 98/98 系統通過，卻讓內部 event F1 降至 69.72%，而且 16.33% holdout
案例需要安全層介入，因此 v5 被拒絕、registry 不變。完整方法與誠實邊界見
[外部校準報告](docs/EXTERNAL_CALIBRATION.md)。

若以求職面試為目的，可直接依照 [10 分鐘展示走查](docs/INTERVIEW_WALKTHROUGH.md) 操作；它把
現場 demo、量化取捨、外部回歸與誠實邊界排成一條可在有限時間內驗證的故事線。

## 快速驗證

需求：Python 3.11、PowerShell 7。fixture 模式完全離線且不下載模型。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
.\scripts\verify.ps1
modelforge generate-data
modelforge evaluate --engine fixture-adapted
modelforge serve
```

開啟 `http://127.0.0.1:8000`；OpenAPI 位於 `http://127.0.0.1:8000/docs`。fixture 僅驗證資料、
API、安全 gate、評測與 registry 機制，其數字不代表 LLM 品質，也不能被 registry 正式發布。
同一支 `verify.ps1` 會在 Windows venv 與 Linux CI 執行資料稽核、靜態檢查、測試、wheel 安裝冒煙、
隱私掃描與可用的本機證據核對。

## 服務與發布控制

- 推論請求只接受單一 JSON media type；body 預設 32 KiB、模型 server 回應預設 256 KiB。一般 JSON、
  chunked body 與無換行 SSE 都依實收 bytes 停止，不在解析後才補救。
- 每 replica 有 concurrency、5 秒 queue timeout 與有限 queue depth；逾時後不可取消的同步模型工作
  仍占用槽位，排隊滿載立即 503，Prometheus 分開呈現 HTTP、queue 與 engine job。
- 關閉時先撤 readiness，再在有界寬限內排空背景工作；Kubernetes 的 Pod 終止期限刻意更長。
- 推論路由支援至少 32 字元的 shared bearer key，Kubernetes baseline 缺少 Secret 時不啟動；只有
  loopback／Compose demo 保留無認證模式。輸入原文 logging 不可開啟；這仍不等同企業 IAM、TLS、
  跨 replica rate limit 或集中稽核。
- 新版 registry mutation 會把 typed event 封入 SHA-256 前向鏈；線上服務也可選擇綁定完整 attestation
  的 `released`／`previous-good` runtime identity。現有實測 registry 是可相容讀取的 1.0 歷史資料，
  在受控的新 mutation 前不冒充已封鏈或已具新版線上綁定欄位。
- `modelforge registry anchor` 可唯讀匯出已驗證快照的整檔 SHA-256 與事件鏈 head，交由外部 WORM／
  稽核系統留存；本機 hash chain 本身不取代外部不可變錨定。

## 真實模型流程

```powershell
.\scripts\bootstrap.ps1 -Profile ml -DownloadModel
.\scripts\train.ps1
.\scripts\evaluate.ps1 -Full
modelforge merge --adapter artifacts\qwen3-1.7b-triage-lora-v5\final-adapter `
  --output artifacts\qwen3-1.7b-triage-merged
.\scripts\prepare-llama.ps1
.\scripts\quantize.ps1
.\scripts\serve-gguf.ps1 -Quantization q8_0 -Port 8081
```

另開終端機啟動統一 API：

```powershell
.\scripts\run-demo.ps1 -Engine llama-http -ModelId modelforge-tw-q8_0 `
  -ModelVersion qwen3-1.7b-merged-v5-q8_0-policy-v2 -Quantization q8_0
```

大型權重與實測輸出皆位於 Git 忽略目錄；模型 revision、資料 manifest、設定與 artifact SHA-256
會寫入報告。長時間評測逐筆落盤，可在指紋完全一致時安全續跑。

## 實測矩陣

| 組別 | 權重 | Prompt | 引擎／精度 | 狀態 |
|---|---|---|---|---|
| A | Qwen3-1.7B 未修改 | minimal-v1 | Transformers NF4 | challenge fail |
| B | 同 A | optimized-v1 | Transformers NF4 | challenge fail |
| C | QLoRA v5 | optimized-v2 | Transformers NF4 | full pass / released |
| D | merged v5 | optimized-v2 | Transformers BF16 | challenge fail（raw schema） |
| E | merged v5 | optimized-v2 | llama.cpp Q8_0 | full pass / rollback drill |
| F | merged v5 | optimized-v2 | llama.cpp Q4_K_M | full fail / blocked |

所有效能數字均來自本機真實執行；fixture、合成資料、估算與未執行項目都明確分開。Docker 與
Kubernetes 已完成靜態設定與測試，但本機未安裝對應 runtime，因此不宣稱已在叢集部署。

## 專案結構

```text
src/modelforge_tw/       schema、資料、prompt、引擎、評測、registry、API 與 Web UI
configs/                 資料、訓練、benchmark 與發布 gate 設定
data/source/             原創人工 challenge source
data/generated/          可重建 split；Git 忽略
data/external/           UCI 原始檔與可重建 calibration／holdout；Git 忽略
artifacts/               adapter、合併權重與 GGUF；Git 忽略
outputs/                 評測、benchmark、registry 與 trace；Git 忽略
infra/k8s/               Deployment、Service、ConfigMap 與 training Job
scripts/                 環境、訓練、評測、量化、服務、驗證與隱私檢查
tests/                   單元、API、資料洩漏、安全、registry 與端到端測試
```

架構、資料、模型、效能、安全與展示方式分別見 `docs/`。程式使用 [MIT License](LICENSE)，原創
合成資料使用 CC BY 4.0；模型權重不隨 repository 散布。第三方來源與鎖定版本見
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
