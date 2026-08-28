# ModelForge-TW

繁體中文設備事件模型適配、量化與推論發布平台。它以同一封存評測集比較 Base、prompt-only、
QLoRA、合併權重與 GGUF 量化版本，並用實測報告回答品質、延遲、吞吐、VRAM、檔案大小與安全
發布取捨。

> 專案狀態：公司實務版已完成本機端到端驗證。正式 registry 目前發布 Qwen3-1.7B QLoRA v5
> （Transformers NF4）；llama.cpp Q8 通過絕對門檻並完成 rollback 演練，Q4_K_M 因完整品質門檻
> 失敗而封鎖。資料皆為原創合成內容，不包含公司、客戶或私人資料。

## 核心成果

本作品不是 RAG 聊天機器人。輸入是繁體中文設備事件，輸出是經 JSON Schema、原文證據對齊與
確定性安全政策驗證的事件分流結果。模型不控制設備；高風險、資訊不足、輸出無效或逾時都會
保守轉人工。

| 版本（224 筆 test + challenge） | raw JSON | event macro-F1 | severity macro-F1 | 安全硬門檻 | mean latency | 發布結果 |
|---|---:|---:|---:|---:|---:|---|
| QLoRA v5 / Transformers NF4 | 99.55% | 87.53% | 95.49% | 全過 | 9.432 s | released |
| merged v5 / llama.cpp Q8_0 | 100.00% | 85.16% | 85.34% | 全過 | 0.975 s | 通過；rollback 演練後 retired |
| merged v5 / llama.cpp Q4_K_M | 100.00% | 74.43% | 91.04% | 全過 | 0.829 s | blocked |

安全硬門檻為必要欄位 100%、high/critical 人工複核召回 100%、證據精度 100% 與注入抵抗
100%。Q8 的平均延遲是 NF4 路徑的約 10.3%，但 severity macro-F1 描述性差距為 10.15 個百分點，
因此演練後回滾至較保守的 NF4 v5。Q8 與 Q4 的正式公平比較鎖定相同資料、門檻、prompt、192-token
上限與 `json-schema-v2`；Q4 雖更快，event macro-F1 相對 Q8 退化 10.73 個百分點，不能發布。
完整條件與逐階段失敗紀錄見 [評測文件](docs/EVALUATION.md)、[錯誤分析](docs/ERROR_ANALYSIS.md)與
[證據索引](docs/EVIDENCE_INDEX.md)。

## 快速驗證

需求：Python 3.11、PowerShell 7。fixture 模式完全離線且不下載模型。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
modelforge generate-data
modelforge evaluate --engine fixture-adapted
modelforge serve
```

開啟 `http://127.0.0.1:8000`；OpenAPI 位於 `http://127.0.0.1:8000/docs`。fixture 僅驗證資料、
API、安全 gate、評測與 registry 流程，其數字不代表 LLM 品質。

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
  -ModelVersion qwen3-1.7b-merged-v5-q8_0-json-schema-v2 -Quantization q8_0
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
artifacts/               adapter、合併權重與 GGUF；Git 忽略
outputs/                 評測、benchmark、registry 與 trace；Git 忽略
infra/k8s/               Deployment、Service、ConfigMap 與 training Job
scripts/                 環境、訓練、評測、量化、服務、驗證與隱私檢查
tests/                   單元、API、資料洩漏、安全、registry 與端到端測試
```

架構、資料、模型、效能、安全與展示方式分別見 `docs/`。程式使用 [MIT License](LICENSE)，原創
合成資料使用 CC BY 4.0；模型權重不隨 repository 散布。第三方來源與鎖定版本見
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
