# 10 分鐘面試展示走查

## 展示目標

用一條可驗證的故事線說明：這不是只有微調 notebook 的作品，而是從資料契約、模型適配、量化
比較、安全邊界到發布決策的完整系統。全程只說本機實測或文件明確標示的事實。

## 展示日前驗證

```powershell
.\scripts\verify.ps1
```

完整驗證包含測試、封裝與證據核對，不假設能在面試前兩分鐘內跑完。應在展示日前完成，並把
結果記入 `MVP_ACCEPTANCE.md`。

## 現場 2 分鐘準備

```powershell
.\scripts\run-demo.ps1 -Engine fixture
```

確認 `http://127.0.0.1:8000`、`/readyz` 與 `http://127.0.0.1:8000/docs` 可開啟。fixture 只用來
展示 API、schema 與安全流程；畫面上的 `evidence_kind=fixture` 必須保留，不把它說成 LLM 實測。
真實模型若已在本機準備好，再另用 Q8 demo；面試不應把時間賭在重新下載或載入權重。

## 0:00–1:00｜先講問題與邊界

一句話：ModelForge-TW 把繁體中文設備事件轉成可稽核的結構化分流結果，模型不控制設備；高風險、
未知、無效輸出或逾時都交給人工。

打開首頁，指出三個設計：JSON Schema、輸入原文 evidence、有限 safe-action allowlist。主動說明資料
是原創合成加第三方合成 AI4I，不是真實工廠準確率或安全認證。

## 1:00–3:00｜現場展示安全契約

在 Web UI 依序執行：

1. 「控制櫃突然出現火花並伴隨焦味。」：應升級為 critical、要求人工複核，動作不得被缺資料降級。
2. 資訊不足範例：應保守標記 unknown／review required，不自行補造門檻。
3. 指令注入範例：事件文字仍被當成資料，服務端另以確定性規則標記注入並要求複核。

重點不是展示回答很像人，而是展示模型輸出即使不可信，服務仍會驗證 evidence、enum、reason code
與風險下限。

## 3:00–5:00｜用數字說明適配與量化取捨

打開 [`EVALUATION.md`](EVALUATION.md) 或 README 的核心成果表：

- NF4 QLoRA v5：event macro-F1 86.24%、severity macro-F1 95.35%，正式 released。
- Q8：event macro-F1 85.16%，平均延遲 1.060 秒，通過絕對門檻並保留 previous-good。
- Q4：平均延遲降到 0.825 秒，但 event macro-F1 74.43%，相對 Q8 退化 10.73 個百分點，因此 blocked。

一句結論：速度與檔案大小是優化目標，不是發布豁免；hard gate 與主任務品質都有否決權。

## 5:00–7:00｜展示一次「高分也拒絕」的決策

打開 [`EXTERNAL_CALIBRATION.md`](EXTERNAL_CALIBRATION.md)：

- AI4I calibration 與 holdout 各 98 筆，來源列不重疊，來源、轉換、抽樣與輸出都有 hash。
- prompt v5 在首次 holdout 得到 98/98 政策後通過，但 raw evidence precision 只有 84.85%，16.33%
  案例需要安全層介入。
- 同一 prompt 回到內部 224 筆時 event F1 降到 69.72%、unknown accuracy 降到 45.16%。

因此不建立 v5 manifest、不改 registry。這段最能說明作品有處理 holdout 污染、後處理歸因與回歸
否決，而不是只挑最好看的分數。

## 7:00–8:30｜展示容量事故如何被轉成契約

外部 prompt 約 967 tokens；原本 llama.cpp 4096 total context／4 slots 只有每 slot 1024，連同
192-token output budget 會被截斷。修正後 baseline 為 8192／4＝2048；啟動腳本低於 1536 直接
失敗，context 也進入報告 metadata 與快取指紋。

一句結論：評測失敗先分辨模型、資料、解碼與容量問題；不能把基礎設施截斷當成模型品質。

## 8:30–10:00｜收在發布治理與可重現性

打開 [`EVIDENCE_INDEX.md`](EVIDENCE_INDEX.md) 與 [`measured-evidence.json`](measured-evidence.json)：

- 公開快照保存去識別指標、run ID、資料／gate hash 與完整推論契約。
- 本機若有 Git 忽略的原始報告，`verify-local-evidence.ps1` 會逐欄核對 6 份報告與外部 manifest。
- registry 的 released、previous-good、blocked 與 rollback 都是驗證 artifact hash 後的狀態轉移。
- 線上層同時限制 request／model-response bytes、等待秒數與 queue 筆數；timeout 背景工作仍占容量，
  shutdown 只做有界排空、不假裝能取消同步 GPU 工作。
- 新 mutation 使用 typed hash-chained registry event；完整新版 attestation 可在服務啟動時綁定 runtime
  identity。現有 1.0 實測 registry 只宣稱相容讀取，不倒填成已在當時產生的 1.4 證據。
- Docker／Kubernetes 只有靜態設定與測試，本機沒有可用 runtime，因此不宣稱叢集部署實績。

最後一句：下一個可發布版本需要新的有授權資料或 adapter 訓練，並讓同一 prompt 同時通過內部集與
新的跨來源 holdout；不再使用已看過的 AI4I holdout 調參。

## 常見追問的短答

| 問題 | 回答重點 |
|---|---|
| 為何不用 accuracy？ | 類別均衡與安全風險不同；macro-F1 加上高風險召回、unknown、evidence、注入等不可互相抵銷的 gate。 |
| 為何 NF4 比 Q8 慢？ | NF4 走 Transformers adapter 路徑，Q8 走 llama.cpp；跨引擎速度可描述，但品質 relative gate 只在相容契約下比較。 |
| holdout 100% 為何不發？ | 是政策後結果，且有 16.33% 介入；更重要的是內部回歸失敗，顯示資料來源專用化。 |
| 為何保留 Q4？ | 保留 blocked 證據能證明 gate 真有否決權，也建立後續量化改進的可比較基線。 |
| 能否直接上產線？ | 不能；雖有 bytes／queue／drain 邊界與 shared bearer key，仍缺真實去識別驗證集、領域專家簽核、企業 IAM、TLS、跨 replica rate limit、外部不可變稽核與實際叢集壓測。 |

## 展示禁區

- 不把 fixture 回答、合成資料、估算或 manifest 靜態測試稱為真實模型／叢集實績。
- 不只報政策後 100%，必須同時說 raw evidence 與介入率。
- 不把 row-disjoint AI4I holdout 稱為獨立工廠驗證。
- 不展示或提交 Git 忽略的權重、逐案事件文字、原始報告或可能含本機資訊的檔案。
