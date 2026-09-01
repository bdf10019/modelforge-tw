# 求職摘要與面試導覽

## 一句話

ModelForge-TW 是一套可在單張 8 GB GPU 重現的繁體中文設備事件模型適配與發布平台：從 Base、
prompt-only、QLoRA、合併權重到 GGUF 量化，以同一封存資料與安全 hard gate 比較品質、延遲、
吞吐、顯存、檔案大小與回滾能力。

## 履歷專案描述

- 設計 1,204 筆原創繁體中文設備事件資料，建立 group-aware split、資料 manifest、政策一致性與
  跨切分情境洩漏稽核；所有稽核項目為 0 違規。
- 在 RTX 5060 Laptop 8 GB 上完成 Qwen3-1.7B NF4 QLoRA：僅訓練 8,716,288 個參數
  （0.504%），1 epoch 需 1,467.5 秒，peak allocated VRAM 4.52 GB。
- 建立 224 筆正式發布 gate；QLoRA v5 policy-v2 達 event macro-F1 86.24%、severity macro-F1 95.35%，
  high/critical 複核召回、unknown、證據精度與注入抵抗皆為 100%。
- 導入 UCI AI4I 第三方合成校準與 98 筆 row-disjoint holdout，鎖定來源／轉換／抽樣 hash；辨識
  1024-token slot 截斷、資料轉換矛盾與跨來源泛化缺口，並拒絕「外部 100% 但內部 F1 退化」候選。
- 完成 llama.cpp Q8／Q4 實測：Q8 平均延遲 1.060 秒、四路 2.116 req/s 與 201.9 output tok/s；
  Q4 平均延遲 0.825 秒，但 event macro-F1 74.43%，由自動門檻封鎖。
- 實作 FastAPI、OpenAI-compatible API、真正 streaming TTFT、嚴格 JSON Schema、原文證據驗證、
  逾時 fallback、Prometheus、artifact SHA-256、registry promote 與 previous-good rollback。
- 建立 Windows、Docker、Compose、Kubernetes、測試、隱私掃描與本機 Git 流程；模型權重、生成
  資料與實測輸出均排除於 Git。

## 面試時先展示

完整的分鐘級流程與常見追問見 [`INTERVIEW_WALKTHROUGH.md`](INTERVIEW_WALKTHROUGH.md)。

1. 在 Web UI 選擇「指令注入」，說明輸入為何是不可信資料，以及服務端如何強制人工複核。
2. 開啟 `docs/EVALUATION.md`，用 Base→Prompt→QLoRA 拆分提示與權重適配的效果，再比較 NF4、
   Q8 與 Q4。
3. 展示 registry 事件：policy-v2 Q4 gate fail→blocked、Q8 pass→previous-good、NF4 policy-v2
   attestation→released。
4. 開啟 `docs/EXTERNAL_CALIBRATION.md`，展示如何區分模型原生 evidence、政策後結果與 holdout
   污染邊界，以及為何外部單一資料高分仍被內部回歸否決。

## 技術取捨

- 選 1.7B + NF4 QLoRA，確保訓練與驗收能在作品機 8 GB VRAM 真實完成。
- 不用總體 accuracy 取代安全指標；高風險漏接、證據幻覺或注入失敗不能被其他高分抵銷。
- 不把 fixture 或靜態部署設定包裝成模型／叢集實績；報告以 `evidence_kind` 和限制章節區分。
- 不因 Q4 檔案小、速度快就發布；完整品質與相對量化門檻都有否決權。
- 不讓模型控制設備；所有高風險、未知或無效結果一律轉人工。
- 不把安全層修正後的 100% 冒充模型原生 100%；報告另列 raw evidence 與介入率。

## 可追問的工程細節

- 為什麼 grouped split 比逐筆隨機切分更適合模板化合成資料。
- 為什麼 raw schema、required fields 與安全政策後的結果必須分開計分。
- 如何用 record hash、adapter hash、prompt 與解碼設定指紋安全續跑 224 筆評測。
- 為什麼 strict JSON Schema 還需要 reason-code 白名單，以及如何維持歷史報告相容性。
- TTFT 如何從 streaming 第一個內容 chunk 量測，而不以 0 或總延遲冒充。
- registry 如何驗證 artifact 路徑、SHA-256、release gate、previous-good 與 rollback 狀態。

## 誠實邊界

本作品沒有連接真實工廠、沒有設備控制權，也未宣稱具備生產環境的身分驗證、TLS、集中稽核與
事故應變。Docker／Kubernetes 是可部署基線，尚未在本機叢集執行；品質結果來自原創合成資料、
開發 challenge 與第三方合成 AI4I，不代表外部真實工廠分布。
