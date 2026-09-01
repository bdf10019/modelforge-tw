# Limitations

- 資料為原創合成與 14 筆人工 challenge，不能代表任何真實工廠分布。
- 新增的 UCI AI4I calibration／holdout 仍是第三方合成資料；只覆蓋四個已映射事件類別。v5 雖在
  row-disjoint holdout 得到 100% 系統結果，raw evidence precision 只有 84.85%、16.33% 案例由
  安全層介入，且內部 event F1 退化到 69.72%，所以 v5 不發布。
- 現行 v2 的新版 Q8 內部 gate 通過，但外部 calibration event F1 只有 40.82%、raw schema 85.71%；
  registry 的 released／previous-good 狀態不代表已解決跨來源泛化。
- NF4 v5 雖通過完整 gate，event macro-F1 86.24% 距門檻不遠；Q8 severity macro-F1 與 NF4 的
  描述性差距為 10.01 個百分點，Q4 event macro-F1 則未過門檻。
- NF4、Q8、Q4 最新報告均已鎖定 `risk-precedence-v2` 與 `release-metrics-v2`；但 NF4 與
  llama.cpp 使用不同的 structured-output contract，因此 NF4→Q8 仍只作跨引擎描述性比較。
- 新版診斷報告使用 `risk-precedence-v3`／`release-metrics-v4` 並新增介入統計，與 registry 的
  policy-v2 attestation 不相容，不能直接作正式 relative gate。
- Transformers adapter 是非串流生成；OpenAI-compatible endpoint 在驗證後分塊回傳，TTFT 不代表
  token-level streaming。
- 本機 8 GiB GPU 適合 1.7B QLoRA 與 GGUF；vLLM、SGLang、TensorRT-LLM 需要 Linux/WSL 或更大
  顯存，本專案沒有偽裝成本機已驗證。
- Docker 與 Kubernetes manifest 因本機沒有 Docker/WSL/Kubernetes，只完成靜態語法與設計審查。
- 沒有外部身分驗證、TLS、持久化 observability backend 或 production autoscaling 實測。
- 成本資訊只可作硬體需求估算，不會用未執行估算冒充效能數字。
- llama.cpp 本機服務只綁定 loopback，未配置 API key；不可直接暴露至網路。4096 total context／
  4 slots 已實測不足以容納外部 prompt；目前預設 8192／4，但任何 prompt 或 parallel 變更都要重驗。
