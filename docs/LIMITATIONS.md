# Limitations

- 資料為原創合成與 14 筆人工 challenge，不能代表任何真實工廠分布。
- NF4 v5 雖通過完整 gate，event macro-F1 87.53% 距門檻不遠；Q8 severity macro-F1 與 NF4 的
  描述性差距為 10.15 個百分點，Q4 event macro-F1 則未過門檻。
- Transformers adapter 是非串流生成；OpenAI-compatible endpoint 在驗證後分塊回傳，TTFT 不代表
  token-level streaming。
- 本機 8 GiB GPU 適合 1.7B QLoRA 與 GGUF；vLLM、SGLang、TensorRT-LLM 需要 Linux/WSL 或更大
  顯存，本專案沒有偽裝成本機已驗證。
- Docker 與 Kubernetes manifest 因本機沒有 Docker/WSL/Kubernetes，只完成靜態語法與設計審查。
- 沒有外部身分驗證、TLS、持久化 observability backend 或 production autoscaling 實測。
- 成本資訊只可作硬體需求估算，不會用未執行估算冒充效能數字。
- llama.cpp 本機服務只綁定 loopback，未配置 API key；不可直接暴露至網路。
