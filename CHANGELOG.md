# Changelog

本專案依 [Semantic Versioning](https://semver.org/) 標示公開版本。

## 0.1.1 - 2026-09-09

- 修正 Linux CI 對 Windows 檔案鎖定介面的跨平台型別解析。
- 更新 GitHub Actions 至 Node 24 相容版本。

## 0.1.0 - 2026-09-09

第一個公開 MVP 版本：

- 完成繁體中文設備事件資料、QLoRA 訓練、模型合併與 GGUF 量化流程。
- 提供 Transformers、llama.cpp、FastAPI、OpenAI-compatible API 與本機 Web 展示頁。
- 保存 NF4、Q8、Q4 的完整品質證據、量化取捨與 released／previous-good／blocked 決策。
- 新增可重現 benchmark 1.4 工作負載指紋，以及 8192-context Q8／Q4 本機壓測摘要。
- 完成 bearer 認證設定、請求邊界、併發 queue、readiness、關閉排空與隱私檢查。
- 提供 Docker／Kubernetes 靜態部署基線、307 項通過測試與 MVP 驗收紀錄。

模型權重、生成資料與逐筆原始報告不包含在 Git repository；公開數字由去識別證據摘要保留。
