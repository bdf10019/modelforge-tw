# ModelForge-TW MVP 驗收紀錄

驗收日期：2026-09-09

驗收結果：**通過；公司實務版本機求職 MVP 完成**

應用版本：`modelforge-tw 0.1.3`

收尾前基準提交：`9a938e4`

最終版本：包含本文件的本機 Git 提交，可用
`git log -1 --format=%H -- docs/MVP_ACCEPTANCE.md` 查詢。

## 驗收範圍

本次依 [MVP 收尾計畫](MVP_CLOSEOUT_PLAN.md) 驗收現有資料、模型產物、品質與效能證據、API、
展示頁、發布治理、測試及文件。未新增資料、重新訓練模型、安裝部署平台或變更正式 registry；
前沿研究版不在範圍內，也未修改。

## 固定展示驗收

### Fixture 路徑

- `/healthz` 回報 `ok`、`/readyz` 回報 `ready`、`/docs` 回傳 HTTP 200。
- 執行環境在 API 與畫面標示為 `fixture-adapted · deterministic-fixture · fixture`。
- 高風險案例：電氣異常／危急、停止並升級、需要人工複核。
- 資訊不足案例：事件類型與嚴重度待確認、補充證據、需要人工複核。
- 指令注入案例：機械異常／高、隔離並升級、需要人工複核。
- 實際瀏覽器畫面可顯示事件類別、嚴重度、evidence、安全下一步與人工複核狀態，未發現阻擋
  展示的排版或操作問題。

### 認證路徑

- 使用一次性本機測試 key 啟動 fixture；`/api/config` 正確標示需要認證，畫面顯示密碼欄位。
- 缺少 key 與錯誤 key 均回傳 HTTP 401；正確 key 回傳 HTTP 200。
- 測試 key 未寫入 repository、URL、結果 metadata 或本驗收紀錄，服務停止後即失效。

### Q8 真實模型路徑

- llama.cpp build `b10665-ca3d5a3e1`，Q8_0、8192 total context／4 slots，每 slot 2048。
- 統一 API 回報 `openai-http · modelforge-tw-q8_0 · measured`，readiness 通過。
- 三個固定案例都完成結構化推論；高風險、資訊不足與指令注入案例均要求人工複核，模型身份為
  `qwen3-1.7b-merged-v5-q8_0-policy-v2`。
- 實際瀏覽器完成高風險案例，顯示危急、停止並升級、輸入 evidence 與 measured runtime。
- API 與 llama.cpp 均正常停止；驗收結束時沒有殘留模型或 API 程序。

這三筆是啟動與展示 smoke，不取代 224 筆品質評測，也不新增發布主張。高風險 smoke 的
`event_type` 為 thermal，但確定性安全政策仍維持 critical、`stop_and_escalate` 與人工複核；
主任務品質仍以既有完整報告與 gate 為準。

## 工程驗收

最後整合驗證入口為：

```powershell
.\scripts\verify.ps1
```

驗證結果：

- dependency、資料重建與資料稽核：通過，1,204 筆、零稽核錯誤。
- 格式、lint 與型別檢查：通過；22 個 source modules 無型別錯誤。
- 測試：`307 passed, 2 skipped`。兩項略過都是 Windows 目前沒有建立 directory symlink 的權限；
  對應 Windows reparse-point 路徑仍有獨立測試，不是功能測試失敗。
- wheel 建置、內容核對、隔離安裝與匯入 smoke：通過。
- 隱私／秘密掃描：通過。
- 本機證據：外部報告 6、主要評測 3、Q8→Q4 比較、訓練、artifact 報告 2、歷史 benchmark 2、
  新 context benchmark 2、registry 與外部 manifest 全部核對通過。

## 證據與發布狀態

- 正式 registry 保持 `released=qwen3-1.7b-qlora-v5-policy-v2`。
- `previous_good=qwen3-1.7b-gguf-q8-v5-policy-v2`。
- Q4 candidate 維持 `blocked`。
- 新 8192-context 壓測：Q8 與 Q4 各 400/400 請求完成、零請求錯誤；完整去識別報告與原始
  報告 SHA-256 見 [current-context-benchmarks.json](current-context-benchmarks.json)。
- 效能完成不代表品質 gate 通過；Q4 的 blocked 決策未變。

## 已知限制與例外

- 資料仍是原創合成加第三方合成資料，不代表真實工廠分布或安全認證。
- Base／prompt-only 是早期 14 筆 challenge 對照，不宣稱現行 v5 的完整因果比較。
- Docker／Kubernetes 僅有靜態設定與回歸測試，未宣稱實際叢集部署。
- 歷史訓練與 registry 保留原始 schema，不事後補簽新版血緣。

## 結論

固定 fixture 與 Q8 展示、認證、完整工程驗證、證據一致性、發布狀態與文件均已驗收。
目前沒有阻擋本機求職展示的已知問題；上述限制已揭露且屬於凍結 MVP 範圍外。
公司實務版以此版本完成收尾，後續需求另立版本評估，不在本次持續擴建。
