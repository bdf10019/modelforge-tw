# 專案進度與下一步

更新：2026-09-09。範圍僅為「公司實務版」，不包含前沿研究版。

## 完成度口徑

依 2026-09-09 最新決定，後續只以**延續現有成果的本機求職 MVP**為目標；目前凍結範圍已
**100% 完成並通過驗收**。結果見 [MVP 驗收紀錄](MVP_ACCEPTANCE.md)，執行依據為
[MVP 收尾計畫](MVP_CLOSEOUT_PLAN.md)。原始建議完整版不再是結案門檻，
先前的 80–85% 完整版估計不再用於追蹤本次進度。

百分比是交付範圍的估計，不是測試通過率、模型準確率或正式上線就緒度。先前的 99% 過度混合了
「已有功能」與「已完成實測」；本次只有在展示與工程驗收完成後才標記 MVP 完成。

| 部分 | 已有成果 | 尚未完成或不能宣稱 |
|---|---|---|
| 資料 | 1,204 筆可重建原創合成資料、分組切分與洩漏稽核；第三方合成 calibration／holdout | 人工 challenge 只有 14 筆，非原訂 120–200；未完成近重複／n-gram 稽核 |
| 訓練與量化 | 真實 QLoRA v5、merged 權重、Q8／Q4 artifact 與 hash | 舊訓練報告為歷史 schema 1.0，不能事後補簽成新版訓練血緣 |
| 品質驗證 | NF4／Q8／Q4 各 224 筆完整報告；失敗 Q4 與外部專用 prompt 未發布 | Base／prompt-only 為早期 14 筆對照，尚非現行 v5 的 224 筆同契約比較；bootstrap 信賴區間待補 |
| 推論與展示 | Transformers、llama.cpp、API、即時展示頁、版本比較表與面試走查 | 展示頁不是同一輸入多模型並排執行；沒有企業 IAM 或線上 canary |
| 發布與工程 | 品質 gate、版本登錄、blocked、回滾、測試、封裝與隱私掃描 | 新版 registry／訓練契約有程式測試，歷史正式 registry 不等同已完成新版遷移 |
| 容器與叢集 | Docker／Kubernetes 設定與回歸測試 | 本機沒有 runtime，未完成 image build 或真實叢集部署 |

## 收尾結果

1. 已盤點並封存目前變更，保留新舊證據與正式 registry。
2. 已統一 README、效能文件、證據索引與展示說法。
3. 已完成 fixture、認證與 Q8 真實模型固定展示，並明確區分證據種類。
4. 已完成全套工程驗證：307 passed、2 個 Windows symlink 權限相關測試略過，其他項目通過。
5. 已留下驗收紀錄；本機 Git 提交完成後即以凍結 MVP 結案。

上一輪新設定 Q8／Q4 壓測已各完成 400/400 請求、零請求錯誤，公開封存於
[新設定壓測證據](current-context-benchmarks.json)。不因收尾再跑一次；這不取代品質 gate。
上一輪程式改動已納入最後整合驗收。

完整 A/B 比較、新訓練血緣實跑、容器／叢集驗收、多模型並排 UI、Ollama、canary、擴大
challenge 等均移至 MVP 後選配。它們仍未完成，但不再阻止本次 MVP 結案。

## 證據入口

- [README](../README.md)：成果與快速啟動。
- [實測索引](EVIDENCE_INDEX.md)、[效能報告](BENCHMARKS.md)：報告識別與執行條件。
- [外部校準](EXTERNAL_CALIBRATION.md)：跨來源失敗、holdout 與拒絕發布決策。
- [面試走查](INTERVIEW_WALKTHROUGH.md)、[限制](LIMITATIONS.md)：可展示與不可宣稱的邊界。

百分比不隨每個小修正機械增加；以缺口是否取得實測證據、可否重現驗收來更新。
