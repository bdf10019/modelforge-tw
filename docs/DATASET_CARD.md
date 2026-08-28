# Dataset Card

## 摘要

資料是本專案原創的繁體中文設備事件 fixture，不含真實公司、客戶、設備序號或個人資料。
生成器固定 seed 42 與版本 `1.3.0`；每次輸出 manifest、檔案大小與 SHA-256。

| split | 筆數 | 用途 |
|---|---:|---|
| train | 840 | SFT／QLoRA |
| validation | 140 | checkpoint 選擇 |
| test | 210 | 封存自動評估 |
| challenge | 14 | 人工撰寫的分布外與注入案例 |
| 合計 | 1,204 | — |

七類事件為 mechanical、electrical、thermal、pressure、communication、process、unknown。生成資料在
各 split 內類別平衡；challenge 覆蓋七類、critical/high/medium/low/unknown 與 prompt injection。

## 洩漏與品質控制

- train/validation/test 使用不同核心症狀、文字模板、scenario family ID 與 template ID。
- audit 拒絕重複 record ID、跨 split 正規化文字、跨 split scenario family ID 與跨 split template ID。
- 每個 `evidence_spans` 必須逐字存在於 `incident_text`。
- severity 標籤必須符合確定性風險下限：火花、焦味、大量洩漏等 critical 訊號，以及持續惡化、
  超限等 high 訊號，不得因「單次／偶爾」前綴被標成 low；audit 會拒絕衝突資料。
- `data/source/challenge.jsonl` 是唯一手工 source；`data/generated` 可完整重建且不進 Git。

版本 1.0 曾只用頻率前綴決定 severity，造成少數「偶爾大量洩漏」等不安全標籤；224 筆正式驗收
揭露此問題後，該模型候選已封鎖，1.1 將安全語意納入資料契約。後續稽核又發現 1.1 雖隔離外層
模板，核心症狀仍跨切分重用，因此中止 v2 訓練；1.2 改用各切分獨立症狀，並以症狀雜湊建立
scenario family 硬性稽核。這段歷史保留用來證明 gate 能阻止小型 challenge 未發現的資料設計錯誤。

1.2 將風險語意向上校正後，train 分布偏成 high 330、medium 131、low 138。1.3 改以嚴重度條件式
情境生成，train 的 low／medium／high 各 220 筆、critical 60 筆、unknown 120 筆；每個切分使用
不同核心症狀，並保留風險下限稽核。這能改善校準，但不等同真實場域的嚴重度先驗分布。

## 授權與限制

原創資料以 CC BY 4.0 發布。它是工程展示資料，不代表特定工廠分布、故障率或維修 SOP；不得用來
訓練自動控制系統。正式落地需要由領域專家建立真實、去識別且有資料權利的驗證集。
