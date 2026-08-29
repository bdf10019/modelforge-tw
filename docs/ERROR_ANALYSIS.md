# Error Analysis

本分析來自 Git 忽略的本機實測逐案報告。公開文件只保留原創合成案例的統計與可採取行動的結論。

## 從 Base 到 v5

- Base + minimal：14/14 raw schema 失敗，13/14 event 與 severity 錯誤，注入案例也失敗。
- Base + optimized-v1：event 錯誤降為 3/14，注入抵抗達 100%，但 13/14 仍以 Markdown code fence
  包住 JSON。bounded repair 可以解析，正式 raw gate 仍計失敗。
- QLoRA v1 challenge：14/14 JSON、event 與安全指標通過，只剩兩筆 medium→low；擴到完整集後卻
  因不安全的合成 severity 標籤而失敗。

這證明 prompt 能改善語意，但權重適配才使機器契約穩定；也證明 14 筆開發 challenge 不能取代
完整集。

## 資料版本失敗紀錄

1. 生成器 1.0 先依「單次／偶爾／持續」指定 severity，再隨機搭配症狀，產生「偶爾大量洩漏」
   等不安全標籤。v1 完整 severity macro-F1 僅 72.6%，registry 立即 blocked。
2. 生成器 1.1 修正風險下限；v2 在 checkpoint-10 發現核心情境跨切分重用，主動中止。
3. 生成器 1.2 隔離 scenario family，但風險校正使 high 類別過多；v3／v4 challenge 校準失敗。
4. 生成器 1.3 改為 severity-conditioned 情境，train low／medium／high 各 220，另有 critical 60、
   unknown 120。1,204 筆資料的 exact duplicate、跨切分文字、模板、情境家族、證據與政策衝突
   稽核皆為 0。

處置始終是修正資料與重新訓練，不降低 85% 門檻、不移除安全政策，也不刪除失敗 artifact。

## QLoRA v5 完整集

v5 有 28 筆 event mismatch、12 筆 severity mismatch、1 筆 raw schema invalid。event 主要混淆：

- thermal→pressure：6 筆。
- electrical→thermal：4 筆。
- communication→pressure：3 筆。
- electrical→communication：3 筆。
- pressure→process 與 process→pressure：各 3 筆。

severity 主要為 medium→low 9 筆，以及保守 fallback 導致的 low→unknown 3 筆。即使總體 gate
通過，event macro-F1 87.53% 只高於 85% 門檻 2.53 個百分點，因此不能宣稱接近完美。

唯一 raw invalid 出現在提示注入案例。早期 evaluator 錯把 raw 解析率同時當成 required-field rate，
且 HTTP／Transformers fallback 未重新套用注入政策。修正後，raw invalid 仍誠實計 0，但實際
structured fallback 會維持 required fields、人工複核、注入 reason code 與安全動作；新增回歸測試
避免兩個指標再次混淆。

## Merge 與結構化輸出

merged BF16 challenge 的 raw schema 只有 85.71%，顯示 merge 後行為不能假定與 PEFT adapter 完全
等價。第一輪 Q8 也有兩筆 JSON 損壞。改用嚴格 JSON Schema 後 challenge 達 100%，但完整集有
5 筆 unknown case 在自由 reason code 欄位生成未閉合字串至 token 上限。

最終將所有欄位設為 required，並在新版出站 schema 限制 reason code 白名單。這是解碼前的有限
語法，不是事後補括號；Q8 full raw schema 從 97.77% 提升為 100%。歷史 E3 run 的 unknown accuracy
為 100%，最新 policy-v2 E5 run 為 96.77%；兩者都保留 run ID，不能挑選較高的一輪冒充最新證據。
限制只存在新版出站契約，歷史報告仍可讀取。

## Q8 與 Q4

最新 Q8 完整集通過，但有 33 筆 event mismatch、37 筆 severity mismatch；severity macro-F1 與
NF4 的描述性差距為 10.15 個百分點。兩個引擎的生成契約不同，這不是正式相對 gate；仍足以支持
較保守的發布選擇，因此 Q8 作低延遲 previous-good，不取代 NF4 released 版本。

Q4 完整集有 57 筆 event mismatch、21 筆 severity mismatch；event macro-F1 74.43%、unknown
accuracy 83.87%。相對 Q8 的 event macro-F1 退化 10.73 個百分點，超過允許的 1.5 個百分點，
因此即使格式與所有安全硬門檻為 100%，仍必須 blocked。

## 下一步

1. 增加未參與迭代的外部 severity calibration set，特別補強 medium／low 邊界。
2. 增加 thermal／pressure、electrical／thermal 與 process／pressure 的困難負例。
3. 對 Q8 做量化感知的校準或少量蒸餾，再以相同 224 筆與獨立資料重驗。
4. Q4 只有在重新訓練或改變量化方法後才建立新 manifest；不得覆寫本次 blocked 證據。

本結果不能代表真實工廠分布、設備安全認證或自動控制能力。模型仍是人工分流輔助。
