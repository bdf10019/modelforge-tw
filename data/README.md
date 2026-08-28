# Data boundary

`data/source/challenge.jsonl` 是人工撰寫的原創繁中 challenge fixture；`data/generated` 則由
`modelforge generate-data` 以固定 seed 產生。兩者均以 CC BY 4.0 釋出。

資料只描述虛構設備與通用異常，不來自公司、客戶、真實操作員、維修手冊或徵才頁。生成資料
保存 scenario family、template、asset family 與 generator version，並在寫檔前檢查：

- record ID 重複
- 正規化文字跨 split 重複
- template ID 跨 split 洩漏
- scenario family ID 跨 split 洩漏
- evidence span 無法逐字定位
- severity 標籤與確定性安全風險下限衝突
- 各 split 類別分布

生成檔不提交 Git；`manifest.json` 記錄每個 split 的筆數、bytes 與 SHA-256。正式評測報告還會
保存其實際輸入 records 的 combined SHA-256，避免資料更新後沿用舊分數。

## 產生

```powershell
modelforge generate-data --output data/generated
```

預設產生 1,190 筆 template-based records，加上 14 筆人工 challenge，共 1,204 筆。template-based test 與
challenge 必須分開報告，不能混成一個聲稱具外部效度的總分。

## 授權

資料採 [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)。
建議標示：`ModelForge-TW original synthetic industrial-event dataset, CC BY 4.0`。
