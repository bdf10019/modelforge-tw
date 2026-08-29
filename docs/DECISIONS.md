# Decisions

## ADR-001：使用 Qwen3-1.7B

Apache-2.0、繁體中文能力、可在 8 GiB GPU 做 NF4 QLoRA，並受 Transformers 與 llama.cpp 支援。
0.6B 僅作低資源備選，4B／8B 不在本機已驗證範圍。

## ADR-002：結構化事件分流，不做聊天或 RAG

作品要補足模型適配、量化、評測、serving 與發布治理能力。固定 schema 可建立可重現 gate，並避免
與一般知識檢索作品重複。

## ADR-003：原創合成資料 + 人工 challenge

此方案避免授權與私人資料風險，能保存 generator version、seed、template family 與跨 split 洩漏
檢查；代價是外部效度有限，因此不把分數外推為真實工廠表現。

## ADR-004：安全政策置於模型之外

模型不能單獨承擔高危複核、evidence grounding 或 prompt injection。政策只向上提高風險，證據
必須逐字存在輸入，解析失敗則 fail closed。安全政策後的 structured output 與 raw 模型格式分開
評分，避免 fallback 掩蓋失敗。

## ADR-005：Windows 驗證 Transformers + llama.cpp

本機真實執行 CUDA Transformers 與 official llama.cpp Windows CUDA build。vLLM、SGLang、ONNX
Runtime、TensorRT-LLM 不適合目前環境，因此不提供虛構效能。llama.cpp 對中文 argv 路徑有相容性
問題，腳本使用同磁碟 ASCII hard link，完成後自動清理。

## ADR-006：完整 gate 優先於小型 challenge

v1 在 14 筆 challenge 通過，卻在 224 筆揭露資料標籤錯誤；v3／v4 也揭露類別校準問題。每次都
保留報告、建立新資料 manifest 與新 artifact，不降低 gate。生成器 1.3 採 severity-conditioned
平衡情境，v5 才通過完整門檻。

## ADR-007：結構化輸出使用嚴格 JSON Schema

`json_object` 只鼓勵 JSON，不能保證所有欄位與 enum。最終 OpenAI-compatible adapter 發送 strict
JSON Schema：八個欄位全部 required，event、severity、safe action 與 reason code 均由允許清單
約束。reason-code enum 只加在新版出站 schema，不改歷史報告的儲存型別，以維持向後相容。

任何輸出契約變更都必須提升快取指紋版本，不能沿用舊 raw result。bounded repair 只供線上韌性，
正式 raw schema gate 仍以未修補文字計分。

## ADR-008：NF4 保持 released，Q8 作低延遲 previous-good

Q8 通過所有絕對門檻，mean latency 從最新 NF4 policy-v2 的 8.333 秒降至 1.060 秒；
event／severity macro-F1 與 NF4 的描述性差距為 1.08／10.01 個百分點。兩者生成契約不同，因此
不用這項差距作正式量化 gate；受控發布流程最終將完成 attestation 的 NF4 policy-v2 設為
released，Q8 policy-v2 保留為 previous-good。

Q4 雖只有 1.11 GB 且 mean latency 0.825 秒，event macro-F1 相對同契約 Q8 退化 10.73 個百分點並
低於絕對門檻，因此 blocked。速度、檔案大小或安全硬門檻不能抵銷主任務品質失敗。
