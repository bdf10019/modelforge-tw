# Third-party notices

ModelForge-TW 的程式碼使用 MIT License；原創合成資料使用 CC BY 4.0。下列第三方項目不隨本
repository 散布權重或大型 binary，實際使用時仍應以鎖定版本附帶的 license 為準。

| 項目 | 用途 | 授權／來源 |
|---|---|---|
| Qwen3-1.7B | 候選 base model | Apache-2.0；<https://huggingface.co/Qwen/Qwen3-1.7B> |
| AI4I 2020 Predictive Maintenance Dataset | 第三方合成外部校準與 row-disjoint holdout | CC BY 4.0；DOI `10.24432/C5HS5C`；<https://archive.ics.uci.edu/dataset/601/ai4i> |
| PyTorch | 訓練與 Transformers 推論 | BSD-style；<https://pytorch.org/> |
| Hugging Face Transformers | 模型載入與生成 | Apache-2.0 |
| PEFT | LoRA／QLoRA | Apache-2.0 |
| Datasets | JSONL 資料流程 | Apache-2.0 |
| bitsandbytes | 4-bit training／inference | MIT |
| llama.cpp | GGUF 轉換、量化與推論 | MIT；release b10665，commit `ca3d5a3e10d53f7ea672cb9b6178faca3e2807bc`；<https://github.com/ggml-org/llama.cpp> |
| FastAPI | REST API | MIT |
| Uvicorn | ASGI server | BSD-3-Clause |
| Prometheus Python client | metrics | Apache-2.0 |

模型、CUDA runtime 與 llama.cpp binary 下載到 Git 忽略的本機目錄。產出 manifest 只保存來源、
revision、license、檔案大小與 SHA-256，不把權重提交到 repository。
`scripts/prepare-llama.ps1` 會在解壓與執行前核對兩個 release ZIP 的 SHA-256，並驗證 source commit。
