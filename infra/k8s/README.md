# Kubernetes baseline

這些 manifests 是可審查的部署基線，未宣稱已在本機叢集執行。本機沒有 Docker、Kubernetes 或
NVIDIA device plugin；目前只完成多文件 YAML 解析與設定一致性檢查。

## 前置條件

- Linux NVIDIA 節點、相容驅動與 NVIDIA device plugin。
- 名為 `modelforge-models` 的 ReadOnlyMany／ReadWriteMany PVC，根目錄包含
  `modelforge-tw-q8_0.gguf`，且檔案必須可由 UID 10001 唯讀存取。
- 若執行訓練 Job，另需名為 `modelforge-artifacts`、可由 UID 10001 寫入的 PVC。
- 內部 registry 的 `modelforge-tw:0.1.0` 與 `modelforge-tw-gpu:0.1.0` images。
- 正式部署前把 llama.cpp 的 mutable tag 換成經弱點掃描且鎖定 digest 的 image。
- 套用前必須建立 `modelforge-api-auth` Secret，key 名稱為 `api-key`、值至少 32 字元；manifest
  將它注入 `MODELFORGE_API_KEY`，Secret 遺漏時 Pod 不啟動。實際值應由 secret manager 管理，
  不可寫入 repository 或指令歷史；無認證模式只保留給 loopback／Compose 本機演示。

## 拓樸

`llama-server` Deployment 取得一張 GPU，提供叢集內 OpenAI-compatible API；`modelforge-api` 不
直接掛載權重，負責 schema、安全政策、request ID、timeout、併發與 metrics。API 的 `/readyz`
會連帶檢查 llama.cpp `/health`，因此模型尚未載入或服務中斷時不會接收流量。
設定 Secret 時，兩個 `/v1` 推論路由在解析 JSON 前要求 bearer key；health、readiness 與 config
保留給叢集探針。這個 shared key 是最低部署基線，不等同企業 IAM、每使用者授權或 key rotation。
API baseline 每 replica 最多同時送出 4 筆推論、等待區 16 筆、queue timeout 5 秒；request body 與
模型回應分別限制 32 KiB／256 KiB。Readiness 內部上限 4 秒、probe timeout 5 秒；關閉時應用最多
排空 65 秒，Pod 終止期限為 75 秒，讓 60 秒
下游 timeout 有完成空間；超期同步工作最終仍須由容器平台終止。
所有線上 Pod 與訓練 Job 都明確以 UID 10001 強制 non-root、關閉不需要的 service-account token
自動掛載、使用 RuntimeDefault seccomp、禁止提權並移除 Linux capabilities。Training container
另使用唯讀根檔案系統，只將
artifact PVC、重建的 generated data、模型 cache 與 `/tmp` 掛為可寫目錄；llama.cpp 則有最長
300 秒 startup probe，模型載入期間不會先被 liveness probe 誤判重啟。

範例沒有把 Git 忽略的 `outputs/model-registry.json` 打進 image 或掛入 Pod，因此刻意不設定
`MODELFORGE_REGISTRY_DEPLOYMENT_ROLE`，不能宣稱已做線上 registry binding。正式環境若要啟用，
應從受控來源唯讀掛載具完整新版 training／evaluation attestation 的 registry，再選 `released` 或
`previous-good`。若同時掛載報告與模型 artifact 根目錄，設定 `MODELFORGE_REGISTRY_ARTIFACT_ROOT`
可在 Pod 啟動時重算全部檔案；這驗證 API Pod 的掛載副本，不代表遠端 llama.cpp 記憶體 attestation。
現有 schema 1.0 實測歷史只供相容讀取，缺欄位時服務會拒絕綁定。

## 套用前檢查

```powershell
kubectl apply --dry-run=server -f infra/k8s/app.yaml
kubectl apply --dry-run=server -f infra/k8s/training-job.yaml
```

`dry-run=server` 仍需要目標叢集；本作品機未執行這兩個命令。資源 requests／limits 是 1.7B Q8
起始值。baseline 使用 8192 total context／4 parallel slots，API 宣告每 slot 2048；這是外部校準
發現 1024-token slot 截斷後的修正。正式環境仍必須依實際節點、prompt 長度、context size、
parallel slots 與壓測結果調整。

沒有容器／叢集 runtime 時，repository 測試只驗證 YAML、image／port／probe 對應、非 root、
read-only root filesystem、capability drop、seccomp、service-account token、GPU request/limit 與
training fail-closed 設定；這些是
靜態部署契約，不等同 image build、admission、排程、GPU device plugin 或線上流量實測。
