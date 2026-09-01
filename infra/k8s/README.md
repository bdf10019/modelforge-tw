# Kubernetes baseline

這些 manifests 是可審查的部署基線，未宣稱已在本機叢集執行。本機沒有 Docker、Kubernetes 或
NVIDIA device plugin；目前只完成多文件 YAML 解析與設定一致性檢查。

## 前置條件

- Linux NVIDIA 節點、相容驅動與 NVIDIA device plugin。
- 名為 `modelforge-models` 的 ReadOnlyMany／ReadWriteMany PVC，根目錄包含
  `modelforge-tw-q8_0.gguf`。
- 若執行訓練 Job，另需名為 `modelforge-artifacts` 的可寫 PVC。
- 內部 registry 的 `modelforge-tw:0.1.0` 與 `modelforge-tw-gpu:0.1.0` images。
- 正式部署前把 llama.cpp 的 mutable tag 換成經弱點掃描且鎖定 digest 的 image。

## 拓樸

`llama-server` Deployment 取得一張 GPU，提供叢集內 OpenAI-compatible API；`modelforge-api` 不
直接掛載權重，負責 schema、安全政策、request ID、timeout、併發與 metrics。API 的 `/readyz`
會連帶檢查 llama.cpp `/health`，因此模型尚未載入或服務中斷時不會接收流量。

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
read-only root filesystem、capability drop、GPU request/limit 與 training fail-closed 設定；這些是
靜態部署契約，不等同 image build、admission、排程、GPU device plugin 或線上流量實測。
