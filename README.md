# FAIR1M YOLO-OBB 訓練

用 Ultralytics YOLO 的旋轉框（OBB）模型，在 FAIR1M 衛星遙測資料集上訓練物件偵測。
目前階段只做**訓練與評估**，不含網頁或地圖。

## 環境

- 機器：Apple M2、8 GB RAM、使用 `device=mps`（沒有 CUDA）
- Python 3.11 虛擬環境在 `.venv/`；請用 `.venv/bin/python`，不要用系統的 `python3`
- 從零建立：
  ```bash
  python3.11 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  ```
- 部分 OBB 運算在 MPS 上可能退回 CPU，需要時設定 `PYTORCH_ENABLE_MPS_FALLBACK=1`

## 架構與接手（關掉 terminal、換模型都能續）

訓練由一個**脫離 terminal 的佇列執行器**負責，狀態全部存在檔案裡，不依賴任何 terminal 或 AI 工作階段。

```
configs/queue.txt ──► scripts/run_queue.sh ──► scripts/prepare_data.sh   下載(可續傳) → 驗證 1732/1732 → 轉換(一次)
 (要跑哪些訓練)        (佇列執行器，常駐)     └► scripts/train.py <名稱> …   可續跑的 YOLO-OBB 訓練封裝
                                                    │
                       runs/<名稱>/  weights/last.pt(每 epoch 存) best.pt · results.csv · final_metrics.json · DONE
                       runs/history.log (每次開始/結束一行)   runs/queue.log (執行器輸出)   runs/queue.pid
```

| 檔案 | 作用 |
|---|---|
| `configs/queue.txt` | 訓練清單，一行一個：`<名稱> --model … --imgsz … --epochs …`。執行器每次開工前重讀，可隨時增刪 |
| `scripts/start_queue.sh` | 以新 session + `caffeinate` 啟動執行器；重複執行是安全的（已在跑就不動作） |
| `scripts/run_queue.sh` | 依序跑清單；完成的放 `DONE` 標記並跳過；失敗 3 次就跳過；一次只允許一個執行器；佇列結束時自動產生 `runs/SUMMARY.md`（執行中請勿編輯此檔） |
| `scripts/watchdog.sh` | 每個訓練旁的看門狗：某個 epoch 明顯變慢（記憶體交換）就在存檔後重啟、從 `last.pt` 續跑（不算失敗）；到 `configs/deadline.txt` 的時間就在 epoch 邊界停止、驗證 best.pt 並結束佇列 |
| `scripts/train.py` | 有 `last.pt` 就**續跑**，已完成的就只重做最終驗證；結束後把每類別 AP 寫進 `final_metrics.json` |
| `scripts/prepare_data.sh` | 可重複執行：等待／續傳下載、驗證、轉換（有 `.complete` 標記就跳過） |
| `scripts/convert_fair1m.py` | XML → 5 大類 OBB 標註，RGBA→RGB，大圖切片，9:1 依原圖切分，並產生 `configs/fair1m5.yaml` |
| `scripts/summarize.py` | 彙整所有訓練結果，輸出 `runs/SUMMARY.md` |
| `scripts/status.sh` | 一屏狀態：執行器是否存活、剩餘磁碟、佇列進度、最近紀錄、目前訓練的最近 3 個 epoch |
| `TRAINING_LOG.md` | 每個階段的決策與結果（人與 AI 都先讀這份） |

**常用指令**
```bash
bash scripts/status.sh                 # 現在在做什麼
bash scripts/start_queue.sh            # 沒在跑時重新啟動（會自動續跑中斷的訓練）
kill $(cat runs/queue.pid)             # 停止執行器與目前訓練（last.pt 保留，重啟即續）
```

**各種中斷情況**
| 情況 | 結果 |
|---|---|
| 關閉 terminal / 結束 Claude 工作階段 | 執行器已脫離，**不受影響**，繼續跑 |
| 切換 AI 模型（Claude 的模型） | 不影響執行中的訓練；新的對話請先讀 `TRAINING_LOG.md` 與 `bash scripts/status.sh` |
| 電腦重開機、當機、斷電 | 執行器會停止，**不會自動重啟**；執行 `bash scripts/start_queue.sh`，最多損失 1 個 epoch |
| 想換 YOLO 模型 | 在 `configs/queue.txt` 加一行，例如 `s640 --model weights/yolov8s-obb.pt --imgsz 640 --epochs 40`（官方權重會自動下載） |
| 要提早結束某個訓練 | 停止執行器，將該行在 `queue.txt` 註解掉或在 `runs/<名稱>/` 建立 `DONE`，再啟動 |

**測試過的行為**（用小樣本）：訓練中被 `kill -9` 後重新執行會從下一個 epoch 接續；已完成的訓練不會重新訓練。

## 資料夾

| 路徑 | 用途 |
|---|---|
| `datasets/raw/` | FAIR1M 原始影像與 XML 標註（不進 git） |
| `datasets/processed/` | 轉成 Ultralytics OBB 格式的切片資料 |
| `datasets/samples/` | 小量子集，用來快速測試 |
| `configs/queue.txt` | 訓練佇列（進 git） |
| `configs/fair1m5.yaml` | 資料集 YAML，由轉換腳本產生，內含本機絕對路徑，**不進 git**（`prepare_data.sh` 會重建） |
| `scripts/` | 下載、轉換、訓練、佇列、狀態與彙整腳本 |
| `weights/` | 預訓練權重（`yolov8{n,s,m}-obb.pt`，首次使用時自動下載） |
| `runs/` | 訓練輸出、`history.log`、`SUMMARY.md` |

**不進 git 的東西**（見 `.gitignore`）：`datasets/`、`weights/`、`runs/`、`.venv/`、`.claude/`，
以及 `*.pt`、`*.tif`、`*.jpg`、`*.png`、`*.zip`、`*.log`、`*.pid` 等。訓練資料、模型權重與訓練輸出都不要提交；
如果之後需要放文件用的圖片，用 `git add -f` 個別加入。

## 從零重現

```bash
git clone https://github.com/james0325-bot/FAIR1M.git && cd FAIR1M
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
bash scripts/prepare_data.sh   # 下載約 9 GB、驗證、轉換（資料不在 git 內，需要這一步）
bash scripts/start_queue.sh    # 依 configs/queue.txt 開始訓練
bash scripts/status.sh
```

## 流程

### 1. 取得資料
FAIR1M 官方來源是 <https://www.gaofen-challenge.com/benchmark>（需申請）。
目前使用 Hugging Face 上的第三方鏡像 `blanchon/FAIR1M`，下載到 `datasets/raw/fair1m_hf/`：
1,732 張 TIF 影像加上 1,732 個 XML 標註，約 9.2 GB。這只是官方 15,000 張訓練集中公開的一部分。
實測前 300 張：尺寸以 1000×1000（約 70%）與 1500×1500（約 28%）為主，少數到 3000 或 5000 px
（鏡像 README 寫的「1024×1024」不正確）；約一半是 RGBA 四通道，轉換時要轉成 RGB。

授權：鏡像標示為 `unknown`；原資料集據稱僅限學術與非商業用途，使用前請自行確認。

磁碟空間有限（約 27 GB 可用），轉換與切片後的資料要再估算空間。

### 2. 轉換標註
FAIR1M 的標註是每張影像一個 XML，內含 4 個角點的像素座標與細分類名稱。
Ultralytics OBB 的格式是每張影像一個 `.txt`，每行：

```
class_id x1 y1 x2 y2 x3 y3 x4 y4     # 座標正規化到 0–1
```

XML 的每個物件有 5 個點，第 5 點等於第 1 點（封閉多邊形），轉換時只取前 4 點。

`scripts/convert_fair1m.py` 負責轉換，輸出到 `datasets/processed/fair1m5/`（JPEG，約 1 GB），
並產生 `configs/fair1m5.yaml`（類別名稱與資料路徑）。

#### 類別：只用 5 大類
訓練只用 5 大類，37 個細分類在轉換時合併：

| id | 大類 | 包含的細分類 |
|---|---|---|
| 0 | ship 船 | Passenger Ship、Motorboat、Fishing Boat、Tugboat、Engineering Ship、Liquid Cargo Ship、Dry Cargo Ship、Warship、other-ship |
| 1 | vehicle 車輛 | Small Car、Van、Bus、Cargo Truck、Dump Truck、Trailer、Tractor、Excavator、Truck Tractor、other-vehicle |
| 2 | airplane 飛機 | Boeing737、Boeing747、Boeing777、Boeing787、ARJ21、C919、A220、A321、A330、A350、other-airplane |
| 3 | court 場地 | Baseball Field、Basketball Court、Football Field、Tennis Court |
| 4 | road 道路 | Roundabout、Intersection、Bridge |

注意：前 60 個樣本中車輛佔絕大多數（Small Car、Van 約 1,400 個，其他類別各數十個），類別嚴重不平衡，
評估時要看每個類別的 mAP，不要只看總平均。

### 3. 切片
影像尺寸不一（1000～5000 px），訓練時 Ultralytics 會縮放到 `imgsz`；縮小後小車輛會變得很小。
切片已整合在 `scripts/convert_fair1m.py`：長邊超過 1536 px 的大圖切成 1024×1024 切片、重疊 200 px，
並同步裁切標註；1000～1500 px 的影像不切，直接使用。
- 訓練集與驗證集依**原始影像**切分，避免同一張圖的切片同時出現在兩邊（資料洩漏）。
- 只保留旋轉框中心落在切片內、且大部分面積在切片內的物件。

### 4. 訓練
一律透過佇列執行（見上方「架構與接手」）：在 `configs/queue.txt` 加一行，再 `bash scripts/start_queue.sh`。
也可以直接跑單一訓練（同樣可續跑）：

```bash
.venv/bin/python scripts/train.py n640 --model weights/yolov8n-obb.pt --imgsz 640 --epochs 100 --patience 30
```

預設 `batch=4`、`device=mps`。8 GB 記憶體下，`batch` 和 `imgsz` 是最先要調的參數；出現記憶體不足就降低它們。

### 5. 評估
```bash
yolo obb val model=runs/n640/weights/best.pt data=configs/fair1m5.yaml device=mps
```
看 mAP50 與 mAP50-95，以及每個類別的表現。訓練結束時 `train.py` 會自動把每類別 AP 寫進 `runs/<名稱>/final_metrics.json`，`scripts/summarize.py` 彙整成 `runs/SUMMARY.md`。
FAIR1M 官方測試集標註不公開，所以用自己留出的驗證集評估。

## 網站：上傳照片或用相機辨識

`web/` 是一個給初學者讀的小網站，前後端分離：

| 檔案 | 角色 |
|---|---|
| `web/server.py` | 後端（FastAPI）：收圖片 → 模型辨識 → 回傳 JSON；同時把 `web/static/` 當網頁提供 |
| `web/static/index.html` | 頁面結構 |
| `web/static/style.css` | 外觀 |
| `web/static/app.js` | 前端邏輯：上傳／相機 → 呼叫 `/api/detect` → 畫旋轉框 |

```bash
.venv/bin/pip install -r requirements-web.txt     # 只需裝一次
.venv/bin/python web/server.py                    # 開 http://127.0.0.1:8000
FAIR1M_WEIGHTS=runs/s640/weights/best.pt .venv/bin/python web/server.py   # 換模型
```

- 預設用 `runs/n640/weights/best.pt`，以 CPU 推論（訓練在用 GPU），啟動時載入一次；訓練出新權重後要重啟伺服器。
- 只綁 `127.0.0.1`。瀏覽器只允許在 localhost（或 https）使用相機，不要直接把它開放到區網或網路上。
- 模型用衛星／空拍影像訓練，一般地面照片幾乎不會有正確結果。

## 目前進度
- [x] 環境（torch、ultralytics、MPS）
- [x] 預訓練權重 `weights/yolov8n-obb.pt`
- [x] 下載 FAIR1M 子集（Hugging Face 鏡像）
- [x] 標註轉換與切片腳本（`convert_fair1m.py`）
- [x] 可續跑的訓練封裝與佇列執行器，冒煙測試通過
- [x] 速度基準（640px 全資料約 2.1 分鐘/epoch；1024px 估約 7 分鐘/epoch）
- [ ] 正式訓練（進行中，細節與每次結果見 `TRAINING_LOG.md`）：n640 → n1024 → s640 → s1024 → m640
- [ ] 最終評估與比較

## 已知限制
- 小模型（yolov8n）對小物件（車輛）效果有限，先預期飛機與船表現較好。
- 只使用 5 大類（見上表）；車輛數量遠多於其他類別，類別不平衡嚴重。
- 本機速度有限，長時間訓練可考慮改到雲端 GPU，流程與腳本不變。
