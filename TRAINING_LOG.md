# 訓練紀錄與五天計畫

開始：2026-11-15（使用者給予約五天，機器已設定不休眠、接電源）。
原則：一次只跑一個訓練（8 GB RAM、單一 MPS GPU）；每階段結束後看結果再決定下一步（滾動式修正）；
預留最後約 12 小時做評估與整理，不在最後一刻開新的長訓練。

## 計畫（會隨實測修正）
| 階段 | 內容 | 預估 |
|---|---|---|
| 0 | 下載、轉換、量速度（640 全資料 1 epoch；1024 取 10% 資料 1 epoch） | ~1 小時 |
| 1 | 基準線：yolov8n-obb、imgsz=640，依速度決定 epoch 數 | 待實測 |
| 2 | 依 1 的每類別 mAP 決定：提高 imgsz（小車輛）、或換較大模型（yolov8s-obb）、或調整類別權重 | 待定 |
| 3 | 最佳設定再訓練到收斂 | 待定 |
| 4 | 最終評估（每類別 mAP、混淆矩陣、錯誤案例）與 README 整理 | ~12 小時 |

## 接手須知（給之後的人或新的 AI 工作階段）
- 先執行 `bash scripts/status.sh`；架構與續跑方式見 `README.md` 的「架構與接手」。
- 訓練由脫離 terminal 的執行器在跑（`runs/queue.pid`），不需要任何對話工作階段才能繼續。
- 要新增訓練就編輯 `configs/queue.txt`；要看結果看 `runs/history.log` 與 `runs/<名稱>/final_metrics.json`。
- 已知陷阱：對已完成的 run 呼叫 `resume=True` 會讓 Ultralytics 默默用預設的 dota8 資料集重訓 100 epoch；
  `scripts/train.py` 已用「optimizer 是否被移除」判斷並避開，不要繞過它直接 `yolo train resume`。
- 資料：Hugging Face 鏡像 `blanchon/FAIR1M`（1,732 張，授權 unknown，原資料集據稱僅限學術非商業），
  放在 `datasets/raw/fair1m_hf/`（9 GB）；轉換後在 `datasets/processed/fair1m5/`。
- 磁碟是限制：剩餘空間會隨轉換與訓練下降；`status.sh` 會顯示。低於約 5 GB 時先刪 `runs/` 中不要的實驗。

## 紀錄（最新在下）
- 2026-11-15 19:30 下載 1,732 張影像與標註完成（約 9 GB，花 15 分鐘）。
- 2026-11-15 19:50 完成轉換腳本、可續跑的訓練封裝與佇列執行器；用小樣本測試通過（中斷後續跑、完成後不重訓）。
  發現並修掉一個 bug：對已完成的 run 續跑會誤啟動 dota8 訓練。
- 2026-11-15 19:52 以脫離 terminal 的方式啟動佇列執行器（pid 見 `runs/queue.pid`）：轉換資料 → bench640 → bench1024。
- 2026-11-15 19:56 階段 0 完成。資料：訓練約 1,514 張、驗證 218 張（依原圖切分，19 張驗證圖無物件）；
  轉換後 1.0 GB（JPEG），磁碟剩 16 GB。實測速度（yolov8n-obb、batch 4、MPS）：
  640px 全資料 1 個 epoch ≈ 2.1 分鐘；1024px 10% 資料 ≈ 43 秒（全資料估 ≈ 7 分鐘/epoch）。
  1 個 epoch 後 640px 的 mAP50=0.57（ship 0.37、vehicle 0.66、airplane 0.97、court 0.71、road 0.14）：
  預訓練權重（DOTA）已有相近知識；road 與 ship 最弱。部分影像有大量重複標註（Ultralytics 自動移除，如 1534.jpg 移除 130 個）。
  決策：階段 1 先跑兩個基準線 n640（100 epoch，約 3.5 小時）與 n1024（100 epoch，約 12 小時），
  比較哪個解析度對小車輛與 road 較有幫助，再決定階段 2（換 yolov8s 或調整類別權重）。
- 2026-11-15 20:15 使用者將離開到約 2026-11-20（週五）。為避免 GPU 空轉，預先把階段 2 排進佇列：
  s640、s1024、m640（yolov8s/m-obb 權重已預先下載到 weights/）。預估時間（由 n 的實測按 FLOPs 推估，未驗證）：
  n640 ≈ 3.7 h、n1024 ≈ 12 h、s640 ≈ 9 h、s1024 ≈ 28 h、m640 ≈ 33 h，合計約 86 小時，在約 4.5 天預算內。
  失敗（例如記憶體不足）3 次的工作會自動跳過。新增 `scripts/summarize.py`（輸出 `runs/SUMMARY.md`），
  並啟動一個脫離的收尾程序：佇列結束時自動產生 SUMMARY.md 並在 `runs/history.log` 寫入 `QUEUE FINISHED`。
  使用者回來後：先 `bash scripts/status.sh`，再看 `runs/SUMMARY.md`；若佇列已結束，依結果再決定是否延長訓練或調整類別權重。
- 2026-11-15 23:41 ～ 2026-11-16 17:42（依 `runs/history.log`、`runs/queue.log` 與 `configs/queue.txt` 註解補記）：
  n640 完成（3.7 h，100 epoch）：mAP50=0.799、mAP50-95=0.572。
  n1024（batch 4）連續 3 次 MPS 記憶體不足（`MPS command buffer execution failed: Insufficient Memory`），被自動跳過；
  佇列中的 s1024 改為 n1024b2（batch 2、workers 0）。s640 中途 OOM 1 次後續跑完成（實際約 9.4 h；
  前 12 個 epoch 約 970 s/epoch，疑似記憶體交換；續跑後穩定約 252 s/epoch）。
  n1024b2 OOM 2 次後續跑完成（約 8.3 h，≈ 298 s/epoch）。因 n640 在第 97 epoch 仍緩升，加入 n640e300（300 epoch、patience 100）。
  注意：`history.log` 的 END 時數只算最後一次啟動那段（例如 n1024b2 寫 1.24 h，實際 8.3 h）；
  `results.csv` 的 time 欄在每次續跑時歸零。
- 2026-11-16 22:39 結果比較（best.pt 最終驗證；mAP50-95 / 每類 AP50 ship、vehicle、airplane、court、road）：
  | run | mAP50 | mAP50-95 | ship | vehicle | airplane | court | road |
  |---|---|---|---|---|---|---|---|
  | n640 | 0.799 | 0.572 | 0.674 | 0.882 | 0.972 | 0.825 | 0.642 |
  | s640 | 0.824 | 0.619 | 0.735 | 0.914 | 0.972 | 0.820 | 0.677 |
  | n1024b2 | 0.835 | 0.630 | 0.809 | 0.948 | 0.984 | 0.803 | 0.629 |
  | n640e300（第 128 epoch 停） | 0.809 | 0.575 | 0.660 | 0.886 | 0.968 | 0.817 | 0.716 |
  觀察：1024px 對小物件幫助最大（ship +0.14、vehicle +0.07，相對 n640）；換 s 模型對 road 有幫助；
  n640 延長到 300 epoch 幾乎無益（第 110 epoch 後停滯在 0.577，與 n640 的 0.575 相同），但 road AP50 0.716 是目前最高，
  而 ship 略降，顯示多訓練主要是在類別間互換，非整體進步。
  決策（使用者同意）：提早停止 n640e300，改跑 s1024（yolov8s-obb、1024、batch 2、workers 0，結合 s 與 1024 的優勢），m640 留在其後。
  操作：先停收尾程序與執行器（避免被記為失敗或自動續跑），在 epoch 128 存檔後 48 秒停止 train.py（last.pt 驗證可載入）；
  手動對 best.pt 做最終驗證寫入 `runs/n640e300/final_metrics.json`；加註 `runs/n640e300/STOPPED`；
  queue.txt 中 n640e300 改為註解（取消註解即可從 last.pt 續跑）；22:40 重新啟動執行器與收尾程序。
  `scripts/summarize.py` 修正兩處：續跑過的 run 時數改為各段加總；模型欄改從 history.log 的 START 取得（原本續跑後會顯示 last.pt）；
  新增「提早停止」狀態。注意：SUMMARY.md 主表的 mAP 取訓練中最佳 epoch 的驗證值，每類別表為 best.pt 最終驗證，兩者可能差約 0.001～0.007。
  預估：s1024 ≈ 555 s/epoch → 約 15 h（若 OOM 或記憶體交換會更久；失敗 3 次會自動跳到 m640）；m640 約 19 h（未驗證）。
  兩者合計約 34 h，預計 11-18 上午前後佇列結束，在 11-20 前留有評估時間。
- 2026-11-16 22:50 s1024 第 1 個 epoch 正常完成，未發生 OOM：612 s/epoch（比預估的 555 s 慢約 10%），
  第 1 個 epoch 的 mAP50=0.732（同為第 1 個 epoch，bench640 為 0.57）。
  修正預估：s1024 100 epoch ≈ 17 h，約 11-17 16:00 完成；之後的 m640 約 19 h（未驗證），佇列約 11-18 中午前後結束。
