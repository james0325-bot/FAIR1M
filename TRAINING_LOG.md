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
