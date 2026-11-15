"""後端：收到圖片 -> 用訓練好的 YOLO-OBB 模型辨識 -> 回傳 JSON。

啟動：.venv/bin/python web/server.py   然後開 http://127.0.0.1:8000
換模型：FAIR1M_WEIGHTS=runs/s640/weights/best.pt .venv/bin/python web/server.py
"""
import io
import os
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = os.environ.get("FAIR1M_WEIGHTS", ROOT / "runs/n640/weights/best.pt")
MAX_BYTES = 25 * 1024 * 1024  # 上傳上限 25 MB，避免有人傳超大檔案

torch.set_num_threads(2)  # 只用 2 個 CPU 執行緒，不去搶正在跑的訓練
model = YOLO(str(WEIGHTS))  # 啟動時載入一次；找不到檔案會直接報錯
app = FastAPI()


@app.post("/api/detect")
async def detect(image: UploadFile = File(...), conf: float = Form(0.25)):
    """前端把圖片和信心門檻 POST 過來，這裡回傳偵測到的旋轉框。"""
    data = await image.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "圖片超過 25 MB")
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")  # RGBA 等格式一律轉成 RGB
    except (OSError, Image.DecompressionBombError):
        raise HTTPException(415, "這不是可讀取的圖片")

    # 用 CPU 辨識（訓練正在用 GPU）；imgsz=640 與訓練時一致
    result = model.predict(img, conf=min(max(conf, 0.01), 0.99), imgsz=640, device="cpu", verbose=False)[0]

    detections = []
    if result.obb is not None:
        # xyxyxyxy 是每個物件旋轉框的 4 個角點 [[x,y]*4]，座標是原圖的像素位置
        for pts, c, k in zip(result.obb.xyxyxyxy.tolist(), result.obb.conf.tolist(), result.obb.cls.tolist()):
            detections.append({"name": model.names[int(k)], "conf": round(c, 3), "points": pts})
    return {"width": img.width, "height": img.height, "detections": detections}


# 把 web/static/ 當成網站首頁（index.html、style.css、app.js）。要放在 API 路由之後
app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True))

if __name__ == "__main__":
    # 只綁 127.0.0.1：只有這台電腦連得到。瀏覽器只允許在 localhost 使用相機
    uvicorn.run(app, host="127.0.0.1", port=8000)
