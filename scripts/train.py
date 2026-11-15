#!/usr/bin/env python3
"""Resumable Ultralytics YOLO-OBB training wrapper.

    .venv/bin/python scripts/train.py <run-name> [--model weights/yolov8n-obb.pt] [--imgsz 640]
                                       [--epochs 50] [--batch 4] [--set key=value ...]

- Output goes to runs/<run-name>/ (weights/last.pt is saved every epoch).
- If runs/<run-name>/weights/last.pt exists, training RESUMES from it (works after a crash,
  closed terminal or reboot). If that run already finished, it just re-runs the final validation.
- Switching model = pass a different --model (e.g. weights/yolov8s-obb.pt, or yolo11n-obb.pt;
  Ultralytics downloads official weights automatically if the file name is a known one).
- Every start/end is appended to runs/history.log, final metrics to runs/<run-name>/final_metrics.json.
"""
import argparse
import ast
import json
import time
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"


def log(msg):
    line = f"[{time.strftime('%F %T')}] {msg}"
    print(line, flush=True)
    RUNS.mkdir(exist_ok=True)
    with open(RUNS / "history.log", "a") as f:
        f.write(line + "\n")


def parse_kv(items):
    out = {}
    for it in items:
        k, v = it.split("=", 1)
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out


def run_finished(last):
    """A finished run's last.pt has its optimizer stripped; a mid-training one still has it."""
    import torch

    ck = torch.load(str(last), map_location="cpu", weights_only=False)
    return ck.get("optimizer") is None


def final_val(best, data, imgsz, run_dir):
    m = YOLO(str(best)).val(data=data, imgsz=imgsz, device="mps", workers=0, plots=False, project=str(RUNS), name=run_dir.name + "_val", exist_ok=True)
    per_class = {}
    for i, c in enumerate(m.box.ap_class_index):
        per_class[m.names[int(c)]] = {"AP50": float(m.box.ap50[i]), "AP50-95": float(m.box.ap[i])}
    res = {"mAP50": float(m.box.map50), "mAP50-95": float(m.box.map), "per_class": per_class}
    (run_dir / "final_metrics.json").write_text(json.dumps(res, indent=2))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("--model", default="weights/yolov8n-obb.pt")
    ap.add_argument("--data", default="configs/fair1m5.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--plots", action="store_true", help="save training plots (needs a font download on first use)")
    ap.add_argument("--set", nargs="*", default=[], help="extra Ultralytics overrides, key=value")
    a = ap.parse_args()

    data = str(ROOT / a.data)
    run_dir = RUNS / a.name
    last = run_dir / "weights" / "last.pt"
    t0 = time.time()

    if last.exists() and run_finished(last):
        # NOTE: calling resume=True on a finished checkpoint makes Ultralytics silently start a NEW
        # default training (dota8, 100 epochs), so a finished run must never be passed to train().
        log(f"{a.name} already finished, skipping to final validation")
    elif last.exists():
        log(f"RESUME {a.name} from {last}")
        YOLO(str(last)).train(resume=True)
    else:
        log(f"START {a.name} model={a.model} imgsz={a.imgsz} epochs={a.epochs} batch={a.batch} fraction={a.fraction} extra={a.set}")
        kw = dict(
            data=data, imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, workers=a.workers, fraction=a.fraction,
            patience=a.patience, device="mps", amp=False, deterministic=False, plots=a.plots,
            project=str(RUNS), name=a.name, exist_ok=True,
        )
        kw.update(parse_kv(a.set))
        YOLO(str(ROOT / a.model) if (ROOT / a.model).exists() else a.model).train(**kw)

    best = run_dir / "weights" / "best.pt"
    if best.exists():
        try:
            res = final_val(best, data, a.imgsz, run_dir)
            log(f"END {a.name} {(time.time() - t0) / 3600:.2f} h | mAP50={res['mAP50']:.3f} mAP50-95={res['mAP50-95']:.3f} | "
                + ", ".join(f"{k}={v['AP50']:.3f}" for k, v in res["per_class"].items()))
        except Exception as e:
            log(f"END {a.name} (final validation failed: {e})")
    else:
        log(f"END {a.name} (no best.pt found)")


if __name__ == "__main__":
    main()
