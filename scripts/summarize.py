#!/usr/bin/env python3
"""Collect every run in runs/ into runs/SUMMARY.md (one table, best first).

    .venv/bin/python scripts/summarize.py

Uses runs/<name>/results.csv (epochs done, best epoch, wall time) and final_metrics.json (per-class AP50).
Safe to run any time, including while training is in progress.
"""
import csv
import json
import time
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent / "runs"
CLASSES = ["ship", "vehicle", "airplane", "court", "road"]


def load(run):
    rows = []
    f = run / "results.csv"
    if f.exists():
        rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(open(f))]
    fm = run / "final_metrics.json"
    final = json.loads(fm.read_text()) if fm.exists() else None
    return rows, final


def main():
    out = []
    # args.yaml of a resumed run says model=last.pt, so take the original model from its START line in history.log.
    start_models = {}
    hist = RUNS / "history.log"
    if hist.exists():
        for line in hist.read_text().splitlines():
            parts = line.split()
            if len(parts) > 4 and parts[2] == "START" and parts[4].startswith("model="):
                start_models[parts[3]] = Path(parts[4][len("model="):]).name
    for run in sorted(p for p in RUNS.iterdir() if p.is_dir() and (p / "results.csv").exists()):
        rows, final = load(run)
        if not rows:
            continue
        best = max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"]))
        args = {}
        af = run / "args.yaml"
        if af.exists():
            for line in af.read_text().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    args[k.strip()] = v.strip()
        # Ultralytics restarts the "time" column from ~0 on every resume, so add up each segment's last value.
        t = [float(r["time"]) for r in rows]
        secs = sum(t[i - 1] for i in range(1, len(t)) if t[i] < t[i - 1]) + t[-1]
        out.append({
            "name": run.name, "model": start_models.get(run.name, Path(args.get("model", "?")).name), "imgsz": args.get("imgsz", "?"),
            "epochs": f"{len(rows)}/{args.get('epochs', '?')}", "hours": secs / 3600,
            "best_epoch": best["epoch"], "map50": float(best["metrics/mAP50(B)"]),
            "map5095": float(best["metrics/mAP50-95(B)"]),
            "done": (run / "DONE").exists(), "stopped": (run / "STOPPED").exists(),
            "per_class": (final or {}).get("per_class", {}),
        })
    out.sort(key=lambda r: -r["map5095"])
    lines = [f"# 訓練結果彙整（{time.strftime('%F %T')}）", "",
             "| run | 模型 | imgsz | epoch | 時間(h) | 最佳 epoch | mAP50 | mAP50-95 | 狀態 |", "|---|---|---|---|---|---|---|---|---|"]
    for r in out:
        lines.append(f"| {r['name']} | {r['model']} | {r['imgsz']} | {r['epochs']} | {r['hours']:.1f} | {r['best_epoch']} | "
                     f"{r['map50']:.3f} | {r['map5095']:.3f} | {'完成' if r['done'] else '提早停止' if r['stopped'] else '進行中/中斷'} |")
    lines += ["", "## 每類別 AP50（最終驗證，best.pt）", "", "| run | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * len(CLASSES)]
    for r in out:
        pc = r["per_class"]
        if pc:
            lines.append(f"| {r['name']} | " + " | ".join(f"{pc[c]['AP50']:.3f}" if c in pc else "-" for c in CLASSES) + " |")
    (RUNS / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
