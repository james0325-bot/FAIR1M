#!/usr/bin/env python3
"""Convert the FAIR1M (Hugging Face mirror) subset to Ultralytics OBB format, 5 classes.

Input  (--src):  <src>/images/*.tif   and   <src>/labelXmls/*.xml
Output (--dst):  <dst>/images/{train,val}/*.jpg
                 <dst>/labels/{train,val}/*.txt     (class x1 y1 x2 y2 x3 y3 x4 y4, normalized)
                 configs/fair1m5.yaml

- 37 sub-categories are merged into 5 classes (see CLASS_MAP).
- RGBA/other modes are converted to RGB; images are saved as JPEG to save disk space.
- Images larger than --tile-above px are cut into --tile px tiles with --overlap px overlap;
  smaller ones are kept whole.
- The train/val split is made on the SOURCE image, so tiles of one image never end up on both sides.

Run with the project venv:  .venv/bin/python scripts/convert_fair1m.py
"""
import argparse
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

CLASS_NAMES = ["ship", "vehicle", "airplane", "court", "road"]

CLASS_MAP = {
    # ship
    "Passenger Ship": 0, "Motorboat": 0, "Fishing Boat": 0, "Tugboat": 0, "Engineering Ship": 0,
    "Liquid Cargo Ship": 0, "Dry Cargo Ship": 0, "Warship": 0, "other-ship": 0,
    # vehicle
    "Small Car": 1, "Bus": 1, "Cargo Truck": 1, "Dump Truck": 1, "other-vehicle": 1, "Van": 1,
    "Trailer": 1, "Tractor": 1, "Excavator": 1, "Truck Tractor": 1,
    # airplane
    "Boeing737": 2, "Boeing747": 2, "Boeing777": 2, "Boeing787": 2, "ARJ21": 2, "C919": 2,
    "A220": 2, "A321": 2, "A330": 2, "A350": 2, "other-airplane": 2,
    # court
    "Baseball Field": 3, "Basketball Court": 3, "Football Field": 3, "Tennis Court": 3,
    # road
    "Roundabout": 4, "Intersection": 4, "Bridge": 4,
}


def parse_xml(path):
    """Return a list of (class_id, [(x, y) * 4]) in pixel coordinates, plus a Counter of unknown names."""
    text = path.read_text(encoding="utf-8", errors="replace")
    # Refuse DTD/entity tricks; the FAIR1M files never need them.
    if "<!DOCTYPE" in text or "<!ENTITY" in text:
        raise ValueError("DTD/entity declaration found")
    root = ET.fromstring(text)
    objs, unknown = [], Counter()
    for obj in root.iter("object"):
        name_el = obj.find("possibleresult/name")
        pts_el = obj.findall("points/point")
        if name_el is None or len(pts_el) < 4:
            continue
        name = (name_el.text or "").strip()
        cid = CLASS_MAP.get(name)
        if cid is None:
            unknown[name] += 1
            continue
        pts = []
        for p in pts_el[:4]:  # 5th point repeats the 1st
            x, y = p.text.split(",")
            pts.append((float(x), float(y)))
        objs.append((cid, pts))
    return objs, unknown


def bbox(pts):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def label_lines(objs, ox, oy, w, h, min_visible):
    """Objects shifted into the (ox, oy, w, h) window, clamped, normalized -> label lines."""
    lines, counts = [], Counter()
    for cid, pts in objs:
        shifted = [(x - ox, y - oy) for x, y in pts]
        cx = sum(p[0] for p in shifted) / 4
        cy = sum(p[1] for p in shifted) / 4
        if not (0 <= cx < w and 0 <= cy < h):
            continue
        clamped = [(min(max(x, 0.0), w), min(max(y, 0.0), h)) for x, y in shifted]
        full = area(bbox(shifted))
        if full <= 0 or area(bbox(clamped)) / full < min_visible:
            continue
        coords = []
        for x, y in clamped:
            coords += [f"{x / w:.6f}", f"{y / h:.6f}"]
        lines.append(f"{cid} " + " ".join(coords))
        counts[cid] += 1
    return lines, counts


def starts(length, tile, stride):
    if length <= tile:
        return [0]
    xs = list(range(0, length - tile + 1, stride))
    if xs[-1] != length - tile:
        xs.append(length - tile)
    return xs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="datasets/raw/fair1m_hf/data")
    ap.add_argument("--dst", default="datasets/processed/fair1m5")
    ap.add_argument("--yaml", default="configs/fair1m5.yaml")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="use only N source images (smoke test)")
    ap.add_argument("--tile-above", type=int, default=1536, help="tile images whose longest side exceeds this")
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--overlap", type=int, default=200)
    ap.add_argument("--min-visible", type=float, default=0.5, help="min fraction of an object that must stay in a tile")
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--force", action="store_true", help="delete --dst first if it exists")
    a = ap.parse_args()

    src, dst = Path(a.src), Path(a.dst)
    img_dir, xml_dir = src / "images", src / "labelXmls"
    if not img_dir.is_dir() or not xml_dir.is_dir():
        sys.exit(f"missing {img_dir} or {xml_dir}; has the download finished?")
    if dst.exists():
        if not a.force:
            sys.exit(f"{dst} already exists; pass --force to rebuild it")
        shutil.rmtree(dst)

    stems = sorted(p.stem for p in img_dir.glob("*.tif") if (xml_dir / f"{p.stem}.xml").exists())
    if not stems:
        sys.exit("no image/xml pairs found")
    random.Random(a.seed).shuffle(stems)
    if a.limit:
        stems = stems[: a.limit]
    n_val = max(1, round(len(stems) * a.val_frac))
    split_of = {s: ("val" if i < n_val else "train") for i, s in enumerate(stems)}

    for sp in ("train", "val"):
        (dst / "images" / sp).mkdir(parents=True)
        (dst / "labels" / sp).mkdir(parents=True)

    stride = a.tile - a.overlap
    stats = {sp: Counter() for sp in ("train", "val")}
    unknown_all, skipped = Counter(), []

    for k, stem in enumerate(stems, 1):
        sp = split_of[stem]
        try:
            objs, unknown = parse_xml(xml_dir / f"{stem}.xml")
            unknown_all.update(unknown)
            with Image.open(img_dir / f"{stem}.tif") as im:
                im = im.convert("RGB")
                W, H = im.size
                if max(W, H) > a.tile_above:
                    wins = [(x, y, a.tile, a.tile) for y in starts(H, a.tile, stride) for x in starts(W, a.tile, stride)]
                else:
                    wins = [(0, 0, W, H)]
                for x, y, w, h in wins:
                    w, h = min(w, W - x), min(h, H - y)
                    lines, counts = label_lines(objs, x, y, w, h, a.min_visible)
                    name = stem if len(wins) == 1 else f"{stem}_{x}_{y}"
                    im.crop((x, y, x + w, y + h)).save(dst / "images" / sp / f"{name}.jpg", quality=a.jpeg_quality)
                    (dst / "labels" / sp / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
                    stats[sp]["images"] += 1
                    for cid, n in counts.items():
                        stats[sp][CLASS_NAMES[cid]] += n
        except Exception as e:  # keep going; report at the end
            skipped.append((stem, f"{type(e).__name__}: {e}"))
        if k % 100 == 0 or k == len(stems):
            print(f"  {k}/{len(stems)} source images", flush=True)

    Path(a.yaml).parent.mkdir(parents=True, exist_ok=True)
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    Path(a.yaml).write_text(
        f"# generated by scripts/convert_fair1m.py\npath: {dst.resolve()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n"
    )

    print(f"\nwrote {a.yaml}")
    for sp in ("train", "val"):
        s = stats[sp]
        print(f"{sp:5s}: {s['images']} images | " + ", ".join(f"{n}={s[n]}" for n in CLASS_NAMES))
    if unknown_all:
        print("unmapped class names (ignored):", dict(unknown_all))
    if skipped:
        print(f"skipped {len(skipped)} source images:")
        for stem, why in skipped[:10]:
            print(f"  {stem}: {why}")


if __name__ == "__main__":
    main()
