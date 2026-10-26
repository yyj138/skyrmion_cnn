"""Generate double-toron geometry development and sealed evaluation images.

The geometry groups are fixed here before model training. The sealed images are
written to a separate directory and must never be used for training or tuning.
The 2026-10-26 308-image test batch is not read or modified.
"""

import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from ansatz import generate_double_toron, apply_voltage_perturbation
from pom_imaging import compute_pom_image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
GROUPS = {
    "train": [(13.0, 0.42), (13.0, 0.58), (15.8, 0.42), (15.8, 0.58)],
    "val": [(13.6, 0.52), (15.2, 0.48)],
    "sealed": [(14.2, 0.44), (14.6, 0.56)],
}
FIELDS = ["filename", "eta", "U", "kind", "separation", "R", "B", "geometry_group", "split"]
SEPARATIONS = (30, 45, 60, 75)
VOLTAGES = (0.0, 1.0, 2.5, 3.5)


def generate(split, geometries):
    name = "pitch_geometry_sealed_v2" if split == "sealed" else "pitch_geometry_development_v2"
    target = OUTPUT / name
    image_dir = target / "pom_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.time()
    for group_index, (radius, wall) in enumerate(geometries):
        group = f"g_{radius:.1f}_{wall:.2f}"
        for eta in range(30, 41):
            for setting_index, separation in enumerate(SEPARATIONS):
                # Rotate voltage assignment across eta and geometry to avoid
                # learning a spurious voltage/separation shortcut.
                voltage = VOLTAGES[(setting_index + eta - 30 + group_index) % 4]
                filename = f"double_eta{eta}_U{voltage:.2f}_sep{separation}_{group}.png"
                path = image_dir / filename
                if not path.exists():
                    nx, ny, nz, _ = generate_double_toron(
                        separation=separation, eta=eta, R=radius, B=wall
                    )
                    nx, ny, nz = apply_voltage_perturbation(nx, ny, nz, voltage)
                    pixels = (np.asarray(compute_pom_image(nx, ny, nz)) * 255).clip(0, 255).astype(np.uint8)
                    Image.fromarray(pixels).save(path)
                rows.append({
                    "filename": filename, "eta": eta, "U": voltage, "kind": "double",
                    "separation": separation, "R": radius, "B": wall,
                    "geometry_group": group, "split": split,
                })
        print(f"{split}: {group} complete ({len(rows)} rows)", flush=True)
    return target, rows, time.time() - started


def write_dataset(target, rows, elapsed):
    image_dir = target / "pom_images"
    csv_path = target / "labels.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    paths = sorted(image_dir.glob("*.png"))
    assert len(paths) == len(rows) and {p.name for p in paths} == {r["filename"] for r in rows}
    hashes = {}
    for path in paths:
        with Image.open(path) as image:
            assert image.size == (300, 300) and image.mode == "L"
            image.verify()
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(set(hashes.values())) == len(paths)
    manifest = {
        "purpose": "geometry-grouped double-toron pitch development" if "development" in target.name else "one-time sealed evaluation",
        "generator": "simulation-dataset/src/generate_pitch_geometry_v2.py",
        "groups": GROUPS,
        "images": len(rows),
        "elapsed_seconds": round(elapsed, 1),
        "sha256": hashes,
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"{target}: {len(rows)} verified images in {elapsed:.1f}s", flush=True)


def main():
    train_target, train_rows, train_seconds = generate("train", GROUPS["train"])
    val_target, val_rows, val_seconds = generate("val", GROUPS["val"])
    assert train_target == val_target
    write_dataset(train_target, train_rows + val_rows, train_seconds + val_seconds)
    sealed_target, sealed_rows, sealed_seconds = generate("sealed", GROUPS["sealed"])
    write_dataset(sealed_target, sealed_rows, sealed_seconds)


if __name__ == "__main__":
    main()
