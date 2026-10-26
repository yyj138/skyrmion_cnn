"""Add central-radius development groups and a fresh sealed geometry audit.

The sealed groups are fixed before training the v3 candidate and stored away
from all development CSVs. Neither the old 308-image batch nor the v2 sealed
batch is used for training or checkpoint selection.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from ansatz import generate_double_toron, apply_voltage_perturbation
from pom_imaging import compute_pom_image


ROOT = Path(__file__).resolve().parents[1] / "outputs"
GROUPS = {
    "train": [(14.4, 0.42), (14.4, 0.58)],
    "sealed": [(13.2, 0.51), (15.6, 0.49)],
    "audit": [(13.3, 0.50), (15.5, 0.50)],
}
FIELDS = ["filename", "eta", "U", "kind", "separation", "R", "B", "geometry_group", "split"]
SEPARATIONS = (30, 45, 60, 75)
VOLTAGES = (0.0, 1.0, 2.5, 3.5)


def make(split, geometries):
    names = {
        "train": "pitch_geometry_extra_train_v3",
        "sealed": "pitch_geometry_sealed_v3",
        "audit": "pitch_geometry_audit_v4",
    }
    target = ROOT / names[split]
    image_dir = target / "pom_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.time()
    for group_index, (radius, wall) in enumerate(geometries):
        group = f"g_{radius:.1f}_{wall:.2f}"
        for eta in range(30, 41):
            for setting_index, separation in enumerate(SEPARATIONS):
                voltage = (VOLTAGES[setting_index] if split == "audit" else
                           VOLTAGES[(setting_index + eta - 30 + group_index) % 4])
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
        print(f"{split}: {group} complete", flush=True)
    with (target / "labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    paths = sorted(image_dir.glob("*.png"))
    assert len(paths) == len(rows) == 88
    hashes = {}
    for path in paths:
        with Image.open(path) as image:
            assert image.size == (300, 300) and image.mode == "L"
            image.verify()
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(set(hashes.values())) == 88
    manifest = {
        "purpose": "additional development" if split == "train" else "one-time sealed evaluation",
        "generator": "simulation-dataset/src/generate_pitch_geometry_v3.py",
        "groups": geometries,
        "voltage_assignment": "fixed across all eta and geometries" if split == "audit" else "rotated by eta and geometry",
        "images": 88,
        "elapsed_seconds": round(time.time() - started, 1),
        "sha256": hashes,
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"{target}: 88 verified images", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        make("audit", GROUPS["audit"])
    else:
        make("train", GROUPS["train"])
        make("sealed", GROUPS["sealed"])
