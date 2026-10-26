"""Evaluate frozen CNN runs and the platform on the independent 2026-10-26 batch.

This script never trains, selects, or modifies a model. Run from any directory:
    python cnn-inversion/src/eval_testbatch.py
"""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

from data import load_image


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
BATCH = REPO / "simulation-dataset/outputs/testbatch_20261026"
OUT = ROOT / "results/testbatch_20261026.json"
sys.path.insert(0, str(REPO / "platform"))
from predictor import SkyrmionPredictor  # noqa: E402


def verify_batch():
    manifest = json.loads((BATCH / "manifest.json").read_text(encoding="utf-8"))
    tables = {
        kind: pd.read_csv(BATCH / "labels" / f"labels_testbatch_{kind}.csv")
        for kind in ("single", "double", "mixed")
    }
    single, double, mixed = (tables[k] for k in ("single", "double", "mixed"))
    assert (len(single), len(double), len(mixed)) == (88, 220, 308)
    assert set(single.filename).isdisjoint(double.filename)
    assert set(mixed.filename) == set(single.filename) | set(double.filename)
    assert mixed.filename.is_unique
    assert set(mixed.kind) == {"single", "double"}
    for kind, table in (("single", single), ("double", double)):
        assert set(mixed.loc[mixed.kind == kind, "filename"]) == set(table.filename)
    assert sorted(mixed.eta.unique().tolist()) == list(range(30, 41))
    assert sorted(mixed.U.unique().tolist()) == [i / 2 for i in range(8)]
    assert sorted(double.separation.unique().tolist()) == list(range(30, 80, 5))
    assert set(zip(mixed.R, mixed.B)) == {(13.9, 0.48), (14.9, 0.52)}

    files = sorted((BATCH / "pom_images").glob("*.png"))
    assert len(files) == 308
    assert {p.name for p in files} == set(mixed.filename)
    sha = manifest["file_sha256"]
    assert set(sha) == {p.name for p in files}
    hashes = []
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == sha[path.name], f"SHA-256 mismatch: {path.name}"
        hashes.append(digest)
        with Image.open(path) as image:
            assert image.size == (300, 300) and image.mode == "L", path.name
            image.verify()
    assert len(set(hashes)) == 308
    existing_dirs = [
        REPO / "simulation-dataset/outputs/dataset/pom_images",
        REPO / "simulation-dataset/outputs/supplement/pom_images",
    ]
    existing = [p for folder in existing_dirs for p in folder.glob("*.png")]
    assert len(existing) == 1414
    assert {p.name for p in existing}.isdisjoint(mixed.filename)
    existing_hashes = {hashlib.sha256(p.read_bytes()).hexdigest() for p in existing}
    assert set(hashes).isdisjoint(existing_hashes)
    assert manifest["checks"]["zero_overlap_with_existing"] is True
    return mixed


def regression_metrics(y, pred):
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    err = pred - y
    return {
        "n": int(len(y)),
        "r2": float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(np.mean(err)),
    }


def classification_metrics(y, pred, classes):
    y = np.asarray(y)
    pred = np.asarray(pred)
    return {
        "n": int(len(y)),
        "accuracy": float(np.mean(y == pred)),
        "correct": int(np.sum(y == pred)),
        "classes": list(classes),
        "confusion_matrix": [
            [int(np.sum((y == a) & (pred == b))) for b in classes]
            for a in classes
        ],
    }


def by_kind(df, y, pred, metric, *args):
    result = {"all": metric(y, pred, *args)}
    for kind in ("single", "double"):
        mask = (df.kind == kind).to_numpy()
        if mask.any():
            result[kind] = metric(y[mask], pred[mask], *args)
    return result


def main():
    df = verify_batch()
    paths = [str(BATCH / "pom_images" / name) for name in df.filename]
    # Reuse the exact training preprocessing, once per image. No augmentation.
    images = np.stack([load_image(path).numpy() for path in paths])
    runs = {
        "pitch_final": ("single", "pitch", "eta"),
        "voltage": ("single", "regression", "U"),
        "energy_single": ("single", "regression", "free_energy"),
        "energy_double": ("double", "regression", "free_energy"),
        "mixed_pitch_final": ("all", "pitch", "eta"),
        "mixed_voltage": ("all", "regression", "U"),
        "mixed_energy": ("all", "regression", "free_energy"),
    }
    result = {
        "batch": "testbatch_20261026",
        "source_commit": "33179be",
        "tensorflow": tf.__version__,
        "preprocessing": "cnn-inversion/src/data.py:load_image; /255 + per-image z-score",
        "batch_verification": "308 files and labels, 300x300 L, all manifest SHA-256 match, zero overlap with 1414 existing images",
        "models": {},
    }
    for name, (scope, task, target) in runs.items():
        mask = np.ones(len(df), dtype=bool) if scope == "all" else (df.kind == scope).to_numpy()
        subset = df.loc[mask].reset_index(drop=True)
        meta = json.loads((ROOT / "runs" / name / "meta.json").read_text(encoding="utf-8"))
        model = tf.keras.models.load_model(ROOT / "runs" / name / "model.keras", compile=False)
        raw = model.predict(images[mask], batch_size=32, verbose=0)
        if task == "pitch":
            classes = meta["classes"]
            pred = np.asarray(classes)[np.argmax(raw, axis=1)]
            metrics = by_kind(subset, subset.eta.to_numpy(), pred, classification_metrics, classes)
        else:
            pred = raw.reshape(-1).astype(np.float64) * float(meta["label_std"]) + float(meta["label_mean"])
            metrics = by_kind(subset, subset[target].to_numpy(dtype=np.float64), pred, regression_metrics)
        result["models"][name] = {"scope": scope, "target": target, "metrics": metrics}
        print(f"{name}: {metrics['all']}", flush=True)
        del model
        tf.keras.backend.clear_session()

    # Exercise the public platform API itself, including routing and voltage clipping.
    platform = SkyrmionPredictor()
    predictions = []
    for i, path in enumerate(paths, 1):
        predictions.append(platform.predict(path))
        if i % 44 == 0:
            print(f"platform: {i}/308", flush=True)
    kind_pred = np.array([p["kind"] for p in predictions])
    eta_pred = np.array([p["eta"] for p in predictions])
    voltage_pred = np.array([p["voltage_V"] for p in predictions])
    energy_pred = np.array([p["free_energy_J"] for p in predictions])
    result["platform"] = {
        "router": by_kind(df, df.kind.to_numpy(), kind_pred, classification_metrics, ["single", "double"]),
        "pitch": by_kind(df, df.eta.to_numpy(), eta_pred, classification_metrics, list(range(30, 41))),
        "voltage": by_kind(df, df.U.to_numpy(dtype=float), voltage_pred, regression_metrics),
        "energy": by_kind(df, df.free_energy.to_numpy(dtype=float), energy_pred, regression_metrics),
        "voltage_clipped_count": sum(p["voltage_clipped_to_training_range"] for p in predictions),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
