"""Fine-tune a double-toron pitch model with geometry-grouped validation.

The 308-image testbatch_20261026 is never read. Sealed geometry images are
loaded only after training and checkpoint selection have finished.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from data import make_dataset


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
SIM = REPO / "simulation-dataset/outputs"
DEV = SIM / "pitch_geometry_development_v2"
BASE = ROOT / "runs/supplement_geometry_v1/mixed_pitch_final/model.keras"
CLASSES = list(range(30, 41))


def rows_from_existing():
    base = SIM / "dataset"
    labels = pd.read_csv(base / "labels/labels_mixed.csv")
    split = pd.read_csv(base / "splits/split_pitch_classification_mixed.csv")
    orig = labels.merge(split[["filename", "split"]], on="filename", validate="one_to_one")
    orig = orig.loc[orig.kind == "double"].copy()
    orig["path"] = orig.filename.map(lambda f: str(base / "pom_images" / f))

    supplement = SIM / "supplement"
    sup_labels = pd.read_csv(supplement / "labels/labels_supplement_mixed.csv")
    sup_split = pd.read_csv(ROOT / "data/supplement_split.csv")
    sup = sup_labels.merge(sup_split[["filename", "split"]], on="filename", validate="one_to_one")
    sup = sup.loc[sup.kind == "double"].copy()
    sup["path"] = sup.filename.map(lambda f: str(supplement / "pom_images" / f))
    return orig, sup


def rows_from_geometry(root):
    df = pd.read_csv(root / "labels.csv")
    assert set(df.kind) == {"double"}
    df["path"] = df.filename.map(lambda f: str(root / "pom_images" / f))
    return df


def arrays(df):
    assert set(df.eta).issubset(CLASSES)
    paths = df.path.to_numpy(dtype=str)
    labels = df.eta.to_numpy(dtype=np.int32) - 30
    assert all(Path(p).is_file() for p in paths)
    return paths, labels


def evaluate(model, df):
    paths, labels = arrays(df)
    probabilities = model.predict(
        make_dataset(paths, labels, 32, training=False, deterministic=True), verbose=0
    )
    pred = probabilities.argmax(axis=1)
    return {
        "n": int(len(labels)),
        "correct": int(np.sum(pred == labels)),
        "accuracy": float(np.mean(pred == labels)),
        "within_one": float(np.mean(np.abs(pred - labels) <= 1)),
        "mean_abs_eta_error": float(np.mean(np.abs(pred - labels))),
    }


def by_geometry(model, df):
    return {group: evaluate(model, subset) for group, subset in df.groupby("geometry_group")}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="geometry_double_pitch_v2")
    parser.add_argument("--extra-dev", default=None)
    parser.add_argument("--sealed", default="pitch_geometry_sealed_v2")
    parser.add_argument("--freeze-blocks", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--new-repeats", type=int, default=3)
    parser.add_argument("--sup-repeats", type=int, default=2)
    args = parser.parse_args()
    out = ROOT / "runs" / args.out
    sealed_dir = SIM / args.sealed
    tf.keras.utils.set_random_seed(42)
    orig, sup = rows_from_existing()
    geometry = rows_from_geometry(DEV)
    assert set(geometry.split) == {"train", "val"}
    assert set(geometry.loc[geometry.split == "train", "geometry_group"]).isdisjoint(
        set(geometry.loc[geometry.split == "val", "geometry_group"])
    )
    assert len(geometry) == 264

    original_train = orig.loc[orig.split == "train"]
    original_val = orig.loc[orig.split == "val"]
    supplement_train = sup.loc[sup.split == "development"]
    geometry_train = geometry.loc[geometry.split == "train"]
    geometry_val = geometry.loc[geometry.split == "val"]
    assert len(geometry_train) == 176 and len(geometry_val) == 88
    if args.extra_dev:
        extra = rows_from_geometry(SIM / args.extra_dev)
        assert set(extra.split) == {"train"}
        assert set(extra.geometry_group).isdisjoint(set(geometry_val.geometry_group))
        geometry_train = pd.concat([geometry_train, extra], ignore_index=True)
    train = pd.concat(
        [original_train] + [supplement_train] * args.sup_repeats
        + [geometry_train] * args.new_repeats,
        ignore_index=True,
    )
    assert not any("testbatch_20261026" in path for path in train.path)
    train_paths, train_labels = arrays(train)
    val_paths, val_labels = arrays(geometry_val)

    model = tf.keras.models.load_model(BASE, compile=False)
    baseline = {
        "geometry_validation": by_geometry(model, geometry_val),
        "original_validation": evaluate(model, original_val),
    }
    print("baseline", baseline, flush=True)

    # Preserve low-level optical texture filters; adapt upper layers and head.
    for layer in model.layers:
        if any(layer.name.startswith(f"block{i}_") for i in range(1, args.freeze_blocks + 1)):
            layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "model.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(checkpoint, monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
    ]
    history = model.fit(
        make_dataset(train_paths, train_labels, 32, training=True,
                     deterministic=True, seed=42, augment_probability=0.1),
        validation_data=make_dataset(val_paths, val_labels, 32,
                                     training=False, deterministic=True),
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )
    best = tf.keras.models.load_model(checkpoint, compile=False)
    candidate = {
        "geometry_validation": by_geometry(best, geometry_val),
        "original_validation": evaluate(best, original_val),
    }
    print("candidate", candidate, flush=True)

    # The promotion decision is fixed using development data only, before the
    # sealed CSV is loaded. No threshold or weight is adjusted after this point.
    baseline_val = sum(v["correct"] for v in baseline["geometry_validation"].values())
    candidate_val = sum(v["correct"] for v in candidate["geometry_validation"].values())
    promote = candidate_val > baseline_val and candidate["original_validation"]["accuracy"] >= 0.98

    sealed = rows_from_geometry(sealed_dir)
    assert set(sealed.split) == {"sealed"} and len(sealed) == 88
    sealed_once = {
        "baseline": by_geometry(tf.keras.models.load_model(BASE, compile=False), sealed),
        "candidate": by_geometry(best, sealed),
    }
    print("sealed one-time audit", sealed_once, flush=True)
    record = {
        "purpose": "double-toron pitch geometry generalization",
        "source_commit": "33179be",
        "seed": 42,
        "base_model": str(BASE.relative_to(REPO)),
        "base_sha256": sha256(BASE),
        "development_manifest_sha256": sha256(DEV / "manifest.json"),
        "extra_development_manifest_sha256": sha256(SIM / args.extra_dev / "manifest.json") if args.extra_dev else None,
        "sealed_manifest_sha256": sha256(sealed_dir / "manifest.json"),
        "train_unique": {
            "original": int(len(original_train)),
            "supplement_development": int(len(supplement_train)),
            "new_geometry": int(len(geometry_train)),
        },
        "train_repeats": {"original": 1, "supplement_development": args.sup_repeats,
                          "new_geometry": args.new_repeats},
        "new_geometry_validation_count": int(len(geometry_val)),
        "epochs_run": len(history.history["loss"]),
        "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
        "learning_rate": args.lr,
        "frozen_blocks": args.freeze_blocks,
        "baseline": baseline,
        "candidate": candidate,
        "promote_from_development_only": bool(promote),
        "sealed_one_time_audit": sealed_once,
        "history": {key: [float(v) for v in values] for key, values in history.history.items()},
    }
    (out / "metrics.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"promote_from_development_only={promote}; saved {out}", flush=True)


if __name__ == "__main__":
    main()
