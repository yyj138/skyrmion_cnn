"""One-time fixed-grid audit of the old and geometry-trained double pitch models."""

import hashlib
import json
from pathlib import Path

import tensorflow as tf
from PIL import Image

from train_double_pitch_geometry import by_geometry, rows_from_geometry


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
AUDIT = REPO / "simulation-dataset/outputs/pitch_geometry_audit_v4"
OLD = ROOT / "runs/supplement_geometry_v1/mixed_pitch_final/model.keras"
NEW = ROOT / "runs/geometry_double_pitch_v3/model.keras"
RESULT = ROOT / "results/double_pitch_audit_v4.json"


def main():
    df = rows_from_geometry(AUDIT)
    assert len(df) == 88 and set(df.split) == {"audit"}
    manifest = json.loads((AUDIT / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["sha256"]) == set(df.filename)
    for path in df.path:
        path = Path(path)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["sha256"][path.name]
        with Image.open(path) as image:
            assert image.size == (300, 300) and image.mode == "L"
    reports = {}
    for label, path in (("old", OLD), ("candidate", NEW)):
        model = tf.keras.models.load_model(path, compile=False)
        reports[label] = by_geometry(model, df)
        tf.keras.backend.clear_session()
    result = {
        "purpose": "audit with fixed voltage and separation assignments across eta",
        "dataset": str(AUDIT.relative_to(REPO)),
        "old_model": str(OLD.relative_to(REPO)),
        "candidate_model": str(NEW.relative_to(REPO)),
        "groups": reports,
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
