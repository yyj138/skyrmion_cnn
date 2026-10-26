"""Predict parameters from one 300x300 POM image using the selected models.

This bundle has been evaluated on synthetic images. Real microscopy images
require separate calibration and validation before interpreting the outputs.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf
from PIL import Image

CNN_ROOT = Path(__file__).resolve().parents[1] / "cnn-inversion"
SOURCE_DIR = CNN_ROOT / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from data import load_image


class SkyrmionPredictor:
    """Load models once and reuse this object for many platform requests."""

    def __init__(self):
        self.router = tf.keras.models.load_model(
            CNN_ROOT / "runs/supplement_router_v1/model.keras", compile=False
        )
        self.pitch_single = tf.keras.models.load_model(
            CNN_ROOT / "runs/supplement_geometry_v1/pitch_final/model.keras", compile=False
        )
        self.pitch_mixed = tf.keras.models.load_model(
            CNN_ROOT / "runs/geometry_double_pitch_v3/model.keras", compile=False
        )
        self.voltage = tf.keras.models.load_model(
            CNN_ROOT / "runs/supplement_voltage_v1/mixed_voltage/model.keras", compile=False
        )
        self.energy = tf.keras.models.load_model(
            CNN_ROOT / "runs/mixed_energy/model.keras", compile=False
        )
        self.pitch_classes = json.loads(
            (CNN_ROOT / "runs/mixed_pitch_final/meta.json").read_text(encoding="utf-8")
        )["classes"]
        self.voltage_meta = json.loads(
            (CNN_ROOT / "runs/mixed_voltage/meta.json").read_text(encoding="utf-8")
        )
        self.energy_meta = json.loads(
            (CNN_ROOT / "runs/mixed_energy/meta.json").read_text(encoding="utf-8")
        )

    def predict(self, path):
        path = Path(path)
        with Image.open(path) as image:
            if image.size != (300, 300):
                raise ValueError(f"Expected a 300x300 image, got {image.size}")
        tensor = tf.expand_dims(load_image(str(path)), axis=0)
        double_probability = float(self.router(tensor, training=False).numpy()[0, 0])
        kind = "double" if double_probability >= 0.5 else "single"
        pitch_model = self.pitch_mixed if kind == "double" else self.pitch_single
        pitch_probabilities = pitch_model(tensor, training=False).numpy()[0]
        eta = int(self.pitch_classes[int(np.argmax(pitch_probabilities))])
        voltage_scaled = float(self.voltage(tensor, training=False).numpy()[0, 0])
        energy_scaled = float(self.energy(tensor, training=False).numpy()[0, 0])
        voltage_raw = voltage_scaled * float(self.voltage_meta["label_std"]) + float(self.voltage_meta["label_mean"])
        voltage = min(3.5, max(0.0, voltage_raw))
        energy = energy_scaled * float(self.energy_meta["label_std"]) + float(self.energy_meta["label_mean"])
        return {
            "kind": kind,
            "eta": eta,
            "voltage_V": voltage,
            "voltage_clipped_to_training_range": voltage != voltage_raw,
            "free_energy_J": energy,
            "model_scope": "synthetic_POM_300x300",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    args = parser.parse_args()
    result = SkyrmionPredictor().predict(args.image)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
