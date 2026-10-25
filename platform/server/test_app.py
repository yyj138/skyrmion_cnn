"""API checks that do not load the large research models."""

import csv
from io import BytesIO
from pathlib import Path
import unittest

from fastapi.testclient import TestClient
from PIL import Image

from server.app import EXAMPLES, PROJECT_ROOT, create_app


def png_bytes(size=(300, 300)):
    output = BytesIO()
    Image.new("L", size).save(output, format="PNG")
    return output.getvalue()


def example_bytes(example_id="single"):
    return (PROJECT_ROOT / "platform" / "assets" / "examples" / EXAMPLES[example_id]["filename"]).read_bytes()


class FakePredictor:
    def __init__(self):
        self.paths = []

    def predict(self, path):
        path = Path(path)
        assert path.is_file()
        self.paths.append(path)
        return {
            "kind": "single",
            "eta": 36,
            "voltage_V": 1.5,
            "voltage_clipped_to_training_range": False,
            "free_energy_J": 9.0e-14,
            "model_scope": "synthetic_POM_300x300",
        }


class ApiTests(unittest.TestCase):
    def test_model_reuse_upload_validation_and_temporary_cleanup(self):
        predictor = FakePredictor()
        loads = []

        def factory():
            loads.append(True)
            return predictor

        with TestClient(create_app(factory)) as client:
            self.assertEqual(client.get("/api/health").status_code, 200)
            examples = client.get("/api/examples").json()
            self.assertEqual(len(examples), 2)
            for example in examples:
                labels_file = PROJECT_ROOT / "simulation-dataset" / "outputs" / "supplement" / "labels" / f"labels_supplement_{example['id']}.csv"
                with labels_file.open(newline="", encoding="utf-8") as source:
                    label = next(row for row in csv.DictReader(source) if row["filename"] == EXAMPLES[example["id"]]["filename"])
                self.assertEqual(example["reference"]["kind"], label["kind"])
                self.assertEqual(example["reference"]["eta"], int(label["eta"]))
                self.assertEqual(example["reference"]["voltage_V"], float(label["U"]))
                self.assertEqual(example["reference"]["free_energy_J"], float(label["free_energy"]))
            for _ in range(2):
                response = client.post(
                    "/api/predict", files={"image": ("sample.png", example_bytes(), "image/png")}
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["eta"], 36)

            self.assertEqual(len(loads), 1)
            self.assertEqual(len(set(predictor.paths)), 2)
            self.assertTrue(all(not path.exists() for path in predictor.paths))

            bad_size = client.post(
                "/api/predict", files={"image": ("small.png", png_bytes((200, 200)), "image/png")}
            )
            self.assertEqual(bad_size.status_code, 400)
            self.assertIn("300×300", bad_size.json()["detail"])

            blank = client.post(
                "/api/predict", files={"image": ("blank.png", png_bytes(), "image/png")}
            )
            self.assertEqual(blank.status_code, 400)
            self.assertIn("纹理不足", blank.json()["detail"])

            color_image = BytesIO()
            Image.new("RGB", (300, 300), (200, 20, 20)).save(color_image, format="PNG")
            colorful = client.post(
                "/api/predict", files={"image": ("color.png", color_image.getvalue(), "image/png")}
            )
            self.assertEqual(colorful.status_code, 400)
            self.assertIn("彩色", colorful.json()["detail"])

            undecodable = client.post(
                "/api/predict", files={"image": ("bad.png", b"not an image", "image/png")}
            )
            self.assertEqual(undecodable.status_code, 400)
            self.assertEqual(len(predictor.paths), 2)

    def test_model_load_failure_reports_unavailable(self):
        def fail_to_load():
            raise FileNotFoundError("model.keras")

        with TestClient(create_app(fail_to_load)) as client:
            self.assertEqual(client.get("/api/health").status_code, 503)
            self.assertEqual(
                client.post("/api/predict", files={"image": ("sample.png", example_bytes(), "image/png")}).status_code,
                503,
            )


if __name__ == "__main__":
    unittest.main()
