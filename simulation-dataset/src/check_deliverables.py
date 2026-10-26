"""交付前校验: 训练图纯净性 + 标签一致性

用法（工作目录不限）:
    python check_deliverables.py

路径以本文件位置为基准推导（simulation-dataset/outputs/...），不再写死绝对路径。
校验不通过时以非零状态码退出，便于脚本/CI 调用。
"""
import csv
import sys
from pathlib import Path

from PIL import Image

# simulation-dataset/src/check_deliverables.py -> simulation-dataset/
ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
POM_DIR = OUTPUTS / "dataset" / "pom_images"
PRES_DIR = OUTPUTS / "visualizations" / "presentation"
DEMO_LABELS = OUTPUTS / "dataset" / "labels" / "labels_demo.csv"

GRAY_MODES = ("L", "1", "I;16")
failures = []


def check_training_images():
    """1. 训练图全部为灰度（无伪彩污染）"""
    if not POM_DIR.is_dir():
        failures.append(f"训练图目录不存在: {POM_DIR}")
        return
    bad = []
    n = 0
    for path in sorted(POM_DIR.glob("*.png")):
        n += 1
        with Image.open(path) as img:
            if img.mode not in GRAY_MODES:
                bad.append((path.name, img.mode))
    print(f"训练图共 {n} 张，非灰度图: {bad if bad else '无 (全部纯净)'}")
    if bad:
        failures.append(f"{len(bad)} 张训练图不是灰度")


def check_presentation_suffix():
    """2. presentation 图全部带 _pres 后缀（防止伪彩图混入训练目录）"""
    if not PRES_DIR.is_dir():
        print(f"展示图目录不存在（跳过）: {PRES_DIR}")
        return
    pres_files = sorted(p.name for p in PRES_DIR.glob("*.png"))
    wrong = [name for name in pres_files if not name.endswith("_pres.png")]
    print(f"展示图共 {len(pres_files)} 张，命名不合规: {wrong if wrong else '无'}")
    if wrong:
        failures.append(f"{len(wrong)} 张展示图缺少 _pres 后缀")


def check_demo_labels():
    """3. labels_demo.csv 无 NaN 且图片一一对应"""
    if not DEMO_LABELS.is_file():
        failures.append(f"demo 标签不存在: {DEMO_LABELS}")
        return
    with DEMO_LABELS.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    nan_free = all(row["free_energy"] not in ("", "nan", "NaN") for row in rows)
    missing = [row["filename"] for row in rows if not (POM_DIR / row["filename"]).is_file()]
    print(f"demo 标签 {len(rows)} 行，无 NaN: {nan_free}，缺图: {missing if missing else '无'}")
    if not nan_free:
        failures.append("demo 标签存在 NaN/空 free_energy")
    if missing:
        failures.append(f"demo 标签引用了 {len(missing)} 张不存在的图片")


def main():
    print(f"仓库根目录: {ROOT}")
    check_training_images()
    check_presentation_suffix()
    check_demo_labels()
    if failures:
        print("\n校验未通过:")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("\n全部校验通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
