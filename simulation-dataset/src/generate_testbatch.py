"""为成员 B 生成独立测试批次（带参数标签，不参与任何训练）

用途
----
成员 B 需要一批"带参数标签、但与训练/补充数据都不重复"的模拟 POM 图，用于检验
已训练模型与平台在新构型上的表现。本脚本生成该批次。

与既有数据的关系（重要）
------------------------
| 数据集 | 目录 | 数量 | ansatz (R, B) |
|---|---|---|---|
| 原训练集 | outputs/dataset/ | 968 (+6 demo) | 固定论文值 (14.4, 0.50) |
| 补充集   | outputs/supplement/ | 440 | v1 (13.4, 0.45)、v2 (15.4, 0.55)、v3 (14.4, 0.45) |
| **本批次** | outputs/testbatch_<日期>/ | 308 | **w1 (13.9, 0.48)、w2 (14.9, 0.52)** |

- 方法与原数据完全一致：论文 Eq.(3)-(6) 解析 ansatz + 手性扭转 q0·(z−Cz) 注入 +
  经验电压扰动 |nz|·(1−0.15·(U/3.5)²) + Frank-Oseen 体积积分（Eq.1）+ Jones 矩阵
  正交偏振 POM（λ=500nm，Δn=0.06）。**未运行 PDE 松弛**（与原数据同口径）。
- 仅 ansatz 几何参数 (R, B) 取新值，且与 (14.4, 0.5)、(13.4, 0.45)、(15.4, 0.55)、
  (14.4, 0.45) 全部不同 → 构型全新，文件名后缀 w1/w2 与既有的 v1/v2/v3 不冲突。
- 自由能标签对**每个新构型**重新做体积积分，不是复制既有标签。

参数覆盖（确定性网格，无随机抽样；SEED 仅作记录）
-------------------------------------------------
- 单 toron 88 张：11 个 η（30–40）× 8 档 U（0.0–3.5 V），w1/w2 按 (η+U索引) 奇偶交替
- 双 toron 220 张：11 个 η × 10 档间距（30–75，步长 5）× 2 个电压，
  电压配对规则 U[si] 与 U[(si+4) % 8]，w1/w2 按 (η+间距索引) 奇偶交替
- 混合表 308 行：仅合并单/双标签，不另生成图片

输出
----
outputs/testbatch_<日期>/
├── pom_images/                         # 308 张 300×300 单通道灰度 PNG
├── labels/labels_testbatch_single.csv  # 88 行
├── labels/labels_testbatch_double.csv  # 220 行
├── labels/labels_testbatch_mixed.csv   # 308 行
├── generation_log.txt                  # 生成方法、参数、种子、耗时
└── manifest.json                       # 逐文件 sha256 + 去重/规格校验结果

运行
----
    cd simulation-dataset/src
    python generate_testbatch.py                 # 全量（约 25 分钟）
    python generate_testbatch.py --limit 4       # 冒烟：只生成 4 张
    python generate_testbatch.py --verify-only   # 只做规格/去重校验

**本批次只用于测试**：不要并入任何训练集或划分表，也不要用于挑模型/调参。
"""
import argparse
import csv
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from config import L, Lz, VOLTAGE_MIN, VOLTAGE_MAX
from ansatz import (generate_single_toron, generate_double_toron,
                    apply_voltage_perturbation)
from frank_osseen import compute_free_energy
from pom_imaging import compute_pom_image

ROOT = Path(__file__).resolve().parents[1]          # simulation-dataset/
DATASET_DIR = ROOT / "outputs" / "dataset" / "pom_images"
SUPPLEMENT_DIR = ROOT / "outputs" / "supplement" / "pom_images"

# === 本批次 ansatz 变体（均与既有 R/B 组合不同）===
VARIANTS = [
    {"R": 13.9, "B": 0.48, "vid": "w1"},
    {"R": 14.9, "B": 0.52, "vid": "w2"},
]

# === 随机种子：确定性参数网格，无随机抽样，仅记录便于追溯 ===
SEED = 20261026

FIELDS_SINGLE = ["filename", "eta", "U", "free_energy", "kind", "n_torons", "R", "B"]
FIELDS_DOUBLE = ["filename", "eta", "U", "free_energy", "kind", "n_torons",
                 "separation", "R", "B"]


def voltage_grid(n=8):
    return np.linspace(VOLTAGE_MIN, VOLTAGE_MAX, n)


def pitch_grid():
    return list(range(30, 41))


def separation_grid():
    return list(range(30, 80, 5))


def u_str(u):
    return f"{u:.2f}".replace(".", "p")


def save_png(intensity, path):
    from PIL import Image
    arr = (np.asarray(intensity) * 255.0).clip(0, 255).astype(np.uint8)
    # 二维 uint8 → PIL mode "L"；不传已弃用的 mode 参数（Pillow 13 将移除）
    Image.fromarray(arr).save(path)


def build_single(out_dir, limit=None):
    rows = []
    for eta in tqdm(pitch_grid(), desc="单 toron 测试批"):
        for ui, U in enumerate(voltage_grid()):
            variant = VARIANTS[(eta + ui) % 2]
            fname = f"single_eta{eta:02d}_U{u_str(U)}_{variant['vid']}.png"
            nx, ny, nz = generate_single_toron(eta=eta, R=variant["R"], B=variant["B"])
            nx, ny, nz = apply_voltage_perturbation(nx, ny, nz, U)
            save_png(compute_pom_image(nx, ny, nz), out_dir / fname)
            rows.append({
                "filename": fname, "eta": int(eta), "U": float(U),
                "free_energy": float(compute_free_energy(nx, ny, nz, eta=eta, U=U)),
                "kind": "single", "n_torons": 1,
                "R": variant["R"], "B": variant["B"],
            })
            if limit and len(rows) >= limit:
                return rows
    return rows


def build_double(out_dir, limit=None):
    rows = []
    U_list = voltage_grid()
    for eta in tqdm(pitch_grid(), desc="双 toron 测试批"):
        for si, sep in enumerate(separation_grid()):
            variant = VARIANTS[(eta + si) % 2]
            for U in (U_list[si % 8], U_list[(si + 4) % 8]):
                fname = (f"double_eta{eta:02d}_U{u_str(U)}"
                         f"_sep{int(sep):02d}_{variant['vid']}.png")
                nx, ny, nz, _ = generate_double_toron(
                    separation=sep, eta=eta, R=variant["R"], B=variant["B"])
                nx, ny, nz = apply_voltage_perturbation(nx, ny, nz, U)
                save_png(compute_pom_image(nx, ny, nz), out_dir / fname)
                rows.append({
                    "filename": fname, "eta": int(eta), "U": float(U),
                    "free_energy": float(compute_free_energy(nx, ny, nz, eta=eta, U=U)),
                    "kind": "double", "n_torons": 2, "separation": float(sep),
                    "R": variant["R"], "B": variant["B"],
                })
                if limit and len(rows) >= limit:
                    return rows
    return rows


def write_csv(rows, fields, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def md5_of(path):
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def existing_hashes():
    """既有数据（原 968+6 demo 与补充 440）的 MD5 集合。"""
    hashes, names = set(), set()
    for directory in (DATASET_DIR, SUPPLEMENT_DIR):
        if not directory.is_dir():
            continue
        for path in directory.glob("*.png"):
            names.add(path.name)
            hashes.add(md5_of(path))
    return hashes, names


def verify(pom_dir, label_rows, verbose=True):
    """规格 + 去重校验，返回可直接写进 manifest 的字典。"""
    from PIL import Image
    existing_md5, existing_names = existing_hashes()
    bad_size, bad_mode, bad_names, dup_content, dup_internal = [], [], [], [], []
    seen = {}
    for row in label_rows:
        path = pom_dir / row["filename"]
        if not path.is_file():
            bad_names.append(row["filename"])
            continue
        with Image.open(path) as img:
            if img.size != (L, L):
                bad_size.append((row["filename"], img.size))
            if img.mode != "L":
                bad_mode.append((row["filename"], img.mode))
        digest = md5_of(path)
        if digest in existing_md5:
            dup_content.append(row["filename"])
        if digest in seen:
            dup_internal.append((row["filename"], seen[digest]))
        seen[digest] = row["filename"]
        if row["filename"] in existing_names:
            bad_names.append(row["filename"])

    result = {
        "n_rows": len(label_rows),
        "n_files": len(list(pom_dir.glob("*.png"))),
        "filename_collisions": bad_names,
        "bad_size": bad_size,
        "bad_mode": bad_mode,
        "content_duplicates_vs_existing": dup_content,
        "content_duplicates_within_batch": dup_internal,
        "all_300x300_L": not bad_size and not bad_mode,
        "zero_overlap_with_existing": not bad_names and not dup_content and not dup_internal,
    }
    if verbose:
        print(f"  行数 {result['n_rows']} / 文件 {result['n_files']}")
        print(f"  文件重名: {len(bad_names)}  尺寸异常: {len(bad_size)}  非灰度: {len(bad_mode)}")
        print(f"  与既有 1,414 张内容重复: {len(dup_content)}  批内重复: {len(dup_internal)}")
        print(f"  全部 300×300 灰度: {result['all_300x300_L']}   与既有数据零重叠: {result['zero_overlap_with_existing']}")
    return result


def coverage(label_rows):
    """按 kind 统计参数覆盖，供交接清单直接引用。"""
    out = {}
    for kind in ("single", "double"):
        rows = [r for r in label_rows if r["kind"] == kind]
        if not rows:
            continue
        out[kind] = {
            "n": len(rows),
            "eta": sorted({int(r["eta"]) for r in rows}),
            "U": sorted({float(r["U"]) for r in rows}),
            "separation": sorted({float(r["separation"]) for r in rows}) if kind == "double" else None,
            "R_B": sorted({(float(r["R"]), float(r["B"])) for r in rows}),
            "free_energy": [min(float(r["free_energy"]) for r in rows),
                            max(float(r["free_energy"]) for r in rows)],
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="输出目录，默认 outputs/testbatch_<日期>")
    ap.add_argument("--limit", type=int, default=None, help="调试：每种类型最多生成多少张")
    ap.add_argument("--verify-only", action="store_true", help="只校验已有目录，不生成")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else ROOT / "outputs" / f"testbatch_{SEED}"
    pom_dir, label_dir = out_dir / "pom_images", out_dir / "labels"

    if not args.verify_only:
        pom_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        print(f"输出目录: {out_dir}")
        print(f"变体: {[(v['vid'], v['R'], v['B']) for v in VARIANTS]}")

        single = build_single(pom_dir, args.limit)
        write_csv(single, FIELDS_SINGLE, label_dir / "labels_testbatch_single.csv")
        print(f"单 toron: {len(single)} 张")

        double = build_double(pom_dir, args.limit)
        write_csv(double, FIELDS_DOUBLE, label_dir / "labels_testbatch_double.csv")
        print(f"双 toron: {len(double)} 张")

        mixed = sorted(single + double, key=lambda r: r["filename"])
        write_csv(mixed, FIELDS_DOUBLE, label_dir / "labels_testbatch_mixed.csv")
        print(f"混合表: {len(mixed)} 行")

        elapsed = time.time() - t0
    else:
        if not label_dir.is_dir():
            raise SystemExit(f"目录不存在: {out_dir}")
        with (label_dir / "labels_testbatch_mixed.csv").open(encoding="utf-8", newline="") as fh:
            mixed = list(csv.DictReader(fh))
        for row in mixed:
            row["eta"], row["U"] = int(row["eta"]), float(row["U"])
            row["R"], row["B"] = float(row["R"]), float(row["B"])
            if row.get("separation"):
                row["separation"] = float(row["separation"])
        elapsed = None

    print("\n=== 校验 ===")
    checks = verify(pom_dir, mixed)
    cov = coverage(mixed)

    # --verify-only 不应覆盖生成时写下的时间戳/耗时：优先沿用已有 manifest
    previous = {}
    manifest_path = out_dir / "manifest.json"
    if args.verify_only and manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))

    manifest = {
        "batch": out_dir.name,
        "created": previous.get("created") or datetime.now().isoformat(timespec="seconds"),
        "generator": "simulation-dataset/src/generate_testbatch.py",
        "seed_recorded": SEED,
        "random_sampling": False,
        "variants": VARIANTS,
        "method": ("论文 Eq.(3)-(6) 解析 ansatz + 手性扭转 q0(z-Cz) 注入 + "
                   "经验电压扰动 |nz|(1-0.15(U/3.5)^2) + Frank-Oseen 体积积分 (Eq.1) + "
                   "Jones 矩阵正交偏振 POM (500nm, dn=0.06); 未运行 PDE 松弛"),
        "usage": "仅供成员 B 测试；不得并入训练集/划分表，不得用于模型选择",
        "counts": {"single": len(cov.get('single', {}).get('eta', [])) * 8 if 'single' in cov else 0,
                   "double": len([r for r in mixed if r['kind'] == 'double']),
                   "mixed": len(mixed)},
        "coverage": cov,
        "checks": checks,
        "elapsed_seconds": previous.get("elapsed_seconds") if elapsed is None else round(elapsed, 1),
        "file_sha256": {row["filename"]: hashlib.sha256(
            (pom_dir / row["filename"]).read_bytes()).hexdigest() for row in mixed},
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.verify_only:
        log = out_dir / "generation_log.txt"
        with log.open("w", encoding="utf-8") as fh:
            fh.write("成员 B 测试批次生成日志\n" + "=" * 46 + "\n")
            fh.write(f"生成时间: {manifest['created']}\n")
            fh.write(f"随机种子（确定性网格，无抽样）: {SEED}\n")
            fh.write(f"总耗时: {elapsed:.1f} s\n\n")
            fh.write("方法: " + manifest["method"] + "\n")
            fh.write(f"原数据集 ansatz: R=14.4, B=0.50；补充集: v1/v2/v3\n")
            for v in VARIANTS:
                fh.write(f"本批次 {v['vid']}: R={v['R']}, B={v['B']}\n")
            fh.write(f"\n单 toron: {len(single)} 张 (11 eta x 8 U)\n")
            fh.write(f"双 toron: {len(double)} 张 (11 eta x 10 sep x 2 U)\n")
            fh.write(f"合计: {len(mixed)} 张\n")
        print(f"\n生成日志: {log}")

    print(f"清单: {out_dir / 'manifest.json'}")
    return 0 if checks["all_300x300_L"] and checks["zero_overlap_with_existing"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
