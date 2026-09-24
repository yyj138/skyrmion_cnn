# -*- coding: utf-8 -*-
"""
数据读取、划分与在线增强 —— 严格按论文规则，并支持成员 A 交付的划分表。

论文依据：
1. 输入图像与标签
   - 图像为 300x300 单色 POM 强度图（p.3, p.9 Methods）。
   - 均匀指向构型（eta<30）与初始瞬态构型剔除出训练集：
     "POM textures corresponding to the uniform director configurations were
      excluded from the training dataset, likewise the textures of initial
      transient configurations." (p.3)  —— 剔除在成员A产数据侧完成，此处仅读取。
2. 数据增强（仅旋转 + 翻转，训练时在线进行）：
     "Training was performed with 100 original POM images and its quantity was
      increased through rotation and inversion transformations." (Fig.2/3 图注, p.4-5)
     双 toron 任务 1400 张原图（Fig.4 图注），混合任务 1500 张原图（Fig.5 图注）。
3. 数据集划分：
   - 分类: "separated 15% of the data for a final evaluation (test set) ...
            remaining 85% as a training set" (p.3)
   - 单 toron 自由能回归: "80% of the data for training, 15% for validation,
            and 5% for final testing" (p.6)
   - 混合数据集: "10% of data has been separated as a test set, and the
            remaining is divided into training (85%) and validation (5%) sets"
            (Fig.5b 图注, p.8)

成员 A 数据集的集成约定（与 dataset_readme.md 对齐）：
- 优先使用 A 交付的 splits/split_*.csv 划分表（split 列 train/val/test），
  保证 A/B 两端结果可复现一致；仅当未提供时才由本模块按论文比例自行划分。
- 分类类别取数据集中实际出现的 eta 值（A 的数据为 30-40 共 11 类；论文
  Fig.2h 轴为 {23,24,25,26,28,30,32,34,36,38,40} 也是 11 类——数量一致、
  具体值不同，因 ansatz 在 eta<30 无 toron。输出层 11 节点不变，见
  PAPER_BASIS.md 2.4）。
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

AUTOTUNE = tf.data.AUTOTUNE

# ---------------------------------------------------------------------------
# ASSUMPTION（论文未写明，见 PAPER_BASIS.md 3.4/3.5）：
# - 像素值归一化到 [0,1]（/255）。
# - 旋转角度范围取 0-360 均匀随机（rotation 未给角度）；翻转取水平/垂直
#   （"inversion"）。不引入缩放/裁剪/颜色扰动等任何论文未提及的变换。
# ---------------------------------------------------------------------------


def load_image(path):
    """读取单张 POM 图为 float32 灰度张量 (300,300,1)，并做每图 z-score 标准化。

    ASSUMPTION（论文未写明，见 PAPER_BASIS.md 3.4）：
    先做 /255 归一化到 [0,1]，再做每图 z-score（减单图均值、除单图标准差）。
    论文未提输入预处理；每图标准化为数值必需——实测成员 A 的稀疏纹理
    （~99% 像素为零）在仅 /255 下 CNN 完全无法学习（test acc 0.00），
    加每图 z-score 后 test acc 达 1.00（2026-09-08 对照实验，300 张图
    单 toron 分类，其余配置完全相同）。
    """
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=1, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)  # ASSUMPTION: /255
    img.set_shape([300, 300, 1])
    mean = tf.reduce_mean(img)
    std = tf.math.reduce_std(img) + 1e-8                 # ASSUMPTION: 每图 z-score
    return (img - mean) / std


def augment(image, label):
    """在线增强：仅随机旋转(90 度倍数) + 水平/垂直翻转。

    论文依据: "increased through rotation and inversion transformations"
    （Fig.2h/3g/4g/5e 图注）。
    ASSUMPTION: 旋转取 tf.image.rot90 的 k in {0,1,2,3}（90 度倍数，保持像素网格
    无插值）；翻转即 inversion。除此之外不做任何其他变换。
    """
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    return image, label


def augment_stateless(image, label, random_value, base_seed=42,
                      augment_probability=1.0):
    """确定性在线增强：跨重跑一致，并在每个 epoch 重新采样。

    修复的问题：augment() 使用 tf.random.uniform / tf.image.random_flip_*
    这类**全局**随机源，却挂在 num_parallel_calls=AUTOTUNE 的多线程 map 上。
    多个线程同时处理不同图片时，谁先消费全局随机数由 CPU 调度决定，因此
    即使 tf.keras.utils.set_random_seed(seed) 固定，**每次运行每张图拿到的
    增强方式仍不同**（实测：固定 seed=42 两次取 batch，最大逐像素差 15.2，
    2.70% 像素不同；仅 shuffle 则完全一致，定位为 augment 单独致因）。

    修复方式：上游使用 tf.data.Dataset.random 产生可复现、逐次迭代更新的随机值，
    本函数再用 stateless 随机算子派生旋转和翻转。随机值在进入并行 map 前已经
    确定，因此不受线程调度影响；Dataset.random 的 rerandomize_each_iteration=True
    又保证每个 epoch 使用不同但可复现的增强序列。

    按实施方案使用 0、90、180 度旋转，并可叠加水平/垂直翻转；不进行缩放、
    裁剪或颜色变换。
    """
    seed_pair = tf.stack([
        tf.cast(base_seed, tf.int64),
        tf.cast(random_value, tf.int64),
    ])
    seeds = tf.random.experimental.stateless_split(seed_pair, num=4)

    def transform():
        transformed = tf.image.rot90(
            image,
            tf.random.stateless_uniform([], seed=seeds[0], maxval=3,
                                        dtype=tf.int32),
        )
        flip_lr = tf.random.stateless_uniform([], seed=seeds[1]) < 0.5
        flip_ud = tf.random.stateless_uniform([], seed=seeds[2]) < 0.5
        transformed = tf.cond(
            flip_lr, lambda: tf.image.flip_left_right(transformed),
            lambda: transformed)
        transformed = tf.cond(
            flip_ud, lambda: tf.image.flip_up_down(transformed),
            lambda: transformed)
        return transformed

    apply_augmentation = (
        tf.random.stateless_uniform([], seed=seeds[3])
        < tf.cast(augment_probability, tf.float32)
    )
    image = tf.cond(apply_augmentation, transform, lambda: image)
    return image, label


def augment_stateless_fixed(image, label, idx, base_seed=42):
    """历史成功运行使用的固定确定性增强。

    每个样本由固定索引派生一种 0/90/180/270 度旋转与水平/垂直翻转；同一
    样本跨 epoch 不变。该模式用于精确复核 mixed_pitch_v2，不作为默认正式路径。
    """
    s = tf.constant(base_seed, dtype=tf.int32)
    k = tf.random.stateless_uniform([], seed=[s, idx], maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    flip_lr = tf.random.stateless_uniform([], seed=[s, idx + 1000003]) < 0.5
    flip_ud = tf.random.stateless_uniform([], seed=[s, idx + 2000003]) < 0.5
    image = tf.cond(flip_lr, lambda: tf.image.flip_left_right(image), lambda: image)
    image = tf.cond(flip_ud, lambda: tf.image.flip_up_down(image), lambda: image)
    return image, label


def get_pitch_classes(df):
    """从数据中取分类类别（排序去重），并校验类别数 == 11。

    论文依据: 输出层 "softmax activation function with 11 nodes" (p.3)。
    数据集实际类别值以成员 A 交付为准（30-40 共 11 类，因 ansatz 在
    eta<30 不产生 toron；论文 Fig.2h 轴为另一组 11 类值，见 PAPER_BASIS 2.4）。
    """
    classes = sorted(int(v) for v in df["eta"].unique())
    if len(classes) != 11:
        raise ValueError(f"分类类别数应为 11（论文 p.3），实际 {len(classes)}: {classes}")
    return classes


def _make_labels(df, task):
    """按任务生成标签数组。

    task="pitch":   eta 映射到类别索引（类别取数据集实际值，见 get_pitch_classes）
    task="voltage": U 连续值（"regression problem ... continuous variable
                    representing the applied voltage U", p.3）
    task="energy":  free_energy 连续值
    """
    if task == "pitch":
        classes = get_pitch_classes(df)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        labels = df["eta"].map(class_to_idx).to_numpy(dtype=np.int32)
        if np.isnan(labels.astype(float)).any():
            raise ValueError("存在无法映射到类别索引的 eta 值")
        return labels
    if task == "voltage":
        return df["U"].to_numpy(dtype=np.float32).reshape(-1, 1)
    if task == "energy":
        return df["free_energy"].to_numpy(dtype=np.float32).reshape(-1, 1)
    raise ValueError(f"未知任务: {task}")


def load_dataframe(csv_path, image_dir):
    """读取 CSV，校验图片存在，返回 (图片路径数组, DataFrame)。"""
    df = pd.read_csv(csv_path)
    required = {"filename", "eta", "U", "free_energy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少必需列 {missing}，接口约定见 PAPER_BASIS.md 第 4 节")
    image_dir = Path(image_dir)
    paths = [str(image_dir / f) for f in df["filename"]]
    for p in paths:
        if not Path(p).exists():
            raise FileNotFoundError(f"CSV 引用的图片不存在: {p}")
    return np.array(paths), df


def split_from_csv(df, split_csv, seed=42):
    """读取成员 A 的划分表，返回 (train_idx, val_idx, test_idx)。

    split 表需含 filename 与 split 两列（A 的 dataset_readme.md 约定）。
    若表中无 val 行（分类任务按论文只划分 train/test 85/15），则从 train
    行中确定性地划出 15% 作验证集（ASSUMPTION 3.6：论文 Fig.2g 画了
    validation loss 但未规定比例），seed 保证可复现。
    """
    sdf = pd.read_csv(split_csv)
    if not {"filename", "split"}.issubset(sdf.columns):
        raise ValueError(f"划分表缺少 filename/split 列: {split_csv}")
    lookup = dict(zip(sdf["filename"], sdf["split"]))
    splits = {"train": [], "val": [], "test": []}
    for i, fn in enumerate(df["filename"]):
        if fn not in lookup:
            raise ValueError(f"划分表中找不到样本 {fn}，请与成员 A 核对")
        splits[lookup[fn]].append(i)
    tr, va, te = (np.array(splits["train"]), np.array(splits["val"]),
                  np.array(splits["test"]))
    if len(va) == 0 and len(tr) > 0:
        # 兜底：从 train 中划 15% 作 val（见 docstring 的论文依据）
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(tr))
        n_val = max(1, int(round(len(tr) * 0.15)))
        va = tr[perm[:n_val]]
        tr = tr[perm[n_val:]]
    return tr, va, te


def split_indices(n, task, seed=42):
    """按论文规则返回 (train_idx, val_idx, test_idx)（A 未提供划分表时的兜底）。

    - pitch:  85% train / 15% test（"85% as a training set ... 15% test", p.3）。
      ASSUMPTION: 论文 Fig.2g 画了 validation loss 但未给验证集比例，
      此处从训练集再划出 15% 作验证（见 PAPER_BASIS.md 3.6）。
    - voltage/energy: 80% train / 15% val / 5% test（p.6）。
    - mixed: 10% test，剩余按 85%/5% 划分 train/val（Fig.5b 图注）。
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    if task == "pitch":
        n_test = int(round(n * 0.15))
        test_idx = idx[:n_test]
        rest = idx[n_test:]
        n_val = int(round(len(rest) * 0.15))  # ASSUMPTION
        return rest[n_val:], rest[:n_val], test_idx
    if task == "mixed":
        n_test = int(round(n * 0.10))
        test_idx = idx[:n_test]
        rest = idx[n_test:]
        # "the remaining is divided into training (85%) and validation (5%) sets"
        # 按剩余集合的 85%/5% 切分
        n_train = int(round(len(rest) * 0.85))
        n_val = int(round(len(rest) * 0.05))
        return rest[:n_train], rest[n_train:n_train + n_val], test_idx
    # voltage / energy: 80/15/5
    n_train = int(round(n * 0.80))
    n_val = int(round(n * 0.15))
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


def make_dataset(paths, labels, batch_size, training=False,
                 deterministic=False, seed=42, fixed_deterministic=False,
                 augment_probability=1.0):
    """构建 tf.data 流水线；training=True 时做在线增强 + shuffle。

    增强仅旋转+翻转，见 augment() 的论文依据。

    deterministic=True：正式路径。增强序列逐 epoch 变化，但同 seed 重跑一致。
    deterministic=False：仅用于复核旧实验的兼容路径；该路径受多线程全局随机数
        竞争影响，同 seed 重跑也可能不同。
    """
    paths = np.asarray(paths)
    labels = np.asarray(labels)

    if fixed_deterministic:
        idx = np.arange(len(paths), dtype=np.int32)
        ds = tf.data.Dataset.from_tensor_slices((idx, paths, labels))
        if training:
            ds = ds.shuffle(len(paths), seed=seed,
                            reshuffle_each_iteration=True)
        ds = ds.map(lambda i, p, y: (load_image(p), y, i),
                    num_parallel_calls=AUTOTUNE, deterministic=True)
        if training:
            ds = ds.map(
                lambda x, y, i: augment_stateless_fixed(x, y, i, seed),
                num_parallel_calls=AUTOTUNE,
                deterministic=True,
            )
        else:
            ds = ds.map(lambda x, y, i: (x, y))
        return ds.batch(batch_size).prefetch(AUTOTUNE)

    if deterministic:
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if training:
            ds = ds.shuffle(len(paths), seed=seed,
                            reshuffle_each_iteration=True)
        ds = ds.map(lambda p, y: (load_image(p), y),
                    num_parallel_calls=AUTOTUNE, deterministic=True)
        if training:
            random_values = tf.data.Dataset.random(
                seed=seed, rerandomize_each_iteration=True)
            ds = tf.data.Dataset.zip((ds, random_values))
            ds = ds.map(
                lambda sample, random_value: augment_stateless(
                    sample[0], sample[1], random_value, seed,
                    augment_probability),
                num_parallel_calls=AUTOTUNE,
                deterministic=True,
            )
        return ds.batch(batch_size).prefetch(AUTOTUNE)

    # ---- 默认路径：与原实现逐字一致 ----
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: (load_image(p), y), num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    return ds.batch(batch_size).prefetch(AUTOTUNE)


def build_data(csv_path, image_dir, task, split_task=None, split_csv=None, seed=42):
    """读 CSV -> 划分 -> 返回 (paths, labels, train/val/test 索引, meta)。

    优先使用成员 A 的划分表 split_csv（含 split 列）；未提供时按论文比例
    自行划分（split_task 指定规则，mixed 模式下传 "mixed"）。
    meta 记录类别/划分文件名等信息，供训练与评估两端复现同一划分。
    """
    paths, df = load_dataframe(csv_path, image_dir)
    labels = _make_labels(df, task)
    if split_csv:
        tr, va, te = split_from_csv(df, split_csv)
    else:
        tr, va, te = split_indices(len(paths), split_task or task, seed=seed)
    meta = {
        "n_total": len(paths),
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "split_csv": str(split_csv) if split_csv else None,
        "seed": seed,
        "filenames": list(df["filename"]),
        "train_filenames": [df["filename"].iloc[i] for i in tr],
        "val_filenames": [df["filename"].iloc[i] for i in va],
        "test_filenames": [df["filename"].iloc[i] for i in te],
    }
    if task == "pitch":
        meta["classes"] = get_pitch_classes(df)
    return paths, labels, (tr, va, te), meta
