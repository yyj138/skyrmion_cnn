# -*- coding: utf-8 -*-
"""
三套 CNN 模型定义 —— 严格复刻论文
Terroa, Tasinkevych & Dias, "Convolutional neural network analysis of optical
texture patterns in liquid-crystal skyrmions", Scientific Reports 15:10921 (2025).

每处结构选择的论文依据（页码为 PDF 页码）：
- 输入 300x300 单通道 POM 强度图：
    "input images (300 x 300 pixels)" (p.3)；POM 为单色光强度图
    "monochromatic incident light with the wavelength 500nm" (p.9, Methods)
- 卷积块 = 两个 4x4 卷积 + 一个 3x3 max-pooling：
    "three blocks of two 4 x 4 convolutions and one 3 x 3 max-pooling layers" (p.3)
- ReLU 用于所有卷积层和全连接层：
    "We have used rectified linear unit (ReLU) activation functions in all
     convolutional and fully connected layers" (p.3)
- 块数以 visualkeras 架构图（由真实 Keras 模型生成，论文 ref.51）为准，
  正文与图注的矛盾见 PAPER_BASIS.md 第 2 节。

论文未给出的参数（滤波器数量、padding、strides、batch size 等）一律在
PAPER_BASIS.md 第 3 节标注为 ASSUMPTION，并可在此处统一修改。
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------------------
# ASSUMPTION（论文未写明，见 PAPER_BASIS.md 3.1）：
# 每个卷积层的滤波器数量。论文正文与架构图均未标注，此处取 32 作为默认值，
# 可通过 build_* 的 conv_filters 参数修改。
# ---------------------------------------------------------------------------
DEFAULT_CONV_FILTERS = 32

# 论文依据（p.3, Fig.2h 混淆矩阵坐标轴）：
# 分类任务的 11 个螺距类别为 eta in {23,24,25,26,28,30,32,34,36,38,40}
PITCH_CLASSES = [23, 24, 25, 26, 28, 30, 32, 34, 36, 38, 40]
NUM_PITCH_CLASSES = len(PITCH_CLASSES)  # 11，对应 "softmax activation function with 11 nodes" (p.3)

INPUT_SHAPE = (300, 300, 1)  # "input images (300 x 300 pixels)" (p.3)，单色 POM 强度 -> 单通道


def _conv_block(x, filters, name):
    """一个卷积块 = 两个 4x4 卷积(ReLU) + 一个 3x3 max-pooling。

    论文依据：
      "blocks of two 4 x 4 convolutions and one 3 x 3 max-pooling layers" (p.3, p.5)
      "ReLU activation functions have been used after all convolution operations" (p.5)

    ASSUMPTION（论文未写明，见 PAPER_BASIS.md 3.2/3.3）：
      padding='same'、strides=1。注：若用 Keras 默认 'valid'，300x300 输入经过
      4 个块后特征图尺寸降为 0（每层 4x4 卷积各减 3、3x3 池化除 3），
      与论文架构图（Fig.2f/3e 均为 4 块）无法共存，故 padding 必为 'same'。
      MaxPooling2D strides 默认等于 pool_size=3（尺寸变化 300->100->34->12->4）。
    """
    x = layers.Conv2D(filters, (4, 4), activation="relu", padding="same", name=f"{name}_conv1")(x)
    x = layers.Conv2D(filters, (4, 4), activation="relu", padding="same", name=f"{name}_conv2")(x)
    x = layers.MaxPooling2D((3, 3), name=f"{name}_maxpool")(x)
    return x


def build_pitch_classifier(conv_filters=DEFAULT_CONV_FILTERS, n_blocks=4):
    """网络 1：螺距分类网（论文 Fig.2f）。

    论文依据：
      "three fully connected layers (with 32, 32, and 16 nodes, respectively)
       and an output layer ... softmax activation function with 11 nodes" (p.3)
      块数 n_blocks=4：以 Fig.2f visualkeras 架构图为准（图中可辨识 4 个
      [Conv,Conv,MaxPool] 块 + Flatten + 4 个 Dense 框，与"3 FC + 输出层"吻合）。
      注意：正文 p.3 写 "three blocks"，与图矛盾，详见 PAPER_BASIS.md 2.1。
    """
    inputs = keras.Input(shape=INPUT_SHAPE, name="pom_image")
    x = inputs
    for i in range(n_blocks):
        x = _conv_block(x, conv_filters, name=f"block{i+1}")
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(32, activation="relu", name="fc1")(x)   # "32" (p.3)
    x = layers.Dense(32, activation="relu", name="fc2")(x)   # "32" (p.3)
    x = layers.Dense(16, activation="relu", name="fc3")(x)   # "16" (p.3)
    outputs = layers.Dense(NUM_PITCH_CLASSES, activation="softmax", name="pitch_out")(x)
    # "the output layer uses a softmax activation function with 11 nodes" (p.3)
    return keras.Model(inputs, outputs, name="pitch_classifier_fig2f")


def build_voltage_regressor(conv_filters=DEFAULT_CONV_FILTERS):
    """网络 2：电压回归网（论文 Fig.3b）。

    论文依据（Fig.3b 图注, p.5，与架构图实测一致）：
      "It is composed of three blocks of two convolutional and one max-pooling
       layers followed by four fully connected layers with 32 nodes each and an
       output layer with a single node and linear activation function."
      "We have trained this network by optimising the mean square error" (p.5)
      注意：正文 p.5 "increased the number of blocks ... by one, reduced the
      number of fully connected layers by one" 与图注/架构图矛盾，详见
      PAPER_BASIS.md 2.2。此处以图注+架构图为准：3 块 + 4 个 FC(32)。
    """
    inputs = keras.Input(shape=INPUT_SHAPE, name="pom_image")
    x = inputs
    for i in range(3):  # "three blocks" (Fig.3b 图注)
        x = _conv_block(x, conv_filters, name=f"block{i+1}")
    x = layers.Flatten(name="flatten")(x)
    for i in range(4):  # "four fully connected layers with 32 nodes each" (Fig.3b 图注)
        x = layers.Dense(32, activation="relu", name=f"fc{i+1}")(x)
    outputs = layers.Dense(1, activation="linear", name="voltage_out")(x)
    # "an output layer with a single node and linear activation function" (Fig.3b 图注)
    return keras.Model(inputs, outputs, name="voltage_regressor_fig3b")


def build_energy_regressor(conv_filters=DEFAULT_CONV_FILTERS):
    """网络 3：自由能回归网（论文 Fig.3e）。

    论文依据（Fig.3e 图注, p.5，与架构图实测一致）：
      "Network with the similar general structure as in Fig. 2f, now with an
       extra block of two convolutional (yellow) and one max-pooling (red)
       layers, a decrease to two fully connected layers with 32 and 16 nodes,
       and the last layer is now composed of 1 node with a linear activation
       function."
      p.6: "four blocks of two 4x4 convolutional layers with a 3x3 max-pooling
       layer, ReLU activation functions after all convolution operations, and
       a linear activation function in the output layer."
    双 toron 自由能与混合数据集的自由能预测复用同一架构：
      "we have use the same CNN architecture as for the case of an isolated
       skyrmion free energy learning (see Fig. 3e)" (p.6)
      "The CNN architecture is dependent on the output parameter as before." (Fig.5a, p.8)
    """
    inputs = keras.Input(shape=INPUT_SHAPE, name="pom_image")
    x = inputs
    for i in range(4):  # "four blocks" (p.6)
        x = _conv_block(x, conv_filters, name=f"block{i+1}")
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(32, activation="relu", name="fc1")(x)   # "32" (Fig.3e 图注)
    x = layers.Dense(16, activation="relu", name="fc2")(x)   # "16" (Fig.3e 图注)
    outputs = layers.Dense(1, activation="linear", name="energy_out")(x)
    # "1 node with a linear activation function" (Fig.3e 图注)
    return keras.Model(inputs, outputs, name="energy_regressor_fig3e")


if __name__ == "__main__":
    # 逐层打印输出形状，用于与论文 Fig.2f/3b/3e 对照自检
    for build in (build_pitch_classifier, build_voltage_regressor, build_energy_regressor):
        model = build()
        model.summary()
        print("-" * 70)
