# project1-numpy-student 小项目一

用 NumPy 复现 OpenCLIP ViT-B-32 图像编码器的前向计算。不训练，不用 PyTorch 和 OpenCLIP，输入图片和权重，输出每张图的 512 维特征向量。

## 代码结构

整体流程：图片 → 切成 49 个块 → 加 CLS 和位置编码 → 12 个 Transformer 块 → 取 CLS 行 → 投影到 512 维 → 归一化。

代码都在 `image_encoder.py` 里，按 TODO 编号分：

- TODO-01 `patchify`：224×224 的图切成 7×7 个 32×32 的块，每块按通道—行—列展平成 3072 个数，得到 [B,49,3072]。
- TODO-02 `split_heads`：768 维拆成 12 个头、每头 64 维，形状从 [B,N,768] 变成 [B,12,N,64]。
- TODO-03 注意力计算：Q 乘 K 的转置除以 √64，softmax 后乘 V，12 个头合并再做输出投影。
- TODO-04 Transformer 块：注意力和残差，MLP（768→3072→768，GELU）和第二次残差。
- TODO-05 `embed`：patch 乘投影权重（无偏置），拼上 CLS，加位置编码。
- TODO-06 输出：取 CLS 行，过 `ln_post` 和 `visual.proj` 投影到 512 维，最后做 L2 归一化。

## 依赖清单

见 `requirements.txt`，只有 numpy、Pillow、imageio 三个。

## 数据和权重

本仓库只包含自己写的 `image_encoder.py` 和依赖清单，其余运行文件来自课程附件：`helpers.py`、`preprocess_image.py`、`runtime.py`、`run.py`、`check.py`、`config.json`、`weights/`、`assets/`。

