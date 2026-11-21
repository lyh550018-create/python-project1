# project1-numpy-student 小项目一

用 NumPy 复现 OpenCLIP ViT-B-32 图像编码器的前向计算。不训练，不用 PyTorch 和 OpenCLIP，输入图片和权重，输出每张图的 512 维特征向量。

## 代码结构

整体流程：图片 → 切成 49 个块 → 加 CLS 和位置编码 → 12 个 Transformer 块 → 取 CLS 行 → 投影到 512 维 → 归一化。

仓库里的文件：

- `image_encoder.py`：本次作业的代码，TODO-01 至 TODO-06 都写完了
- `helpers.py`：课程提供的数值函数（LayerNorm、GELU、softmax 等）
- `preprocess_image.py`：课程提供的预处理（缩放、裁剪、标准化）
- `runtime.py`、`run.py`、`check.py`：课程提供的加载、运行和检查脚本
- `config.json`、`weights/manifest.json`：模型结构和权重的校验信息
- `transformer_steps.py`：课程提供的小型 Transformer 计算演示
- `assets/`：十张测试图片和参考输出
- `lecture-demos/`、`practice/`、`PPT-CODE-MAP.txt`、`SOURCE-MANIFEST.json`：课程附件里的课堂示例和说明

`image_encoder.py` 里按 TODO 编号分：

- TODO-01 patchify：224×224 的图切成 7×7 个 32×32 的块，每块按通道—行—列展平成 3072 个数，得到 [B,49,3072]。
- TODO-02 split_heads：768 维拆成 12 个头、每头 64 维，形状从 [B,N,768] 变成 [B,12,N,64]。
- TODO-03 注意力计算：Q 乘 K 的转置除以 √64，softmax 后乘 V，12 个头合并再做输出投影。
- TODO-04 Transformer 块：注意力和残差，MLP（768→3072→768，GELU）和第二次残差。
- TODO-05 embed：patch 乘投影权重（无偏置），拼上 CLS，加位置编码。
- TODO-06 输出：取 CLS 行，过 ln_post 和 visual.proj 投影到 512 维，最后做 L2 归一化。

## 依赖清单

见 `requirements.txt`，只有 numpy、Pillow、imageio 三个。

## 数据和权重

`weights/image_encoder.npz` 有 312 MB，超过 GitHub 单文件 100 MB 的限制，没有放进仓库。从课程附件复制到 `weights/` 目录下即可，`check.py` 会先按 `manifest.json` 里的 SHA-256 核对这个文件。

`assets/reference_outputs.npz` 已经放在仓库里，`check.py` 用它和计算结果逐阶段比对。

## 运行

在仓库根目录执行：

```bash
python check.py --stage all
python run.py --output output_tokens.npz
```
