# project1-numpy-student

小项目一：用纯 NumPy 复现 OpenCLIP ViT-B-32 图像编码器的前向计算。

给定图像和已训练权重，实现图像塔从 `[B,3,224,224]` 到 `[B,512]` 单位特征向量的完整计算，输出与独立的 FP32 参考实现逐阶段对齐。不涉及训练，不使用 PyTorch / OpenCLIP / GPU。

## 仓库内容

| 文件 | 说明 |
| --- | --- |
| `image_encoder.py` | 本次作业的实现，TODO-01 至 TODO-06 全部完成 |
| `requirements.txt` | 依赖清单 |

运行所需的其他文件来自课程发布的 `project1-numpy-student` 附件，包括：

```
helpers.py            LayerNorm、精确 GELU、softmax 等数值助手（课程给定）
preprocess_image.py   图像预处理（课程给定）
runtime.py            权重与图像加载（课程给定）
run.py / check.py     命令行入口与分阶段数值检查（课程给定）
config.json           模型架构参数
weights/              152 个视觉张量 + manifest.json
assets/               十张测试图像与参考输出
```

## 权重与数据获取

`weights/image_encoder.npz` 约 312 MB，超过 GitHub 单文件 100 MB 限制，未包含在本仓库。从课程附件中复制该文件到 `weights/` 目录下即可，`check.py` 会在计算开始前按 `manifest.json` 中的 SHA-256 校验其完整性。`assets/` 中的测试图像与参考输出同理来自课程附件。

## 运行

需要 Python 3.9+，安装依赖后在本目录执行：

```bash
pip install -r requirements.txt

python check.py --stage all     # 八个阶段数值检查
python run.py --output output_tokens.npz
python run.py --images path/to/a.jpg path/to/b.png --output my_tokens.npz
```

## 数值结果

`check.py` 八个阶段（assets / preprocess / patches / tokens / block0 / last_tokens / output / batch）全部 PASS。output 阶段最大绝对误差 5.662e-07，满足 atol=2e-5、rtol=2e-4；内部阶段容差 atol=2e-4。batch 阶段包含乱序与重复样本，验证了实现不依赖批量内顺序、不缓存参考行。

## 实现说明

- 切块：`[B,3,224,224]` 经 `reshape + transpose(0,2,4,1,3,5)` 得到 7×7 个按行优先排列的 patch，每块展平为 3072 维。
- 多头注意力：12 头、每头 64 维，Q/K/V 由合并投影一次算出再等分，`split_heads` 交换后两轴使各头独立计算，分数缩放因子显式转为 float32。
- Transformer 块：预归一化 + 双残差结构，MLP 为 768→3072→768，激活用精确 GELU。
- 输出头：取 CLS 行，过 `ln_post` 后用 `visual.proj`（唯一不转置的线性层）投影到 512 维，L2 归一化为单位向量。
