# D2NN-with-Pytorch

## Environment:

torch==1.12.1
torchvision==0.13.1
numpy==1.23.5
matplotlib==3.7.1
tqdm==4.65.0

------------------------------------------------

Reading Sequence

D2NN $\Rightarrow$ Beam-Diffraction $\Rightarrow$ D2NN-plus

D2NN-plus = D2NN + Beam-Diffraction

### D2NN

1. Only ***5 Phase Layers*** are concerned.
2. The ***Beam Propagation Evolution*** are not concerned.
3. The rate of training is ***Faster***.
4. The Accuracy is limited to ***70-80%*** for 5 diffractive layers.

### D2NN-plus

1. The free space is ***Totally Meshed***.
2. The ***Beam Propagation Evolution*** are concerned.
3. Only 5 meshed layers are attached with ***Phase Learning***.
4. ***Similar Accuracy*** with D2NN.

The accuracy of D2NN model is troubling me for a long time. It seems difficult to be optimized to 90% accuracy.



---------------

**Congratulations!**

By using 4F system D2NN, I've got **97%** Accuracy for **MNIST** validation dataset.

Now, I'm trying to apply D2NN on **CIFAR-10** dataset with multichannel explorations. ***God Blessing Me !***

## CIFAR-10: ViT-Base vs D2NN-12 (same FC head)

新增脚本：`scripts/compare_vit_d2nn_cifar10.py`。

两个模型都使用同一个全连接分类头结构（`Linear -> ReLU -> Dropout -> Linear`），便于公平对比：

- `ViT-Base + Shared FC`
- `D2NN 12层衍射 + Shared FC`

运行方式：

```bash
python scripts/compare_vit_d2nn_cifar10.py \
  --epochs 5 \
  --batch-size 64 \
  --lr 1e-4
```

结果会写入：`results/cifar10_vit_vs_d2nn.json`。

如果在 Jupyter/云笔记本里运行（会自动注入 `-f xxx.json` 参数），请这样调用：

```python
from scripts.compare_vit_d2nn_cifar10 import main
main()
```

脚本已兼容 notebook 的额外参数，并且当指定 CUDA 但环境不可用时会自动回落到 CPU。

### Notebook 版本（推荐云环境）

新增可直接下载运行的 notebook：`ViT_D2NN_CIFAR10_Compare.ipynb`

特点：

- 模块化分单元调试（数据、模型、训练、评估、可视化分开）
- 每个代码单元带中文注释和执行成功提示
- 训练/评估均包含 tqdm 进度条
- 包含样本展示、训练曲线、混淆矩阵、随机预测 demo
- 兼容云端 Jupyter/Colab 运行方式
