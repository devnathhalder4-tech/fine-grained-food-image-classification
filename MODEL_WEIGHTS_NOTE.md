# 模型权重说明

GitHub 网页上传对大文件不稳定，因此本仓库不包含训练得到的 `.pt` 模型权重。

复现实验可运行：

```bash
cd code
python train.py --model resnet18 --download --epochs 10 --batch-size 64
python train.py --model mobilenet_v2 --epochs 10 --batch-size 64
python train.py --model shallow_cnn --epochs 10 --batch-size 64
```

已保留评估结果、混淆矩阵、训练曲线和 Grad-CAM 图片，足够展示实验结论。
