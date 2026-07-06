# 细粒度食物图像分类系统

本项目对应《人工智能基础期中 Team Project》题目一的 1B 组：在 Food-101 中选取 20 类亚洲主食/亚洲风味食物，完成细粒度食物图像分类、数据增强、三种 CNN 架构对比、混淆矩阵分析，并提供 Grad-CAM 与 Streamlit Demo 入口。

## 1. 类别子集

类别文件位于 `configs/classes_1b_asian.txt`，共 20 类：

`bibimbap, chicken_curry, dumplings, edamame, falafel, fried_rice, gyoza, hot_and_sour_soup, hummus, miso_soup, pad_thai, peking_duck, pho, ramen, samosa, sashimi, seaweed_salad, spring_rolls, sushi, takoyaki`

Food-101 原始划分为每类 750 张训练图、250 张测试图，因此本子集包含约 15000 张训练图和 5000 张测试图。

## 2. 环境安装

```powershell
cd code
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 3. 训练三种模型

首次运行可添加 `--download` 自动下载 Food-101。若已手动下载，请保证目录结构能被 `torchvision.datasets.Food101` 识别，并去掉 `--download`。

```powershell
python train.py --model resnet18 --download --epochs 10 --batch-size 32
python train.py --model mobilenet_v2 --download --epochs 10 --batch-size 32
python train.py --model shallow_cnn --download --epochs 10 --batch-size 32
```

也可以一键运行：

```powershell
.\run_all_models.ps1
```

输出权重与训练历史保存在：

```text
outputs/resnet18/best.pt
outputs/mobilenet_v2/best.pt
outputs/shallow_cnn/best.pt
```

## 4. 评估与混淆矩阵

```powershell
python evaluate.py --checkpoint outputs/resnet18/best.pt
```

评估会输出：

```text
outputs/eval/resnet18/classification_report.json
outputs/eval/resnet18/confusion_matrix.csv
outputs/eval/resnet18/confusion_matrix.png
```

报告中重点分析容易混淆的类别，例如 `dumplings` 与 `gyoza`，`ramen` 与 `pho`，`sushi` 与 `sashimi`，`miso_soup` 与 `hot_and_sour_soup`。

## 5. 单图预测

```powershell
python predict.py --checkpoint outputs/resnet18/best.pt --image path\to\food.jpg
```

## 6. Grad-CAM 可视化

```powershell
python grad_cam.py --checkpoint outputs/resnet18/best.pt --image path\to\food.jpg --out outputs/grad_cam.png
```

Grad-CAM 用于检查模型是否关注食材主体、汤面、米饭区域、摆盘边界等关键视觉区域，而不是盘子、桌布或背景。

## 7. Web Demo

```powershell
streamlit run app.py
```

打开浏览器后上传图片即可查看 Top-K 菜品预测结果。

## 8. 推荐实验设置

| 模型 | 初始化 | 优化器 | 学习率 | Epoch | 输入尺寸 |
|---|---|---|---:|---:|---:|
| ResNet-18 | ImageNet 预训练 | AdamW | 3e-4 | 10 | 224 |
| MobileNet-V2 | ImageNet 预训练 | AdamW | 3e-4 | 10 | 224 |
| Shallow-CNN | 随机初始化 | AdamW | 3e-4 | 10 | 224 |

数据增强包括随机裁剪、水平翻转、颜色抖动和小角度旋转。细粒度食物分类容易受摆盘、拍摄角度、光照和背景影响，因此增强策略对泛化能力很关键。

