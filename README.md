# 细粒度食物图像分类系统

## 一、项目介绍

本项目为《人工智能基础》课程 Team Project 题目一 1B 组作业，目标是在 Food-101 数据集上完成亚洲主食类细粒度食物图像分类。与普通图像分类不同，本任务需要区分外观非常接近的菜品，例如 ramen 与 pho、sushi 与 sashimi、dumplings 与 gyoza 等，因此更依赖局部纹理、食材组合、汤底、米饭、面条和摆盘结构等细粒度视觉特征。

项目完成了以下内容：

- 构建 Food-101 1B 亚洲主食 20 类子集；
- 实现数据增强，包括随机裁剪、水平翻转、颜色抖动和小角度旋转；
- 对比三种 CNN 模型：ResNet-18、MobileNet-V2、自定义 Shallow-CNN；
- 使用 classification report 和 confusion matrix 分析模型性能；
- 使用 Grad-CAM 可视化模型关注区域；
- 实现 Streamlit Web Demo，支持上传图片并输出 Top-K 菜品预测结果。

## 二、项目结构

```text
.
├── README.md                         # 项目说明
├── requirements.txt                  # Python 依赖
├── contribution.txt                  # 成员分工说明
├── report.pdf                        # PDF 实验报告
├── 人工智能.docx                     # Word 实验报告
├── MODEL_WEIGHTS_NOTE.md             # 模型权重说明
├── Untitled0.ipynb                   # Google Colab 实验记录
├── training_curves.png               # 训练曲线
├── cheatsheet_1B.md                  # 答辩速查说明
├── code/                             # 源代码
│   ├── train.py                      # 训练脚本
│   ├── evaluate.py                   # 评估脚本
│   ├── predict.py                    # 单图预测脚本
│   ├── grad_cam.py                   # Grad-CAM 可视化脚本
│   ├── app.py                        # Streamlit Web Demo
│   ├── requirements.txt              # 代码目录依赖文件
│   ├── configs/classes_1b_asian.txt  # 1B 组 20 类类别列表
│   └── src/                          # 数据集、模型、变换和工具函数
├── eval/                             # 评估结果
│   ├── resnet18/
│   ├── mobilenet_v2/
│   └── shallow_cnn/
└── gradcam/                          # Grad-CAM 可视化结果
```

## 三、数据集说明

使用 Food-101 数据集，每类包含 1000 张图片，其中训练集 750 张、测试集 250 张。本项目选取 20 个亚洲主食/亚洲风味食物类别：

```text
bibimbap, chicken_curry, dumplings, edamame, falafel,
fried_rice, gyoza, hot_and_sour_soup, hummus, miso_soup,
pad_thai, peking_duck, pho, ramen, samosa,
sashimi, seaweed_salad, spring_rolls, sushi, takoyaki
```

因此本实验子集约包含 15000 张训练图片和 5000 张测试图片。

## 四、环境配置

建议使用 Google Colab T4 GPU 或本地 NVIDIA GPU 环境。Python 版本建议为 3.10 及以上。

### 1. 创建环境

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS：

```bash
source .venv/bin/activate
```

### 2. 安装依赖

在项目根目录运行：

```bash
pip install -r requirements.txt
```

或进入代码目录运行：

```bash
cd code
pip install -r requirements.txt
```

主要依赖包括 PyTorch、torchvision、scikit-learn、matplotlib、seaborn、Pillow、tqdm 和 streamlit。

## 五、运行方法

以下命令默认从项目根目录执行。

### 1. 训练模型

第一次运行时添加 `--download` 下载 Food-101 数据集：

```bash
cd code
python train.py --model resnet18 --download --epochs 10 --batch-size 64 --num-workers 2
```

训练 MobileNet-V2：

```bash
python train.py --model mobilenet_v2 --epochs 10 --batch-size 64 --num-workers 2
```

训练自定义 Shallow-CNN：

```bash
python train.py --model shallow_cnn --epochs 10 --batch-size 64 --num-workers 2
```

如果显存不足，可将 `--batch-size 64` 改为 `--batch-size 32`。

### 2. 评估模型

```bash
python evaluate.py --checkpoint outputs/resnet18/best.pt --num-workers 2
python evaluate.py --checkpoint outputs/mobilenet_v2/best.pt --num-workers 2
python evaluate.py --checkpoint outputs/shallow_cnn/best.pt --num-workers 2
```

评估输出包括：

- `classification_report.json`
- `confusion_matrix.csv`
- `confusion_matrix.png`

本仓库已保留三种模型的评估结果，位于 `eval/` 目录。

### 3. 单图预测

```bash
python predict.py --checkpoint outputs/mobilenet_v2/best.pt --image path/to/image.jpg
```

### 4. Grad-CAM 可视化

```bash
python grad_cam.py --checkpoint outputs/mobilenet_v2/best.pt --image path/to/image.jpg --out outputs/gradcam_result.png
```

本仓库已保留 ramen、pho、sushi、dumplings、takoyaki 五类 Grad-CAM 结果，位于 `gradcam/` 目录。

### 5. Web Demo

```bash
streamlit run app.py
```

打开浏览器后上传食物图片，即可查看 Top-K 菜品预测结果。Google Colab 环境中可使用 localtunnel 或 Colab iframe 暴露 8501 端口。

## 六、实验结果

三种模型在 5000 张测试图像上的结果如下：

| 模型 | Top-1 Accuracy | Macro-F1 | 说明 |
|---|---:|---:|---|
| ResNet-18 | 91.88% | 91.87% | 预训练残差网络，表现稳定 |
| MobileNet-V2 | 92.70% | 92.69% | 本实验最佳模型，兼顾精度与轻量化 |
| Shallow-CNN | 53.52% | 51.70% | 无预训练浅层网络，明显欠拟合 |

实验表明，ImageNet 预训练模型明显优于从零训练的浅层 CNN。MobileNet-V2 在本任务中取得最高准确率，适合作为最终模型和 Web Demo 展示模型。

## 七、结果分析

混淆矩阵显示，模型主要在外观相似类别之间产生误判，例如：

- `dumplings` 与 `gyoza`：二者都具有面皮包馅结构，差异主要来自煎痕和外皮纹理；
- `ramen` 与 `pho`：都属于汤面类，需要区分汤色、面条粗细和配菜；
- `sushi` 与 `sashimi`：都可能包含生鱼片，若米饭区域不明显则容易混淆；
- `miso_soup` 与 `hot_and_sour_soup`：都是汤类，视觉纹理较弱；
- `spring_rolls` 与 `samosa`：都具有油炸外皮，容易受角度和切面影响。

Grad-CAM 结果显示，最佳模型 MobileNet-V2 大多数情况下能够关注食物主体区域，例如面条、汤底、米饭、鱼片、饺子外皮和章鱼烧表面纹理，而不是单纯依赖背景。

## 八、模型权重说明

由于 GitHub 网页上传大文件不稳定，本仓库未上传 `.pt` 模型权重。模型可通过训练脚本复现。仓库中已提供评估结果、混淆矩阵、训练曲线和 Grad-CAM 图片，可以完整展示实验结论。

## 九、报告与复现材料

- PDF 报告：`report.pdf`
- Word 报告：`人工智能.docx`
- Colab 实验记录：`Untitled0.ipynb`
- 成员分工：`contribution.txt`
- 评估结果：`eval/`
- Grad-CAM 结果：`gradcam/`

## 十、结论

本项目完成了 Food-101 1B 亚洲主食细粒度分类任务，实现了三种 CNN 架构对比、数据增强、混淆矩阵分析、Grad-CAM 可视化和 Web Demo。实验结果证明，预训练 CNN 对细粒度食物分类具有明显优势，其中 MobileNet-V2 取得最佳性能。项目代码、实验结果和报告均已整理，可用于课程提交和答辩展示。
