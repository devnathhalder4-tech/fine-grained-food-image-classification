$ErrorActionPreference = "Stop"

$models = @("resnet18", "mobilenet_v2", "shallow_cnn")
foreach ($model in $models) {
    python train.py --model $model --download --epochs 10 --batch-size 32
    python evaluate.py --checkpoint "outputs/$model/best.pt"
}
