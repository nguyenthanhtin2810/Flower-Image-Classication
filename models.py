import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class PretrainResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(in_features=2048, out_features=num_classes)
        for name, pram in self.backbone.named_parameters():
            if "fc" not in name:
                pram.requires_grad = False


    def forward(self, x):
        x = self.backbone(x)
        return x

if __name__ == '__main__':
    model = PretrainResNet50()
    input_data = torch.rand(8, 3, 224, 224)
    if torch.cuda.is_available():
        model.cuda()  # in_place function
        input_data = input_data.cuda()
    while True:
        result = model(input_data)
        print(result.shape)
        break
