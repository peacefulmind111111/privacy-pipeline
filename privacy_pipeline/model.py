import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus.grad_sample import GradSampleModule
from opacus.validators import ModuleValidator


class BasicBlockEnhanced(nn.Module):
    """ResNet block with ELU and GroupNorm."""

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.gn1 = nn.GroupNorm(8, planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.gn2 = nn.GroupNorm(8, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.GroupNorm(8, planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = F.elu(out)
        out = self.gn1(out)

        out = self.conv2(out)
        out = F.elu(out)
        out = self.gn2(out)

        out = out.clone() + self.shortcut(x)
        return F.elu(out)


class ResNet20Enhanced(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(8, 16)

        self.layer1 = self._make_layer(16, 3, stride=1)
        self.layer2 = self._make_layer(32, 3, stride=2)
        self.layer3 = self._make_layer(64, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlockEnhanced(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = F.elu(out)
        out = self.gn1(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


def build_model(device: torch.device) -> GradSampleModule:
    net = ResNet20Enhanced().to(device)
    errs = ModuleValidator.validate(net, strict=False)
    if errs:
        net = ModuleValidator.fix(net).to(device)
    return GradSampleModule(net)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> float:
    was_training = model.training
    model.eval()
    correct, total = 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        preds = model(X).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    if was_training:
        model.train()
    return 100.0 * correct / total
