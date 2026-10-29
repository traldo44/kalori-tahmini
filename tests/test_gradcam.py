import timm
import torch

from src.gradcam import _target_layer, gradcam


def test_gradcam_convnext_shape_and_range():
    model = timm.create_model("convnext_tiny", pretrained=False, num_classes=1).eval()
    cam, z = gradcam(model, torch.randn(1, 3, 224, 224), _target_layer(model))
    assert cam.shape == (7, 7)
    assert 0.0 <= cam.min() and cam.max() <= 1.0 + 1e-6
    assert isinstance(z, float)


def test_target_layer_resnet():
    model = timm.create_model("resnet18", pretrained=False, num_classes=1)
    assert _target_layer(model) is model.layer4
