import os

import torch
import pytest

from model_utils import initialize_model


@pytest.fixture
def model():
    model_path = r"model\resnet50_retinamnist.pth"
    assert os.path.exists(model_path), f"Model file not found at {model_path}"
    return initialize_model()


def test_model_prediction_shape(model):
    dummy_input = torch.randn(1, 3, 224, 224)  # One image with 3 channels, 224x224
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape[1] == 5  # RetinaMNIST has 5 classes


def test_model_prediction_values(model):
    dummy_input = torch.randn(1, 3, 224, 224)  # One image with 3 channels, 224x224
    with torch.no_grad():
        output = model(dummy_input)
    predicted_class = torch.argmax(output, dim=1).item()
    assert predicted_class in range(5)  # Ensure prediction is one of the 5 classes


def test_model_predicts_confidence(model):
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    # Check if outputs sum close to 1 after softmax
    probs = torch.nn.functional.softmax(output, dim=1)
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-3)
