import pytest
import torch
from vision.models.vision_model import MultiTaskVisionModel

def test_model_forward_shape(mocker):
    """Verify model forward pass produces correct logit shapes."""
    # Mock ViTModel to avoid downloading weights during tests
    mock_vit = mocker.patch("vision.models.backbone.ViTModel.from_pretrained")
    mock_instance = mock_vit.return_value
    mock_instance.config.hidden_size = 384
    # Mock the return value of vit(x)
    mock_output = mocker.Mock()
    mock_output.last_hidden_state = torch.randn(1, 1, 384)
    mock_instance.return_value = mock_output
    
    model = MultiTaskVisionModel()
    dummy_input = torch.randn(1, 3, 96, 96)
    outputs = model.forward(dummy_input)
    
    assert outputs["gender_logits"].shape == (1, 2)
    assert outputs["emotion_logits"].shape == (1, 3)

def test_model_predict_structure(mocker):
    """Verify predict method returns expected dictionary structure."""
    mock_vit = mocker.patch("vision.models.backbone.ViTModel.from_pretrained")
    mock_instance = mock_vit.return_value
    mock_instance.config.hidden_size = 384
    mock_output = mocker.Mock()
    mock_output.last_hidden_state = torch.randn(1, 1, 384)
    mock_instance.return_value = mock_output
    
    model = MultiTaskVisionModel()
    dummy_input = torch.randn(1, 3, 96, 96)
    prediction = model.predict(dummy_input)
    
    assert "gender" in prediction
    assert "emotion" in prediction
    assert "gender_conf" in prediction
    assert "emotion_conf" in prediction
    assert prediction["gender"] in ["male", "female"]
    assert prediction["emotion"] in ["sad", "neutral", "happy"]
