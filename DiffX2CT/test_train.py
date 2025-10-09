import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import patch, MagicMock

# Assuming train.py is in the same directory or accessible via PYTHONPATH
from DiffX2CT.train import evaluate_epoch

class MockEMA:
    def apply_shadow(self):
        pass
    def restore(self):
        pass

def test_evaluate_epoch_uses_mse_loss():
    """
    Test that evaluate_epoch uses F.mse_loss instead of F.l1_loss.
    """
    # 1. Setup Mocks
    device = torch.device("cpu")

    # Mock model
    mock_model = MagicMock(spec=nn.Module)
    mock_model.ema = MockEMA()
    mock_model.eval = MagicMock()
    # Mock the conditioning_encoder attribute
    mock_model.conditioning_encoder = MagicMock(return_value=torch.randn(1, 256))
    # Make the model callable to simulate forward pass
    mock_model.return_value = torch.randn(1, 1, 16, 16, 16)

    # Mock scheduler
    mock_scheduler = MagicMock()
    mock_scheduler.config.num_train_timesteps = 1000
    mock_scheduler.add_noise.return_value = torch.randn(1, 1, 16, 16, 16)

    # Mock dataloader
    dummy_data = [
        (torch.randn(1, 1, 16, 16, 16),
         torch.randn(1, 3, 128, 128),
         torch.randn(1, 3, 128, 128),
         torch.randn(1, 3, 16, 16, 16))
    ]
    mock_dataloader = dummy_data

    # 2. Patch the loss functions
    with patch('torch.nn.functional.mse_loss', wraps=F.mse_loss) as mock_mse_loss, \
         patch('torch.nn.functional.l1_loss', wraps=F.l1_loss) as mock_l1_loss:

        # 3. Call the function to be tested
        evaluate_epoch(device, mock_model, mock_scheduler, mock_dataloader)

        # 4. Assertions
        mock_l1_loss.assert_not_called()
        mock_mse_loss.assert_called_once()
        print("Test assertion: mse_loss should be called, and l1_loss should not be.")