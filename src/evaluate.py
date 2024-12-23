import torch
from torch.utils.data import DataLoader
from src.dataset import CrowdDataset
from src.model import MCNN
import numpy as np

def evaluate(model_path, data_dir_A, data_dir_B=None, batch_size=4):
    """
    Evaluates the model on the test dataset(s).

    Args:
        model_path (str): Path to the trained model.
        data_dir_A (str): Path to Part_A test dataset.
        data_dir_B (str, optional): Path to Part_B test dataset. Defaults to None.
        batch_size (int): Batch size for evaluation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = MCNN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load test datasets
    test_dataset_A = CrowdDataset(data_dir_A)
    test_loader_A = DataLoader(test_dataset_A, batch_size=batch_size)

    mae_A, mse_A = evaluate_dataset(model, test_loader_A, len(test_dataset_A), device)
    print("Results for Part_A:")
    print(f"Mean Absolute Error (MAE): {mae_A}")
    print(f"Mean Squared Error (MSE): {mse_A}")

    if data_dir_B:
        test_dataset_B = CrowdDataset(data_dir_B)
        test_loader_B = DataLoader(test_dataset_B, batch_size=batch_size)

        mae_B, mse_B = evaluate_dataset(model, test_loader_B, len(test_dataset_B), device)
        print("Results for Part_B:")
        print(f"Mean Absolute Error (MAE): {mae_B}")
        print(f"Mean Squared Error (MSE): {mse_B}")

def evaluate_dataset(model, test_loader, dataset_size, device):
    mae, mse = 0.0, 0.0

    with torch.no_grad():
        for images, density_maps in test_loader:
            images, density_maps = images.to(device), density_maps.to(device)

            # Predict density maps
            outputs = model(images)

            # Calculate errors
            mae += torch.sum(torch.abs(outputs.sum(dim=(2, 3)) - density_maps.sum(dim=(2, 3)))).item()
            mse += torch.sum((outputs.sum(dim=(2, 3)) - density_maps.sum(dim=(2, 3))) ** 2).item()

    mae /= dataset_size
    mse = (mse / dataset_size) ** 0.5

    return mae, mse