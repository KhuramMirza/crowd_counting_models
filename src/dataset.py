import os
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

class CrowdDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_size=(224, 224), output_size=(112, 112)):
        """
        Args:
            root_dir (str): Directory with images and density maps.
            transform (callable, optional): Optional transform to be applied on images.
            target_size (tuple): The target size for resizing images.
            output_size (tuple): The size to which density maps are resized to match the model output.
        """
        self.root_dir = root_dir
        self.image_paths = sorted(os.listdir(os.path.join(root_dir, "images")))
        self.density_paths = sorted(os.listdir(os.path.join(root_dir, "density_maps")))
        self.transform = transform
        self.target_size = target_size
        self.output_size = output_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, "images", self.image_paths[idx])
        density_path = os.path.join(self.root_dir, "density_maps", self.density_paths[idx])

        # Load image and density map
        image = cv2.imread(img_path)
        density_map = np.load(density_path)

        # Resize image to target size
        image_resized = cv2.resize(image, self.target_size)

        # Resize density map to match model's expected output size (default: 112x112)
        density_map_resized = cv2.resize(density_map, self.output_size)

        # Scale density map to maintain crowd count consistency
        scaling_factor = (density_map.shape[0] / self.output_size[0]) * (density_map.shape[1] / self.output_size[1])
        density_map_resized *= scaling_factor

        # Convert to tensors
        image_tensor = torch.tensor(image_resized, dtype=torch.float32).permute(2, 0, 1) / 255.0  # Normalize image
        density_map_tensor = torch.tensor(density_map_resized, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        return image_tensor, density_map_tensor
