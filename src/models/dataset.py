import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class MalariaDataset(Dataset):
    def __init__(self, csv_file, image_dir, transforms=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            image_dir (string): Directory with all the images.
            transforms (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.annotations = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transforms = transforms
        
        # Group by image name to easily return all boxes for an image
        self.image_groups = self.annotations.groupby('img_name')
        self.image_names = list(self.image_groups.groups.keys())
        
        # Build category mapping dynamically or use predefined based on knowledge
        unique_labels = self.annotations['label'].unique()
        self.label_to_int = {lbl: i+1 for i, lbl in enumerate(unique_labels)}
        # 0 is always reserved for background
        self.label_to_int['background'] = 0

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        group = self.image_groups.get_group(img_name)
        
        # Load image
        img_path = os.path.join(self.image_dir, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            # Fallback if the path structure is slightly different
            # since img_name might contain 'training_images/foo.jpg'
            img_path = os.path.join(self.image_dir, os.path.basename(img_name))
            image = Image.open(img_path).convert("RGB")
            
        # Get bounding box coordinates and labels
        boxes = []
        labels = []
        for _, row in group.iterrows():
            xmin = row['x_min']
            ymin = row['y_min']
            xmax = row['x_max']
            ymax = row['y_max']
            
            # torchvision requires boxes in [x0, y0, x1, y1] format, where 0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H.
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(self.label_to_int[row['label']])
                
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        
        if self.transforms is not None:
            image = self.transforms(image)
        else:
            image = T.ToTensor()(image)
            
        return image, target
