import os
import torch
from torch.utils.data import DataLoader
from models.dataset import MalariaDataset
from models.faster_rcnn import get_model_instance_segmentation

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    train_csv = "data/processed/train_annotation.csv"
    val_csv = "data/processed/test_annotation.csv"
    
    # Check if data is ready
    if not os.path.exists(train_csv):
        print(f"Error: Could not find '{train_csv}'. Please run 'python data/prepare_data.py' first.")
        return

    # 1. Dataset & DataLoader
    train_dataset = MalariaDataset(csv_file=train_csv, image_dir="./")
    # Number of classes is length of unique labels + background (0)
    num_classes = len(train_dataset.label_to_int)
    
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )

    # 2. Model Initialization
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Training on device: {device}")
    
    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    # 3. Optimizer & Schedulers
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # 4. Training Loop (Simple)
    num_epochs = 5
    
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for images, targets in train_data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
            
        lr_scheduler.step()
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss/len(train_data_loader):.4f}")

    # 5. Save the trained model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/faster_rcnn_malariai.pth")
    print("Model saved to models/faster_rcnn_malariai.pth")

if __name__ == "__main__":
    main()
