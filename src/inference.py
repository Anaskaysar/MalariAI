import os
import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torchvision.transforms as T
from models.faster_rcnn import get_model_instance_segmentation

def infer_image(image_path, model_path, num_classes=3):
    """
    Load a trained Faster R-CNN model, perform inference on a single image,
    and display the results with bounding boxes.
    """
    if not os.path.exists(image_path):
        print(f"Error: Could not find image at {image_path}")
        return

    if not os.path.exists(model_path):
        print(f"Error: Could not find model at {model_path}. Train the model first.")
        return

    # Initialize model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    model = get_model_instance_segmentation(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load and transform image
    image = Image.open(image_path).convert("RGB")
    image_tensor = T.ToTensor()(image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        predictions = model(image_tensor)

    # Process predictions
    boxes = predictions[0]['boxes'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()

    # Setup plot for visualization
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)

    # Define color scheme for classes (e.g. 1: RBC, 2: non_RBC)
    colors = {1: 'r', 2: 'b', 0: 'g'}
    
    threshold = 0.5  # Confidence threshold

    for box, label, score in zip(boxes, labels, scores):
        if score < threshold:
            continue
            
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        
        color = colors.get(label, 'r')
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        # Add label text (you could map index to string using dataset mappings here)
        plt.text(x_min, y_min, f"Class {label} ({score:.2f})", color=color, 
                 verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.5))

    ax.axis('off')
    
    # Save the output image
    os.makedirs("results", exist_ok=True)
    out_path = f"results/infer_{os.path.basename(image_path)}"
    plt.savefig(out_path, bbox_inches='tight')
    print(f"Saved inference results to {out_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Run inference using the trained Faster R-CNN model.")
    parser.add_argument("--image", type=str, default="ImageMal.jpg", help="Path to input image")
    parser.add_argument("--model", type=str, default="models/faster_rcnn_malariai.pth", help="Path to model weights")
    parser.add_argument("--num-classes", type=int, default=3, help="Number of classes used during training")
    args = parser.parse_args()

    infer_image(args.image, args.model, args.num_classes)

if __name__ == "__main__":
    main()
