import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model_instance_segmentation(num_classes):
    """
    Returns a torchvision Faster R-CNN model loaded with pre-trained ResNet50.
    The predictor head is replaced to accommodate our custom number of classes.
    """
    # Load a model pre-trained on COCO
    # We use fasterrcnn instead of maskrcnn because our dataset provides 
    # only bounding boxes, not polygon masks.
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model
