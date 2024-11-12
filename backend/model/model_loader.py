import torch
import torchvision.transforms as transforms
from tensorflow.keras.models import load_model
from model.unet import UNET
import logging

logger = logging.getLogger(__name__)

def load_models():
    try:
        # Load YOLOv5 model with specific version and trust_repo=True
        model_car = torch.hub.load('ultralytics/yolov5', 'yolov5s', 
                                 pretrained=True, 
                                 trust_repo=True)
        model_car.classes = [2]  # Filter for 'car' class only
        model_car.conf = 0.6  # Set confidence threshold
    except Exception as e:
        logger.error(f"Error loading YOLOv5 model: {e}")
        raise RuntimeError("Failed to load YOLOv5 model")

    try:
        # Load window detection model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_window = UNET(in_channels=3, out_channels=1).to(device)
        
        try:
            checkpoint_path = "/Users/vishalkamboj/webdev/tint_detection/backend/models/my_checkpoint.pth.tar"
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model_window.load_state_dict(checkpoint["state_dict"])
        except Exception as e:
            logger.error(f"Error loading window model checkpoint: {e}")
            raise RuntimeError("Failed to load window model checkpoint")
            
        model_window.eval()
    except Exception as e:
        logger.error(f"Error setting up window detection model: {e}")
        raise RuntimeError("Failed to initialize window detection model")

    try:
        # Load tint detection model
        model_tint = load_model('/Users/vishalkamboj/webdev/tint_detection/backend/models/tint.h5')
    except Exception as e:
        logger.error(f"Error loading tint detection model: {e}")
        raise RuntimeError("Failed to load tint detection model")

    return model_car, model_window, model_tint

def get_transforms():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
    ])