from dotenv import load_dotenv
import os
import logging

# Load environment variables
load_dotenv()

class Settings:
    MONGO_URI = os.getenv("MONGO_URI")
    MODEL_PATH = '/Users/vishalkamboj/webdev/tint_detection/backend/models/tint.h5'
    CHECKPOINT_PATH = "/Users/vishalkamboj/webdev/tint_detection/backend/models/my_checkpoint.pth.tar"
    
    # Model configurations
    CONFIDENCE_THRESHOLD = 0.6
    FRAME_SKIP = 10
    BATCH_SIZE = 100
    
    # Logging configuration
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

settings = Settings()