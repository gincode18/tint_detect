import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi import  Request
from typing import List
import torch
from pathlib import Path
from pymongo import MongoClient
from pymongo import UpdateOne
from bson import ObjectId
from dotenv import load_dotenv
import os
import logging
import time
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import io
import base64
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Load environment variables from .env

load_dotenv()

# Initialize MongoDB client using the environment variable
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri)  # Ensure `client` is initialized here
db = client["traffic_db"]
videos_collection = db["videos"]
images_collection = db["images"]

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI()

# Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# Load YOLOv5 model
model_car = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model_car.classes = [2]  # Filter for 'car' class only
model_car.conf = 0.6  # Set confidence threshold to filter weak detections

# Set up model_windows
# Define the model (U-Net, as provided)
class DoubleConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.ups = torch.nn.ModuleList()
        self.downs = torch.nn.ModuleList()
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                torch.nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = torch.nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = torch.nn.functional.interpolate(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)

# Initialize the model
model = UNET(in_channels=3, out_channels=1).to("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "/Users/vishalkamboj/webdev/tint_detection/backend/models/my_checkpoint.pth.tar"
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the Tint Level Prediction Model
model_path = '/Users/vishalkamboj/webdev/tint_detection/backend/models/tint.h5' 
model = load_model(model_path)

# Define label mapping
label_mapping = {
    0: {'tint': 'High', 'light_quality': 'day'},
    1: {'tint': 'Light', 'light_quality': 'day'},
    2: {'tint': 'Light-Medium', 'light_quality': 'day'},
    3: {'tint': 'Medium', 'light_quality': 'day'},
    4: {'tint': 'Medium-High', 'light_quality': 'day'}
}

# Function to preprocess the image
def preprocess_image(image_path):
    # Load the image, resize it to 224x224 pixels, and normalize pixel values
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


# Mount the videos directory as static
app.mount("/videos", StaticFiles(directory="videos"), name="videos")

@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...)):
    logger.info("Starting video upload")
    start_time = time.time()

    # Insert a new video document to get the MongoDB _id
    video_doc = {"car_images": []}
    video_id = videos_collection.insert_one(video_doc).inserted_id
    logger.info(f"Created MongoDB entry for video with ID: {video_id}")

    # Create directory for storing images of this video
    video_folder = Path(f"videos/{video_id}")
    video_folder.mkdir(parents=True, exist_ok=True)

    # Save uploaded video temporarily
    video_path = video_folder / f"{video_id}.mp4"
    try:
        with video_path.open("wb") as f:
            content = await file.read()
            f.write(content)
        logger.info(f"Uploaded video saved at {video_path}")
    except Exception as e:
        logger.error(f"Error saving uploaded video: {e}")
        raise HTTPException(status_code=500, detail="Error saving video")

    # Process video and save car images in batches of 100
    car_images = detect_cars_in_video(str(video_path), video_id, frame_skip=10)

    # Update video document with car images metadata in MongoDB
    videos_collection.update_one(
        {"_id": video_id},
        {"$set": {"car_images": car_images}}
    )
    logger.info(f"Updated MongoDB entry for video {video_id} with car images metadata")

    # Clean up temporary video file if desired
    video_path.unlink()

    total_time = time.time() - start_time
    logger.info(f"Video processing completed in {total_time:.2f} seconds")

    return JSONResponse({"video_id": str(video_id), "car_images": car_images})


def detect_cars_in_video(video_path: str, video_id, frame_skip: int = 10, batch_size: int = 100) -> List[dict]:
    logger.info("Starting car detection in video")

    cap = cv2.VideoCapture(video_path)
    car_images_metadata = []
    frame_count = 0
    last_saved_box = None
    batched_updates = []

    # Create thumbnail of the first frame
    ret, frame = cap.read()
    if ret:
        thumbnail_path = Path(f"videos/{video_id}/thumbnail.png")
        # Save the original frame as the thumbnail without resizing
        cv2.imwrite(str(thumbnail_path), frame)
        logger.info(f"Saved video thumbnail at {thumbnail_path}")


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process every nth frame
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        results = model_car(frame)
        for det in results.xyxy[0]:
            if det[5] == 2:  # Check for 'car' class
                x1, y1, x2, y2 = map(int, det[:4])
                confidence = det[4].item()

                # Avoid saving duplicate bounding boxes
                if last_saved_box and is_similar_box(last_saved_box, (x1, y1, x2, y2)):
                    continue

                car_image = frame[y1:y2, x1:x2]

                # Prepare image metadata with default tint values
                image_doc = {
                    "video_id": video_id,
                    "bounding_box": [x1, y1, x2, y2],
                    "confidence": confidence,
                    "tint_level": None,  # Default tint level is None
                    "light_quality": None  # Default light quality is None
                }
                image_id = images_collection.insert_one(image_doc).inserted_id
                logger.info(f"Inserted car image with ID: {image_id} for video {video_id}")

                # Save car image to disk
                image_path = Path(f"videos/{video_id}/{image_id}.png")
                cv2.imwrite(str(image_path), car_image)
                logger.info(f"Saved car image at {image_path}")

                # Add image data for batched update
                car_images_metadata.append({
                    "image_id": str(image_id),
                    "url": f"http://localhost:8000/videos/{video_id}/{image_id}.png",
                    "tint_level": None,  # Include default tint level in metadata
                    "light_quality": None  # Include default light quality in metadata
                })

                # Prepare batch for MongoDB update
                batched_updates.append(UpdateOne({"_id": image_id}, {"$set": image_doc}))

                # Update last saved bounding box
                last_saved_box = (x1, y1, x2, y2)

                # Push batch of updates if batch size is met
                if len(batched_updates) >= batch_size:
                    images_collection.bulk_write(batched_updates)
                    logger.info(f"Inserted batch of {len(batched_updates)} images to MongoDB")
                    batched_updates.clear()

        frame_count += 1

    # Write remaining images if any
    if batched_updates:
        images_collection.bulk_write(batched_updates)
        logger.info(f"Inserted final batch of {len(batched_updates)} images to MongoDB")

    cap.release()
    logger.info("Car detection completed")
    return car_images_metadata


def is_similar_box(box1, box2, threshold=0.7) -> bool:
    """Check if two bounding boxes are similar based on IOU (Intersection over Union)."""
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2

    inter_x1, inter_y1 = max(x1, x1_), max(y1, y1_)
    inter_x2, inter_y2 = min(x2, x2_), min(y2, y2_)
    inter_area = max(0, inter_x2 - inter_x1 + 1) * max(0, inter_y2 - inter_y1 + 1)

    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x2_ - x1_ + 1) * (y2_ - y1_ + 1)
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area
    return iou > threshold

@app.get("/video/{video_id}")
async def get_video_info(
    video_id: str,
    page: int = 1,
    page_size: int = 10
):
    try:
        # Define the tint and light quality mappings
        TINT_MAPPING = {
            0: 'High',
            1: 'Light',
            2: 'Light-Medium',
            3: 'Medium',
            4: 'Medium-High'
        }
        
        LIGHT_QUALITY_MAPPING = {
            0: 'day',
            1: 'evening',
            2: 'morning'
        }

        # Find the video document by video_id in MongoDB
        video_doc = videos_collection.find_one({"_id": ObjectId(video_id)})
        if not video_doc:
            raise HTTPException(status_code=404, detail="Video not found")

        # Get the total number of car images
        total_images = len(video_doc["car_images"])
        
        # Calculate pagination parameters
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        # Get the paginated subset of car images
        paginated_images = video_doc["car_images"][start_idx:end_idx]

        # Calculate tint level statistics from all images
        car_images = images_collection.find(
            {"video_id": video_id},
            {"_id": 1, "tint_level": 1, "light_quality": 1}
        )

        # Calculate the average tint level (excluding None values)
        tint_levels = []
        for image in car_images:
            if "tint_level" in image and image["tint_level"] is not None:
                tint_levels.append(image["tint_level"])

        # Calculate average tint level if there are any valid measurements
        if tint_levels:
            avg_tint_level = sum(tint_levels) / len(tint_levels)
            tint_category = TINT_MAPPING[min(TINT_MAPPING.keys(), key=lambda k: abs(k - avg_tint_level))]
        else:
            tint_category = None
            avg_tint_level = None
        
        # Calculate total pages
        total_pages = (total_images + page_size - 1) // page_size
        
        return {
            "video_id": str(video_id),
            "car_images": paginated_images,
            "tint_analysis": {
                "category": tint_category,
                "numeric_average": float(avg_tint_level) if avg_tint_level is not None else None
            },
            "pagination": {
                "current_page": page,
                "page_size": page_size,
                "total_images": total_images,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_previous": page > 1
            }
        }

    except IndexError:
        # Handle the case where page number is out of range
        raise HTTPException(
            status_code=400,
            detail="Page number out of range"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# New endpoint to list all video IDs
@app.get("/video")
async def list_all_videos():
    try:
        # Retrieve all video IDs from the database
        video_ids = videos_collection.find({}, {"_id": 1})
        # Convert ObjectIds to strings
        video_id_list = [str(video["_id"]) for video in video_ids]
        return {"video_ids": video_id_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def crop_largest_rect_area(image_array, threshold=10):
    """
    Crops the largest rectangular area from an image array by removing black portions.
    
    Args:
        image_array: numpy array of the image
        threshold: intensity threshold to determine black regions
    
    Returns:
        numpy array of the cropped image
    """
    # Convert the image to grayscale if it's not already
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array

    # Threshold the image to make black regions black and others white
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:  # If no contours found, return original image
        return image_array
        
    # Find the largest contour (by area)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Crop the image using the bounding box
    cropped_image = image_array[y:y+h, x:x+w]
    
    return cropped_image

@app.post("/windows")
async def detect_windows(request: Request):
    data = await request.json()
    image_id = data['image_id']
    video_id = data['video_id']
    
    # Load the image based on video_id and image_id
    image_path = Path(f"videos/{video_id}/{image_id}.png")

    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    
    input_image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        prediction = torch.sigmoid(model(input_image))
        prediction = (prediction > 0.5).float().cpu()

    # Convert prediction to a numpy array
    prediction_array = prediction.squeeze(0).squeeze(0).numpy()

    # Convert the original image to a numpy array
    original_image_array = np.array(image)

    # Resize the prediction array to match the original image dimensions
    prediction_resized = np.array(Image.fromarray(prediction_array).resize(
        original_image_array.shape[1::-1], 
        Image.NEAREST
    ))

    # Use the resized mask to create a colorized output
    color_mask = np.expand_dims(prediction_resized, axis=-1)
    color_mask = np.repeat(color_mask, 3, axis=-1)

    # Highlight the windshield area
    highlighted_image = np.where(color_mask == 1, original_image_array, 0)

    # Crop the largest rectangular area (removing black portions)
    cropped_image = crop_largest_rect_area(highlighted_image)

    # Calculate the new dimensions after cropping
    new_height, new_width = cropped_image.shape[:2]
    
    # Only resize if the image is too small
    min_size = 800
    if new_width < min_size and new_height < min_size:
        aspect_ratio = new_width / new_height
        if new_width > new_height:
            new_width = min_size
            new_height = int(min_size / aspect_ratio)
        else:
            new_height = min_size
            new_width = int(min_size * aspect_ratio)
            
        cropped_image = cv2.resize(
            cropped_image, 
            (new_width, new_height), 
            interpolation=cv2.INTER_CUBIC
        )

    # Save the output image
    output_path = Path(f"videos/{video_id}/{image_id}_window.png")
    cv2.imwrite(str(output_path), cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))

    # Generate the URL for the output
    full_output_url = f"http://localhost:8000/{output_path}"

    return JSONResponse(content={
        "message": "Window detected and saved",
        "output_path": full_output_url,
        "dimensions": {
            "width": new_width,
            "height": new_height
        }
    })
    data = await request.json()
    image_id = data['image_id']
    video_id = data['video_id']
    
    # Load the image based on video_id and image_id
    image_path = Path(f"videos/{video_id}/{image_id}.png")

    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    
    input_image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        prediction = torch.sigmoid(model(input_image))  # Using the correct model
        prediction = (prediction > 0.5).float().cpu()  # Move to CPU for display

    # Convert prediction to a numpy array
    prediction_array = prediction.squeeze(0).squeeze(0).numpy()

    # Convert the original image to a numpy array
    original_image_array = np.array(image)

    # Resize the prediction array to match the original image dimensions
    prediction_resized = np.array(Image.fromarray(prediction_array).resize(original_image_array.shape[1::-1], Image.NEAREST))

    # Use the resized mask to create a colorized output (apply the prediction mask to the original image)
    color_mask = np.expand_dims(prediction_resized, axis=-1)  # Convert to (H, W, 1)
    color_mask = np.repeat(color_mask, 3, axis=-1)  # Repeat along the color channels

    # Highlight the windshield area by overlaying the mask on the original image
    highlighted_image = np.where(color_mask == 1, original_image_array, 0)

    # Save the output image
    output_path = Path(f"videos/{video_id}/{image_id}_window.png")
    plt.imsave(output_path, highlighted_image)

    # Generate the URL for the output
    full_output_url = f"http://localhost:8000/{output_path}"

    return JSONResponse(content={"message": "Window detected and saved", "output_path": full_output_url})

# Endpoint to predict tint level of a car window image
@app.post("/tint")
async def predict_tint(request: Request):
    try:
        data = await request.json()
        image_id = data['image_id']
        video_id = data['video_id']
        # Load the image based on video_id and image_id
        image_path = os.path.join("videos", video_id, f"{image_id}_window.png")
        
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")
        
        # Preprocess the image
        X_image_test = preprocess_image(image_path)
        
        # Predict the tint level
        prediction = model.predict([X_image_test, np.array([[0, 0]])])  # Assuming no additional attributes
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_attributes = label_mapping.get(predicted_class_index, "Unknown Class")

         # Update the tint level in MongoDB
        result = images_collection.update_one(
            {"_id": ObjectId(image_id)},
            {"$set": {"tint_level": predicted_attributes["tint"]}}
        )
        
        # Return the predicted tint level
        return {"video_id": video_id, "image_id": image_id, "tint_level": predicted_attributes["tint"]}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def health():
        return {
            "health": "we are live",
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
