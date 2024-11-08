import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
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

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.classes = [2]  # Filter for 'car' class only
model.conf = 0.6  # Set confidence threshold to filter weak detections

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

        results = model(frame)
        for det in results.xyxy[0]:
            if det[5] == 2:  # Check for 'car' class
                x1, y1, x2, y2 = map(int, det[:4])
                confidence = det[4].item()

                # Avoid saving duplicate bounding boxes
                if last_saved_box and is_similar_box(last_saved_box, (x1, y1, x2, y2)):
                    continue

                car_image = frame[y1:y2, x1:x2]

                # Prepare image metadata
                image_doc = {"video_id": video_id, "bounding_box": [x1, y1, x2, y2], "confidence": confidence}
                image_id = images_collection.insert_one(image_doc).inserted_id
                logger.info(f"Inserted car image with ID: {image_id} for video {video_id}")

                # Save car image to disk
                image_path = Path(f"videos/{video_id}/{image_id}.png")
                cv2.imwrite(str(image_path), car_image)
                logger.info(f"Saved car image at {image_path}")

                # Add image data for batched update
                car_images_metadata.append({
                    "image_id": str(image_id),
                    "url": f"http://localhost:8000/videos/{video_id}/{image_id}.png"
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
        
        # Calculate total pages
        total_pages = (total_images + page_size - 1) // page_size
        
        return {
            "video_id": str(video_id),
            "car_images": paginated_images,
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

@app.get("/")
async def health():
        return {
            "health": "we are live",
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
