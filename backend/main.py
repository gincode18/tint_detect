import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import List
import torch
from pathlib import Path
from pymongo import MongoClient
from bson import ObjectId
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Initialize MongoDB client using the environment variable
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri)  # Ensure `client` is initialized here
db = client["traffic_db"]
videos_collection = db["videos"]
images_collection = db["images"]

# Initialize FastAPI
app = FastAPI()

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.classes = [2]  # Filter for 'car' class only

# Mount the videos directory as static
app.mount("/videos", StaticFiles(directory="videos"), name="videos")

@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...)):
    logger.info("Starting video upload")

    # Log the start time
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

    # Process video and save car images
    car_images = detect_cars_in_video(str(video_path), video_id)

    # Update video document with car images metadata
    videos_collection.update_one(
        {"_id": video_id},
        {"$set": {"car_images": car_images}}
    )
    logger.info(f"Updated MongoDB entry for video {video_id} with car images metadata")

    # Clean up temporary video file if desired
    video_path.unlink()

    # Log the total time taken
    total_time = time.time() - start_time
    logger.info(f"Video processing completed in {total_time:.2f} seconds")

    return JSONResponse({"video_id": str(video_id), "car_images": car_images})


def detect_cars_in_video(video_path: str, video_id) -> List[dict]:
    logger.info("Starting car detection in video")

    cap = cv2.VideoCapture(video_path)
    car_images_metadata = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        for det in results.xyxy[0]:
            if det[5] == 2:  # Check for 'car' class
                x1, y1, x2, y2 = map(int, det[:4])
                car_image = frame[y1:y2, x1:x2]

                # Insert image metadata in MongoDB to get _id
                image_doc = {"video_id": video_id, "bounding_box": [x1, y1, x2, y2]}
                image_id = images_collection.insert_one(image_doc).inserted_id
                logger.info(f"Inserted car image with ID: {image_id} for video {video_id}")

                # Save car image with image_id as the filename
                image_path = Path(f"videos/{video_id}/{image_id}.png")
                cv2.imwrite(str(image_path), car_image)
                logger.info(f"Saved car image at {image_path}")

                # Store image URL for frontend access
                car_images_metadata.append({
                    "image_id": str(image_id),
                    "url": f"http://localhost:3000/videos/{video_id}/{image_id}.png"
                })

    cap.release()
    logger.info("Car detection completed")
    return car_images_metadata

@app.get("/video/{video_id}")
async def get_video_info(video_id: str):
    try:
        # Find the video document by video_id in MongoDB
        video_doc = videos_collection.find_one({"_id": ObjectId(video_id)})
        if not video_doc:
            raise HTTPException(status_code=404, detail="Video not found")

        return {
            "video_id": str(video_id),
            "car_images": video_doc["car_images"]
        }
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
