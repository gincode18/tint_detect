from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
from bson import ObjectId
import cv2
import time

from config import settings
from database import videos_collection, images_collection
from model.model_loader import load_models
from utils.image_processing import is_similar_box

router = APIRouter()
model_car, _, _ = load_models()

@router.post("/upload_video/")
async def upload_video(file: UploadFile = File(...)):
    settings.logger.info("Starting video upload")
    start_time = time.time()

    video_doc = {}
    video_id = videos_collection.insert_one(video_doc).inserted_id
    settings.logger.info(f"Created MongoDB entry with ID: {video_id}")

    video_folder = Path(f"videos/{video_id}")
    video_folder.mkdir(parents=True, exist_ok=True)

    video_path = video_folder / f"{video_id}.mp4"
    try:
        with video_path.open("wb") as f:
            content = await file.read()
            f.write(content)
        settings.logger.info(f"Uploaded video saved at {video_path}")
    except Exception as e:
        settings.logger.error(f"Error saving video: {e}")
        raise HTTPException(status_code=500, detail="Error saving video")

    car_images_metadata = detect_cars_in_video(str(video_path), video_id)
    video_path.unlink()

    total_time = time.time() - start_time
    settings.logger.info(f"Processing completed in {total_time:.2f} seconds")

    return JSONResponse({"video_id": str(video_id), "car_images": car_images_metadata})

def detect_cars_in_video(video_path: str, video_id, frame_skip: int = 10):
    settings.logger.info("Starting car detection")
    cap = cv2.VideoCapture(video_path)
    car_images_metadata = []
    frame_count = 0
    last_saved_box = None

    ret, frame = cap.read()
    if ret:
        thumbnail_path = Path(f"videos/{video_id}/thumbnail.png")
        cv2.imwrite(str(thumbnail_path), frame)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        results = model_car(frame)
        for det in results.xyxy[0]:
            if det[5] == 2:
                x1, y1, x2, y2 = map(int, det[:4])
                confidence = det[4].item()

                if last_saved_box and is_similar_box(last_saved_box, (x1, y1, x2, y2)):
                    continue

                car_image = frame[y1:y2, x1:x2]
                image_id = ObjectId()
                image_url = f"http://localhost:8000/videos/{video_id}/{image_id}.png"
                
                image_doc = {
                    "_id": image_id,
                    "video_id": video_id,
                    "bounding_box": [x1, y1, x2, y2],
                    "confidence": confidence,
                    "url": image_url
                }

                images_collection.insert_one(image_doc)
                image_path = Path(f"videos/{video_id}/{image_id}.png")
                cv2.imwrite(str(image_path), car_image)

                car_images_metadata.append({
                    "image_id": str(image_id),
                    "url": image_url
                })

                last_saved_box = (x1, y1, x2, y2)

        frame_count += 1

    cap.release()
    settings.logger.info("Car detection completed")
    return car_images_metadata