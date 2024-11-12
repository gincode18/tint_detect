from fastapi import APIRouter, Request, HTTPException
import os
import numpy as np
from bson import ObjectId

from database import images_collection
from model.model_loader import load_models
from utils.image_processing import preprocess_image

router = APIRouter()

_, _, model_tint = load_models()

LABEL_MAPPING = {
    0: {'tint': 'High', 'light_quality': 'day'},
    1: {'tint': 'Light', 'light_quality': 'day'},
    2: {'tint': 'Light-Medium', 'light_quality': 'day'},
    3: {'tint': 'Medium', 'light_quality': 'day'},
    4: {'tint': 'Medium-High', 'light_quality': 'day'}
}

@router.post("/tint")
async def predict_tint(request: Request):
    try:
        data = await request.json()
        image_id = data['image_id']
        video_id = data['video_id']
        
        image_path = os.path.join("videos", video_id, f"{image_id}_window.png")
        
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")
        
        X_image_test = preprocess_image(image_path)
        prediction = model_tint.predict([X_image_test, np.array([[0, 0]])])
        predicted_class_index = int(np.argmax(prediction, axis=1)[0])
        
        if predicted_class_index not in LABEL_MAPPING:
            raise HTTPException(status_code=500, detail="Predicted class index not in label mapping")

        predicted_attributes = LABEL_MAPPING[predicted_class_index]
        tint_level_numeric = predicted_class_index

        images_collection.update_one(
            {"_id": ObjectId(image_id)},
            {
                "$set": {
                    "tint_level": tint_level_numeric,
                    "light_quality": predicted_attributes["light_quality"]
                }
            }
        )
        
        return {
            "video_id": video_id,
            "image_id": image_id,
            "tint_level": tint_level_numeric,
            "tint_category": predicted_attributes["tint"],
            "light_quality": predicted_attributes["light_quality"]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))