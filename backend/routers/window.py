from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch

from model.model_loader import load_models, get_transforms
from utils.image_processing import crop_largest_rect_area

# Load the models once
_, model_window, _ = load_models()
transform = get_transforms()

router = APIRouter()

@router.post("/windows")
async def detect_windows(request: Request):
    try:
        data = await request.json()
        image_id = data['image_id']
        video_id = data['video_id']
        
        image_path = Path(f"videos/{video_id}/{image_id}.png")
        if not image_path.exists():
            raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = torch.sigmoid(model_window(input_image))
            prediction = (prediction > 0.5).float().cpu()

        prediction_array = prediction.squeeze(0).squeeze(0).numpy()
        original_image_array = np.array(image)

        prediction_resized = np.array(Image.fromarray(prediction_array).resize(
            original_image_array.shape[1::-1], 
            Image.NEAREST
        ))

        color_mask = np.expand_dims(prediction_resized, axis=-1)
        color_mask = np.repeat(color_mask, 3, axis=-1)

        highlighted_image = np.where(color_mask == 1, original_image_array, 0)
        cropped_image = crop_largest_rect_area(highlighted_image)

        new_height, new_width = cropped_image.shape[:2]
        
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

        output_path = Path(f"videos/{video_id}/{image_id}_window.png")
        cv2.imwrite(str(output_path), cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))

        full_output_url = f"http://localhost:8000/{output_path}"

        return JSONResponse(content={
            "message": "Window detected and saved",
            "output_path": full_output_url,
            "dimensions": {
                "width": new_width,
                "height": new_height
            }
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))