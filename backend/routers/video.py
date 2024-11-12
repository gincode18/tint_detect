from fastapi import APIRouter, HTTPException
from bson import ObjectId

from database import videos_collection, images_collection

router = APIRouter()

@router.get("/video/{video_id}")
async def get_video_info(video_id: str, page: int = 1, page_size: int = 10):
    try:
        video_object_id = ObjectId(video_id)
        total_images = images_collection.count_documents({"video_id": video_object_id})
        
        start_idx = (page - 1) * page_size
        car_images_cursor = images_collection.find(
            {"video_id": video_object_id},
            {"_id": 1, "bounding_box": 1, "confidence": 1, "tint_level": 1, "light_quality": 1, "url": 1}
        ).skip(start_idx).limit(page_size)

        car_images = []
        for image in car_images_cursor:
            image["image_id"] = str(image["_id"])
            del image["_id"]
            car_images.append(image)

        total_pages = (total_images + page_size - 1) // page_size
        
        return {
            "video_id": video_id,
            "car_images": car_images,
            "pagination": {
                "current_page": page,
                "page_size": page_size,
                "total_images": total_images,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_previous": page > 1
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/video")
async def list_all_videos():
    try:
        video_ids = videos_collection.find({}, {"_id": 1})
        video_id_list = [str(video["_id"]) for video in video_ids]
        return {"video_ids": video_id_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))