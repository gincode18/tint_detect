# Video Upload and Car Tint Detection API

This project allows users to upload videos, detect cars windows tint level in the uploaded videos, and store the images of detected cars along with metadata in MongoDB. It also generates a thumbnail of the video and stores it along with the images. All the relevant metadata, including image URLs, is saved in the database.

## Requirements

- Python 3.7+
- MongoDB (Cloud or Local instance)
- OpenCV (`cv2`)
- `pydantic` and `fastapi`
- `uvicorn` for running the FastAPI app

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/yourrepository.git
cd yourrepository
```

### 2. Install dependencies
Make sure you have Python 3.7 or higher installed. Create and activate a virtual environment, then install the required dependencies:

```bash
# Create a virtual environment (if using Python 3.x)
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install the required dependencies
pip install -r requirements.txt
```

The `requirements.txt` file should include:
```
fastapi
uvicorn
opencv-python-headless
pydantic
torch
torchvision
pymongo
python-dotenv
```

### 3. Set up MongoDB
- Ensure you have MongoDB running (either locally or on MongoDB Atlas).
- Update your `.env` file with the MongoDB connection URL.

Create a `.env` file in the root of your project and add the following:

```bash
MONGO_URL="mongodb://<your_mongo_url>"
```

### 4. Run the application
To start the FastAPI application:

```bash
uvicorn main:app --reload
```

This will start the server on `http://127.0.0.1:8000`.

---

## Routes

### 1. **POST /upload_video/**
This endpoint allows you to upload a video for car detection. The video is processed to detect cars, and car images are saved along with metadata.

#### Request:
- **Method**: POST
- **Body**: `multipart/form-data` containing the video file (`file`)
    - Field Name: `file`
    - File Type: `.mp4` (video file)

#### Example:
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/upload_video/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@path_to_your_video.mp4'
```

#### Response:
Returns a JSON response with the `video_id` and a list of `car_images` metadata (image IDs and URLs for the detected car images).

```json
{
  "video_id": "60b8d6d72ef5c742b54c2cb1",
  "car_images": [
    {
      "image_id": "60b8d6d72ef5c742b54c2cbe1",
      "url": "http://localhost:3000/videos/60b8d6d72ef5c742b54c2cb1/60b8d6d72ef5c742b54c2cbe1.png"
    },
    ...
  ]
}
```

---

### 2. **GET /videos**
This endpoint retrieves all the video IDs stored in the database.

#### Request:
- **Method**: GET

#### Example:
```bash
curl -X 'GET' \
  'http://127.0.0.1:8000/videos/'
```

#### Response:
Returns a list of video IDs.

```json
[
  "60b8d6d72ef5c742b54c2cb1",
  "60b8d6d72ef5c742b54c2cb2"
]
```

### 3. **GET /tint/{image_id}**
This endpoint allows you to retrieve the tint level of the car in a particular image identified by the `image_id`. The tint level is determined based on the color of the window (using image processing techniques) and could return a value such as "Low", "Medium", or "High".

#### Request:
- **Method**: GET
- **Parameters**: `image_id` (required) — The MongoDB image ID that identifies the car image.
  
#### Example:
```bash
curl -X 'GET' \
  'http://127.0.0.1:8000/tint/60b8d6d72ef5c742b54c2cbe1'
```

#### Response:
The response returns the tint level of the car's windows in the image, which can be categorized as `"Low"`, `"Medium"`, or `"High"` based on the analysis of the window color.

```json
{
  "image_id": "60b8d6d72ef5c742b54c2cbe1",
  "tint_level": "Medium"
}
```
---

## File Structure

```
/project-root
│
├── main.py                 # FastAPI application logic
├── models.py               # Pydantic models for validation
├── requirements.txt        # Dependencies
├── .env                    # Environment variables (MongoDB URL)
├── videos/                 # Folder where video files and car images are stored
│   ├── <video_id>/
│   │   ├── <image_id>.png  # Detected car images
│   │   ├── thumbnail.png   # Video thumbnail
│   │   └── <other_car_images>.png
│   └── ...
└── logs/                   # Log files (if configured)
```

---

## Logs

Logs are generated using the `logger` configured in the project. Logs will capture key information regarding each step, including video upload, processing, and database updates.

- Logs are saved by default in the `logs` directory.
- You can configure the log level (e.g., `INFO`, `DEBUG`, `ERROR`) and file output by editing the logging configuration in the `main.py` file.

---

## Performance Considerations

- **Batching Database Inserts**: To improve performance, car images are processed and saved in batches. This minimizes the number of database write operations.
- **Temporary Video File Deletion**: After the video is processed and the metadata is saved, the temporary video file is deleted to free up space.

---

## Troubleshooting

1. **"MongoDB connection failed"**:
   - Ensure that your MongoDB instance is running and that the connection URL is correctly set in the `.env` file.
   
2. **"File upload failed"**:
   - Ensure that the file is a valid `.mp4` video file and that it is not too large for your server to handle.

3. **Performance issues**:
   - The video processing step can take a long time for large videos. You can try lowering the `frame_skip` value in the `detect_cars_in_video` function to speed it up, but note this may decrease detection accuracy.

---

## Conclusion

This project provides a simple API for uploading videos, detecting cars within those videos, and storing the detected car images and metadata in MongoDB. It includes a thumbnail of the video and allows easy access to detected car images via URLs.

---

Feel free to adjust any details according to the exact structure of your application, like paths and MongoDB configurations.