# Video Upload and Car Tint Detection API

This project allows users to upload videos, detect cars windows tint level in the uploaded videos, and store the images of detected cars along with metadata in MongoDB. It also generates a thumbnail of the video and stores it along with the images. All the relevant metadata, including image URLs, is saved in the database.

## Requirements

- Python 3.7+
- MongoDB (Cloud or Local instance) or docker
```bash
docker run -d \
  --name mongodb \
  -p 27017:27017 \
  -e MONGO_INITDB_ROOT_USERNAME=mongoadmin \
  -e MONGO_INITDB_ROOT_PASSWORD=secret \
  -v mongodb_data:/data/db \
  mongo:latest
```
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

# Video Upload and Car Tint Detection API

[Previous sections remain the same until the routes...]

## Routes

[Previous routes remain the same...]

### 4. **POST /windows**
This endpoint processes a detected car image to identify and extract the windshield/window area. It performs window detection using a trained UNET model, removes black portions around the detected window, and returns a processed image that highlights just the window area.

#### Request:
- **Method**: POST
- **Content-Type**: application/json
- **Body**:
```json
{
  "video_id": "60b8d6d72ef5c742b54c2cb1",
  "image_id": "60b8d6d72ef5c742b54c2cbe1"
}
```

#### Processing Steps:
1. Loads the specified car image
2. Detects windows using the UNET model
3. Creates a mask highlighting the window area
4. Removes black portions around the detected window
5. Resizes the image if necessary (minimum 800px dimension)
6. Saves the processed image with maintained aspect ratio

#### Example:
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/windows' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "video_id": "60b8d6d72ef5c742b54c2cb1",
    "image_id": "60b8d6d72ef5c742b54c2cbe1"
  }'
```

#### Response:
Returns a JSON response containing the path to the processed image and its dimensions:

```json
{
  "message": "Window detected and saved",
  "output_path": "http://localhost:8000/videos/60b8d6d72ef5c742b54c2cb1/60b8d6d72ef5c742b54c2cbe1_window.png",
  "dimensions": {
    "width": 800,
    "height": 600
  }
}
```

#### Error Responses:
- **404 Not Found**: If the specified image cannot be found
- **500 Internal Server Error**: If there's an error processing the image


## Additional Requirements

The `/windows` route requires additional dependencies:
```
torch
torchvision
numpy
Pillow
opencv-python-headless
```

## Performance Considerations

[Previous sections remain the same, add the following:]

- **Window Detection Processing**: The window detection and image processing can be computationally intensive. Consider implementing caching mechanisms for frequently accessed images.
- **Image Size Optimization**: While the route maintains a minimum size of 800px for quality, you may want to adjust this based on your specific needs and storage constraints.

## Troubleshooting

[Previous sections remain, add the following:]

4. **"Window detection failed"**:
   - Ensure the input image exists in the specified path
   - Verify that the UNET model is properly loaded
   - Check if the image format is supported (PNG/JPEG)

5. **"Black image output"**:
   - This might occur if no windows are detected in the image
   - Verify that the car image is clear and windows are visible
   - Adjust the threshold parameter in the cropping function if needed

### 5. **GET /tint**
This endpoint allows you to retrieve the tint level of the car's window in a particular image identified by the `video_id` and `image_id`.

The tint level is determined based on the color of the window using image processing techniques and a machine learning model. The predicted tint level can be categorized as "Low", "Medium", or "High".

#### Request:
- **Method**: GET
- **Parameters**:
  - `video_id` (required): The MongoDB ID of the video that contains the car image.
  - `image_id` (required): The MongoDB ID of the car image.

#### Example:
```bash
  curl --location 'http://127.0.0.1:8000/tint' \
--header 'Content-Type: application/json' \
--data '{
    "video_id": "6730bf2963ad51a0842d08e5",
    "image_id": "6730bf2963ad51a0842d08e6"
}'
```

#### Response:
The response returns the tint level of the car's window in the image.

```json
{
    "video_id": "6732663fe1d217c9fb0eb622",
    "image_id": "6732663fe1d217c9fb0eb623",
    "tint_level": 3,
    "tint_category": "Medium",
    "light_quality": "day"
}
```

#### Error Responses:
- **404 Not Found**: If the specified `video_id` or `image_id` cannot be found.
- **500 Internal Server Error**: If there's an error processing the image or predicting the tint level.

## Implementation Details

The `/tint` endpoint uses a pre-trained machine learning model to predict the tint level of the car's window in the specified image. The model was trained on a dataset of car window images labeled with their tint levels.

The key steps involved in the `/tint` endpoint are:

1. Load the car image based on the provided `video_id` and `image_id`.
2. Preprocess the image by resizing it to the required input size of the machine learning model.
3. Pass the preprocessed image through the trained model to obtain the predicted tint level.
4. Map the predicted tint level to a human-readable category (e.g., "Low", "Medium", "High").
5. Return the `video_id`, `image_id`, and the predicted `tint_level` in the response.

## Performance Considerations

- **Model Loading and Inference**: Loading the machine learning model and performing the tint level prediction can be computationally intensive, especially for larger images. Consider implementing caching mechanisms to avoid unnecessary model loading and inference for frequently accessed images.
- **Image Storage and Retrieval**: Ensure that the storage and retrieval of car images are efficient, as the `/tint` endpoint will need to access the images for each request. Optimize the file storage and database operations accordingly.

## Troubleshooting

1. **"Image not found"**:
   - Verify that the specified `video_id` and `image_id` are valid and correspond to an existing car image in the system.
   - Check the file system and database to ensure the image is stored correctly.

2. **"Error predicting tint level"**:
   - Ensure that the machine learning model is properly loaded and configured.
   - Check the model's input requirements (e.g., image size, normalization) and verify that the preprocessing is correct.
   - Review the model's performance on a test dataset to identify any issues with the tint level prediction.

3. **"Unexpected tint level"**:
   - If the predicted tint level seems inaccurate, review the model's training data and hyperparameters.
   - Consider fine-tuning or retraining the model if necessary to improve its performance on the specific car window images.

Feel free to expand on these sections as needed to provide comprehensive documentation for the `/tint` API endpoint.
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