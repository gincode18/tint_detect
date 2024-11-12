from pymongo import MongoClient
from config import settings

client = None
db = None
videos_collection = None
images_collection = None

def init_db():
    global client, db, videos_collection, images_collection
    client = MongoClient(settings.MONGO_URI)
    db = client["traffic_db"]
    videos_collection = db["videos"]
    images_collection = db["images"]