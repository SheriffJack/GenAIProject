from pymongo import MongoClient
import os

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "misinformation_detector")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

users_collection = db["users"]
predictions_collection = db["predictions"]

def add_user(username, email):
    if users_collection.find_one({"email": email}):
        return {"message": "User already exists"}
    users_collection.insert_one({"username": username, "email": email})
    return {"message": "User added successfully"}

def get_user(email):
    return users_collection.find_one({"email": email})

def log_prediction(input_text, label, confidence, user_email=None):
    record = {
        "text": input_text,
        "label": label,
        "confidence": confidence,
        "user_email": user_email,
    }
    predictions_collection.insert_one(record)
    return {"message": "Prediction logged"}

def get_all_predictions(limit=50):
    records = predictions_collection.find().sort("_id", -1).limit(limit)
    return [
        {
            "text": r["text"],
            "label": r["label"],
            "confidence": r["confidence"],
            "user_email": r.get("user_email", "Anonymous"),
        }
        for r in records
    ]

def clear_all_predictions():
    predictions_collection.delete_many({})
    return {"message": "All records cleared"}

def test_connection():
    try:
        client.admin.command("ping")
        print("✅ MongoDB connection successful.")
    except Exception as e:
        print("❌ MongoDB connection failed:", e)

if __name__ == "__main__":
    test_connection()
