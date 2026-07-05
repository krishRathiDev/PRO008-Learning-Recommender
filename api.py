from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI(title="Learning Recommender API")

with open('svd_model.pkl', 'rb') as f:
    model = pickle.load(f)

rec_df = model['rec_df']
item_avg = model['item_avg']
global_avg = model['global_avg']
user_item = model['user_item']

class PredictRequest(BaseModel):
    user_id: int
    item_id: int

class RecommendRequest(BaseModel):
    user_id: int
    top_n: int = 5

def predict_rating(user_id, item_id):
    if user_id in rec_df.index and item_id in rec_df.columns:
        return float(np.clip(rec_df.loc[user_id, item_id], 1, 5))
    elif item_id in item_avg.index:
        return float(item_avg[item_id])
    else:
        return float(global_avg)

@app.get("/")
def home():
    return {"message": "Learning Recommender API is running!"}

@app.post("/predict")
def predict(req: PredictRequest):
    rating = predict_rating(req.user_id, req.item_id)
    return {
        "user_id": req.user_id,
        "item_id": req.item_id,
        "predicted_rating": round(rating, 4)
    }

@app.post("/recommend")
def recommend(req: RecommendRequest):
    if req.user_id not in rec_df.index:
        return {"error": "User not found"}
    user_ratings = rec_df.loc[req.user_id]
    already_rated = user_item.loc[req.user_id].dropna().index.tolist()
    unrated = user_ratings.drop(labels=already_rated, errors='ignore')
    top_n = unrated.nlargest(req.top_n)
    recommendations = [
        {"item_id": int(iid), "predicted_rating": round(float(score), 4)}
        for iid, score in top_n.items()
    ]
    return {"user_id": req.user_id, "recommendations": recommendations}