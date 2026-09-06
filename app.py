import os
import json
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="CineMatch API Server")

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for data and model matrices
df_movies = None
df_ratings = None
tfidf_matrix = None
content_sim = None
collab_sim = None
movie_id_to_idx = {}
idx_to_movie_id = {}

@app.on_event("startup")
def load_data_and_train():
    global df_movies, df_ratings, tfidf_matrix, content_sim, collab_sim, movie_id_to_idx, idx_to_movie_id
    
    # Ensure data directory paths are correct
    movies_path = os.path.join('data', 'movies.csv')
    ratings_path = os.path.join('data', 'ratings.csv')
    
    if not os.path.exists(movies_path) or not os.path.exists(ratings_path):
        raise FileNotFoundError("Data files (movies.csv / ratings.csv) not found. Please run generate_data.py first.")
        
    df_movies = pd.read_csv(movies_path)
    df_ratings = pd.read_csv(ratings_path)
    
    # Build mappings
    movie_id_to_idx = {int(movie_id): idx for idx, movie_id in enumerate(df_movies['id'].values)}
    idx_to_movie_id = {idx: int(movie_id) for idx, movie_id in enumerate(df_movies['id'].values)}
    
    # --- 1. Content-Based TF-IDF ---
    df_movies['metadata_text'] = df_movies['genre'] + " " + df_movies['description']
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_movies['metadata_text'])
    content_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # --- 2. Collaborative Filtering (Item-Item) ---
    # Pivot: rows are movies, columns are users
    user_item_matrix = df_ratings.pivot(index='movie_id', columns='user_id', values='rating')
    # Reindex to make sure all movie IDs are represented
    all_movie_ids = df_movies['id'].values
    user_item_matrix = user_item_matrix.reindex(all_movie_ids, fill_value=0)
    user_item_matrix_filled = user_item_matrix.fillna(0)
    
    # Compute Item Cosine Similarity
    collab_sim = cosine_similarity(user_item_matrix_filled)
    print("Backend engine loaded and precomputed similarity matrices successfully.")

# Input Pydantic model for recommendations
class RecommendRequest(BaseModel):
    seed_id: int
    user_ratings: Dict[str, float]  # Format: {"movie_id": rating_val}

@app.get("/api/movies")
def get_movies():
    """Returns a list of all movies in the database."""
    if df_movies is None:
        raise HTTPException(status_code=500, detail="Data not loaded yet.")
    return df_movies[['id', 'title', 'genre', 'description']].to_dict(orient='records')

@app.post("/api/recommend")
def get_recommendations(req: RecommendRequest):
    """Calculates seed-based and personalized recommendations on-the-fly."""
    if df_movies is None or content_sim is None or collab_sim is None:
        raise HTTPException(status_code=500, detail="Model and data not loaded.")
        
    seed_id = req.seed_id
    if seed_id not in movie_id_to_idx:
        raise HTTPException(status_code=400, detail=f"Seed movie ID {seed_id} not found.")
        
    seed_idx = movie_id_to_idx[seed_id]
    user_ratings_parsed = {int(k): float(v) for k, v in req.user_ratings.items()}
    
    # --- 1. SEED-BASED RECOMMENDATIONS (Like the original static version) ---
    # A. Content-Based recommendations for this seed movie
    cb_scores = list(enumerate(content_sim[seed_idx]))
    cb_scores = sorted(cb_scores, key=lambda x: x[1], reverse=True)
    
    seed_content_recs = []
    for idx, score in cb_scores:
        m_id = idx_to_movie_id[idx]
        if m_id == seed_id:
            continue
        if len(seed_content_recs) >= 5:
            break
        r_movie = df_movies.iloc[idx]
        seed_content_recs.append({
            "id": int(m_id),
            "title": r_movie['title'],
            "genre": r_movie['genre'],
            "score": round(float(score) * 100, 1)
        })
        
    # B. Collaborative recommendations for this seed movie
    seed_collab_recs = []
    collab_scores = list(enumerate(collab_sim[seed_idx]))
    collab_scores = sorted(collab_scores, key=lambda x: x[1], reverse=True)
    
    for idx, score in collab_scores:
        m_id = idx_to_movie_id[idx]
        if m_id == seed_id:
            continue
        if len(seed_collab_recs) >= 5:
            break
        r_movie = df_movies.iloc[idx]
        seed_collab_recs.append({
            "id": int(m_id),
            "title": r_movie['title'],
            "genre": r_movie['genre'],
            "score": round(float(score) * 100, 1)
        })

    # --- 2. PERSONALIZED RECOMMENDATIONS (Based on ALL active user ratings) ---
    # Initialize response structures
    pers_content_recs = []
    pers_collab_recs = []
    
    if user_ratings_parsed:
        # A. Personalized Content-Based via User Profile Vector
        # We build a user profile: sum of movie TF-IDF vectors weighted by centered ratings (rating - 3.0)
        # Ratings centered around 3.0 so 4-5 are positive preferences, 1-2 are negative preferences.
        # If all ratings are 3.0, we just use raw ratings to avoid a zero-vector.
        ratings_mean = np.mean(list(user_ratings_parsed.values()))
        centering_factor = 3.0 if ratings_mean != 3.0 else 0.0
        
        user_profile_vec = np.zeros(tfidf_matrix.shape[1])
        for m_id, rating in user_ratings_parsed.items():
            if m_id in movie_id_to_idx:
                m_idx = movie_id_to_idx[m_id]
                weight = rating - centering_factor
                user_profile_vec += tfidf_matrix[m_idx].toarray()[0] * weight
                
        # If user vector is non-zero, compute cosine similarity with all movie vectors
        norm = np.linalg.norm(user_profile_vec)
        if norm > 0:
            user_profile_vec = user_profile_vec / norm
            # Compute cosine similarity manually
            # movie vectors are normalized in tfidf_matrix
            sims = tfidf_matrix.dot(user_profile_vec)
            pers_cb_scores = list(enumerate(sims))
            pers_cb_scores = sorted(pers_cb_scores, key=lambda x: x[1], reverse=True)
            
            for idx, score in pers_cb_scores:
                m_id = idx_to_movie_id[idx]
                if m_id in user_ratings_parsed:  # Exclude already rated movies
                    continue
                if len(pers_content_recs) >= 5:
                    break
                r_movie = df_movies.iloc[idx]
                pers_content_recs.append({
                    "id": int(m_id),
                    "title": r_movie['title'],
                    "genre": r_movie['genre'],
                    "score": round(max(0.0, float(score)) * 100, 1)
                })
        
        # B. Personalized Collaborative filtering using Item-Item similarity
        # For each unrated movie i, predicted_rating = sum(sim(i, j) * rating(j)) / sum(|sim(i, j)|)
        predicted_ratings = []
        for idx in range(len(df_movies)):
            m_id = idx_to_movie_id[idx]
            if m_id in user_ratings_parsed:  # Exclude already rated movies
                continue
                
            sim_sum = 0.0
            weighted_rating_sum = 0.0
            
            for rated_id, rating in user_ratings_parsed.items():
                if rated_id in movie_id_to_idx:
                    rated_idx = movie_id_to_idx[rated_id]
                    sim_val = collab_sim[idx, rated_idx]
                    
                    # We only consider items with positive similarity to avoid negative correlations distorting rating scale
                    if sim_val > 0:
                        sim_sum += sim_val
                        weighted_rating_sum += sim_val * rating
            
            if sim_sum > 0:
                pred_rating = weighted_rating_sum / sim_sum
                # Normalize predicted rating (1-5) to a match percentage (0-100%)
                # 1 star -> 0% match, 5 stars -> 100% match
                pct_match = ((pred_rating - 1) / 4) * 100
                predicted_ratings.append((idx, pct_match))
                
        predicted_ratings = sorted(predicted_ratings, key=lambda x: x[1], reverse=True)
        for idx, score in predicted_ratings[:5]:
            m_id = idx_to_movie_id[idx]
            r_movie = df_movies.iloc[idx]
            pers_collab_recs.append({
                "id": int(m_id),
                "title": r_movie['title'],
                "genre": r_movie['genre'],
                "score": round(float(score), 1)
            })
            
    # Fallbacks if user has no ratings or calculations yield no results
    if not pers_content_recs:
        pers_content_recs = seed_content_recs
    if not pers_collab_recs:
        pers_collab_recs = seed_collab_recs

    # Return structured payload
    return {
        "seed_movie": {
            "id": seed_id,
            "title": df_movies.iloc[seed_idx]['title'],
            "genre": df_movies.iloc[seed_idx]['genre'],
            "description": df_movies.iloc[seed_idx]['description']
        },
        "seed_based": {
            "content_based": seed_content_recs,
            "collaborative": seed_collab_recs
        },
        "personalized": {
            "content_based": pers_content_recs,
            "collaborative": pers_collab_recs
        }
    }
