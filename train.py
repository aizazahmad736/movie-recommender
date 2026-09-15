import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
df_movies = pd.read_csv('data/movies.csv')
df_ratings = pd.read_csv('data/ratings.csv')

# --- 1. Content-Based Filtering ---
# Combine genre and description to form text feature
df_movies['metadata_text'] = df_movies['genre'] + " " + df_movies['description']

# Apply TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_movies['metadata_text'])

# Compute Content Cosine Similarity
content_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# --- 2. Collaborative Filtering (Item-Item) ---
# Create User-Item interaction matrix (Pivot Table: Movies as rows, Users as columns)
# Note: Item-Item similarity is calculated by analyzing ratings given by users to movies.
user_item_matrix = df_ratings.pivot(index='movie_id', columns='user_id', values='rating')

# Fill NaNs with 0 (representing unrated) or mean-center ratings.
# For Cosine Similarity, centering helps mitigate the "optimism/pessimism" bias of users.
# Here we fill with 0 to calculate standard Cosine Similarity.
user_item_matrix_filled = user_item_matrix.fillna(0)

# Compute Item Cosine Similarity based on user ratings
collab_sim = cosine_similarity(user_item_matrix_filled)

# We map movie_id (which is 1-indexed and might have gaps) to index
movie_id_to_idx = {id: idx for idx, id in enumerate(df_movies['id'].values)}
idx_to_movie_id = {idx: id for idx, id in enumerate(df_movies['id'].values)}

# Output Recommendations Dictionary
output_data = {}

for idx, row in df_movies.iterrows():
    movie_id = int(row['id'])
    
    # 1. Content-Based Recommendations
    content_scores = list(enumerate(content_sim[idx]))
    # Sort by similarity, exclude self (index `idx`)
    content_scores = sorted(content_scores, key=lambda x: x[1], reverse=True)
    content_recs = []
    for r_idx, score in content_scores:
        if r_idx == idx:
            continue
        if len(content_recs) >= 5:
            break
        r_movie = df_movies.iloc[r_idx]
        content_recs.append({
            "id": int(r_movie['id']),
            "title": r_movie['title'],
            "genre": r_movie['genre'],
            "score": round(float(score) * 100, 1) # convert to percentage
        })

    # 2. Collaborative Filtering Recommendations
    # Find the row in collab_sim corresponding to the movie_id
    collab_recs = []
    # If the movie has ratings and exists in the collaborative similarity matrix
    if movie_id in user_item_matrix.index:
        matrix_idx = list(user_item_matrix.index).index(movie_id)
        collab_scores = list(enumerate(collab_sim[matrix_idx]))
        collab_scores = sorted(collab_scores, key=lambda x: x[1], reverse=True)
        
        for sim_idx, score in collab_scores:
            rec_movie_id = int(user_item_matrix.index[sim_idx])
            if rec_movie_id == movie_id:
                continue
            if len(collab_recs) >= 5:
                break
            # Retrieve movie info
            r_movie = df_movies[df_movies['id'] == rec_movie_id].iloc[0]
            collab_recs.append({
                "id": int(rec_movie_id),
                "title": r_movie['title'],
                "genre": r_movie['genre'],
                "score": round(float(score) * 100, 1) # convert to percentage
            })
    else:
        # Fallback to general popular movies if no collaborative ratings exist
        collab_recs = content_recs

    # 3. Hybrid Filtering Recommendations (50% Content + 50% Collaborative)
    hybrid_candidates = []
    for r_idx in range(len(df_movies)):
        if r_idx == idx:
            continue
        c_score = max(0.0, float(content_sim[idx, r_idx]))
        col_score = max(0.0, float(collab_sim[idx, r_idx]))
        h_score = (0.5 * c_score + 0.5 * col_score) * 100
        hybrid_candidates.append((r_idx, h_score))

    hybrid_candidates = sorted(hybrid_candidates, key=lambda x: x[1], reverse=True)
    hybrid_recs = []
    for r_idx, score in hybrid_candidates[:5]:
        r_movie = df_movies.iloc[r_idx]
        hybrid_recs.append({
            "id": int(r_movie['id']),
            "title": r_movie['title'],
            "genre": r_movie['genre'],
            "score": round(score, 1)
        })

    # Store in output database
    output_data[str(movie_id)] = {
        "id": movie_id,
        "title": row['title'],
        "genre": row['genre'],
        "description": row['description'],
        "content_based": content_recs,
        "collaborative": collab_recs,
        "hybrid": hybrid_recs
    }

# Save database to JSON file
with open('data/recommendations.json', 'w') as f:
    json.dump(output_data, f, indent=4)

print("Training finished! Recommendations exported to data/recommendations.json")
