from flask import Flask, render_template, request
import pickle
import faiss
import numpy as np
import pandas as pd
import re

app = Flask(__name__)

# Load FAISS index and metadata
with open("models/movies_meta.pkl", "rb") as f:
    movies_data, vectorizer, feature_matrix = pickle.load(f)

index = faiss.read_index("models/movie_index.faiss")
print(movies_data.keys() if hasattr(movies_data, "keys") else movies_data.columns)
title_map = {title.lower(): i for i, title in enumerate(movies_data['title'])}



# Recommendation function
def get_recommendations(movie_name, top_n=10):
    movie_name_lower = movie_name.lower().strip()
    if movie_name_lower not in title_map:
        return [f"No match found for '{movie_name}'"]

    movie_idx = title_map[movie_name_lower]
    query_vector = feature_matrix[movie_idx].reshape(1, -1).astype(np.float32)
    distances, indices = index.search(query_vector, top_n + 10)

    recommendations = []
    seen = set()

    for idx in indices[0]:
        if idx == movie_idx:
            continue

        title = movies_data.iloc[idx]['title']
        title_clean = re.sub(r'\s*\(\d{4}\)$', '', title).strip().title()
        
        rec = {
            'title': title_clean,
            'genres': ', '.join(eval(movies_data.iloc[idx]['genres'])) if pd.notnull(movies_data.iloc[idx]['genres']) else '',
            'rating': movies_data.iloc[idx]['rating'],
            'poster': movies_data.iloc[idx]['poster'],
            'trailer': movies_data.iloc[idx]['trailer']
        }

        if title_clean not in seen:
            recommendations.append(rec)
            seen.add(title_clean)
        if len(recommendations) >= top_n:
            break

    return recommendations

# Home route
@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []
    movie_name = ""
    if request.method == "POST":
        movie_name = request.form["movie"]
        recommendations = get_recommendations(movie_name)
    return render_template("index.html", recommendations=recommendations, movie_name=movie_name)

if __name__ == "__main__":
    app.run(debug=True)
