from flask import Flask, render_template, request
import pickle
import faiss
import numpy as np

app = Flask(__name__)

# ✅ Load precomputed FAISS index and metadata
with open("models/movies_meta.pkl", "rb") as f:
    movies_data, vectorizer, feature_matrix = pickle.load(f)

index = faiss.read_index("models/movie_index.faiss")

# Convert titles to lowercase for easier matching
title_map = {title.lower(): i for i, title in enumerate(movies_data['title'])}

def get_recommendations(movie_name, top_n=10):
    movie_name_lower = movie_name.lower().strip()

    if movie_name_lower not in title_map:
        return [f"No match found for '{movie_name}'"]

    # Get index of the movie
    movie_idx = title_map[movie_name_lower]

    # Get the feature vector of this movie
    query_vector = feature_matrix[movie_idx].reshape(1, -1).astype(np.float32)

    # Search in FAISS index
    distances, indices = index.search(query_vector, top_n + 10)  # fetch extra

    # Build recommendation list
    recommendations = []
    seen = set()

    for idx in indices[0]:
        if idx == movie_idx:
            continue  # ✅ skip same movie
        title = movies_data.iloc[idx]['title']
        if title not in seen:
            recommendations.append(title)
            seen.add(title)
        if len(recommendations) >= top_n:
            break

    return recommendations



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
