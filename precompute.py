import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load movie data
movies_data = pd.read_csv("movies.csv")

# ✅ Rename "movie title" to "title" for consistency
if "movie title" in movies_data.columns:
    movies_data.rename(columns={"movie title": "title"}, inplace=True)

# ✅ Select only available features from CSV
selected_features = []
for col in ['genres', 'cast', 'director', 'plot']:
    if col in movies_data.columns:
        selected_features.append(col)

print("\nUsing features:", selected_features)

# ✅ Fill missing values
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

# ✅ Combine features into one string
combined_features = movies_data[selected_features].agg(' '.join, axis=1)

# ✅ Convert text to vectors
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# ✅ Compute similarity matrix
similarity = cosine_similarity(feature_vectors)

# ✅ Save precomputed data for Flask app
with open("precomputed.pkl", "wb") as f:
    pickle.dump((movies_data, similarity), f)

print("\n✅ Precomputed similarity saved to precomputed.pkl")
