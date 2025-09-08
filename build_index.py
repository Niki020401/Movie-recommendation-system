import pandas as pd
import faiss
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Create "models" folder if it doesn’t exist
os.makedirs("models", exist_ok=True)

# Load dataset
movies_data = pd.read_csv("data/movies.csv")

# Ensure correct column
if "movie title" in movies_data.columns:
    movies_data.rename(columns={"movie title": "title"}, inplace=True)

# Select features
selected_features = [col for col in ['genres', 'cast', 'director', 'plot'] if col in movies_data.columns]

# Fill missing values
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

# Combine features
combined_features = movies_data[selected_features].agg(' '.join, axis=1)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
feature_vectors = vectorizer.fit_transform(combined_features)

# Convert to numpy
feature_matrix = feature_vectors.toarray().astype('float32')

# Build FAISS index
d = feature_matrix.shape[1]
index = faiss.IndexFlatIP(d)
faiss.normalize_L2(feature_matrix)
index.add(feature_matrix)

# Save index & metadata
faiss.write_index(index, "models/movie_index.faiss")
with open("models/movies_meta.pkl", "wb") as f:
    pickle.dump((movies_data, vectorizer, feature_matrix), f)

print("✅ FAISS index and metadata saved.")
