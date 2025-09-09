import pandas as pd
import pickle
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Load the dataset 
movies_data = pd.read_csv("data/movies.csv")

# 2. Standardize column name
movies_data = movies_data.rename(columns={"movie title": "title"})

# 3. Create a text column for feature extraction
movies_data["features"] = (
    movies_data["plot"].fillna("") + " " +
    movies_data["genres"].apply(lambda x: " ".join(eval(x)) if pd.notnull(x) else "")
)

# 4. Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
feature_matrix = vectorizer.fit_transform(movies_data["features"]).toarray().astype("float32")

# 5. Build FAISS index
index = faiss.IndexFlatL2(feature_matrix.shape[1])
index.add(feature_matrix)

# 6. Save FAISS index
faiss.write_index(index, "models/movie_index.faiss")

# 7. Save metadata + vectorizer + feature matrix
with open("models/movies_meta.pkl", "wb") as f:
    pickle.dump((movies_data, vectorizer, feature_matrix), f)

print("✅ FAISS index and metadata saved successfully!")
