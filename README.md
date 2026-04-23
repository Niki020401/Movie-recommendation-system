# 🎬 Movie Recommendation System  

**High-performance movie recommendation system powered by SBERT embeddings and FAISS for real-time semantic search.**

---

## 🚀 Why This Project Exists  

Traditional recommendation systems rely on:  
- **Keyword matching** (shallow and inaccurate)  
- **Ratings-based filtering** (limited personalization)  

These approaches fail to understand **user intent**.

This system solves the problem using **semantic search + vector similarity**, allowing users to describe movies in natural language and receive accurate recommendations instantly.

---

## 🧠 System Architecture  

### 🔹 1. Semantic Encoding (SBERT)  
- Converts movie metadata (title, overview, genre, cast) into **dense embeddings**  
- Converts user queries into the **same vector space**  
- Captures **context, intent, and meaning**  

**Example:**  
> “time-loop psychological thriller” ≈ “mind-bending sci-fi paradox”  

---

### 🔹 2. Vector Indexing (FAISS)  
- Stores embeddings in a **FAISS index**  
- Uses **Approximate Nearest Neighbor (ANN)** search  
- Reduces complexity from **O(n)** to near constant time  

⚡ **Result:** Millisecond-level retrieval  

---

### 🔹 3. Recommendation Engine  
- Retrieves **Top-K similar movies**  
- Refines ranking using:  
  - Metadata filters (year, rating, genre)  
  - Relevance scoring  

---

## ⚙️ Key Features  

- ✔ Natural Language Querying  
- ✔ Semantic Search (not keyword-based)  
- ✔ High-Speed FAISS Retrieval  
- ✔ Scalable Vector Architecture  
- ✔ Dynamic Query Filtering  
- ✔ Extensible Hybrid Recommendation Design  

---

## 🛠️ Tech Stack  

| Layer          | Technology                          |
|----------------|------------------------------------|
| NLP            | Sentence-BERT (`all-MiniLM-L6-v2`) |
| Vector Search  | FAISS                              |
| Backend        | Python                             |
| UI             | Streamlit                          |
| Data Handling  | Pandas, NumPy                      |

---

## 📊 Performance Highlights  

- Efficient handling of **high-dimensional embeddings**  
- Retrieval latency in **milliseconds**  
- Scales to **large datasets** without brute-force search  
- Outperforms traditional filtering approaches  

---

## 📦 Setup  

```bash
git clone https://github.com/your-username/CineSage.git
cd CineSage
pip install -r requirements.txt
streamlit run app.py
