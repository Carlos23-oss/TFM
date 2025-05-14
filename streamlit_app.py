# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

# Function to load artifacts
def load_artifacts(path: str, use_gpu: bool = False) -> tuple[pd.DataFrame, np.ndarray, faiss.Index]:
    """Load the three artifacts saved in `save_artifacts`."""
    try:
        df = pd.read_parquet(os.path.join(path, "df_rec.parquet"))
        emb = np.load(os.path.join(path, "embeddings.npy"))
        index = faiss.read_index(os.path.join(path, "faiss.index"))
        if use_gpu and not isinstance(index, faiss.IndexGPU):
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        return df, emb, index
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Artifact not found in {path}: {e}")

# Function to recommend based on movie titles
def recommend(
    user_titles: list[str],
    df: pd.DataFrame,
    index: faiss.IndexFlatIP,
    embeddings: np.ndarray,
    k: int = 5,
) -> pd.DataFrame:
    """Return the *k* movies most similar to `user_titles`."""
    if "tmdb_original_title" not in df.columns:
        raise KeyError(
            "The DataFrame does not contain the column 'tmdb_original_title'. "
            "Ensure the column name is correct."
        )

    titles_lower = df["tmdb_original_title"].astype(str).str.lower()
    mask = titles_lower.isin([t.lower() for t in user_titles])

    if not mask.any():
        raise ValueError(
            "None of the provided titles are found in the catalog."
        )

    user_idx = np.where(mask)[0]
    query_emb = embeddings[user_idx].mean(axis=0, keepdims=True)

    _, idxs = index.search(query_emb, k + len(user_idx))
    seen = set(user_idx)
    rec_indices = [idx for idx in idxs[0] if idx not in seen][:k]

    return df.iloc[rec_indices].reset_index(drop=True)

# Function to recommend based on free text
def recommend_from_text(
    query_text: str,
    df: pd.DataFrame,
    index: faiss.Index,
    embeddings: np.ndarray,
    k: int = 5,
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
) -> pd.DataFrame:
    """Return the *k* movies most similar to a free-text description."""
    try:
        model = SentenceTransformer(model_name)
        model.max_seq_length = 128
        query_emb = model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
        _, idxs = index.search(query_emb.astype("float32"), k)
        rec_indices = idxs[0].tolist()
        return df.iloc[rec_indices].reset_index(drop=True)
    except Exception as e:
        raise RuntimeError(f"Error in text-based recommendation: {e}")

# Streamlit app
st.title("Movie Recommender System")
st.markdown("Recommend movies based on your favorite titles or a description of your mood!")

# Path to artifacts (adjust if necessary)
save_path = "recommender_artifacts"

# Check if artifacts directory exists
if not os.path.exists(save_path):
    st.error(f"Artifacts directory '{save_path}' not found.")
    st.stop()

# Load artifacts
with st.spinner("Loading artifacts..."):
    try:
        df_loaded, emb_loaded, ix_loaded = load_artifacts(save_path, use_gpu=False)
        st.success("Artifacts loaded successfully!")
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        st.stop()

# Sidebar for recommendation type selection
recommendation_type = st.sidebar.selectbox(
    "Choose recommendation type:",
    ["Based on movie titles", "Based on text description"]
)

# Columns to display in recommendations
display_columns = [
    "tmdb_original_title",
    "imdb_startyear",
    "imdb_director",
    "imdb_writer",
    "tmdb_production_countries",
    "combined_genres",
    "runtime"
]

# Recommendation based on movie titles
if recommendation_type == "Based on movie titles":
    st.subheader("Select your favorite movies")
    # Get unique movie titles for the dropdown
    movie_options = df_loaded["tmdb_original_title"].astype(str).unique().tolist()
    selected_movies = st.multiselect(
        "Choose at least one movie:",
        options=movie_options,
        default=["Lost Highway", "Blue Velvet"]  # Optional default
    )

    if st.button("Get Recommendations"):
        if not selected_movies:
            st.warning("Please select at least one movie.")
        else:
            try:
                recommendations = recommend(
                    selected_movies,
                    df_loaded,
                    ix_loaded,
                    emb_loaded,
                    k=5
                )
                st.subheader("Recommended Movies")
                st.dataframe(recommendations[display_columns])
            except Exception as e:
                st.error(f"Error generating recommendations: {e}")

# Recommendation based on text description
else:
    st.subheader("Describe your mood or preferences")
    free_text = st.text_area(
        "Enter a description (e.g., 'I love Scorsese and feel happy today'):",
        value="I'm very happy today and love Scorsese"
    )

    if st.button("Get Recommendations"):
        if not free_text.strip():
            st.warning("Please enter a description.")
        else:
            try:
                recommendations = recommend_from_text(
                    free_text,
                    df_loaded,
                    ix_loaded,
                    emb_loaded,
                    k=5
                )
                st.subheader("Recommended Movies")
                st.dataframe(recommendations[display_columns])
            except Exception as e:
                st.error(f"Error generating recommendations: {e}")
