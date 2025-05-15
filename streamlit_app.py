app_code = """
import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

def load_artifacts(path: str, use_gpu: bool = False) -> tuple[pd.DataFrame, np.ndarray, faiss.Index]:
    df = pd.read_parquet(os.path.join(path, "df_rec.parquet"))
    emb = np.load(os.path.join(path, "embeddings.npy"))
    index = faiss.read_index(os.path.join(path, "faiss.index"))
    if use_gpu and not isinstance(index, faiss.IndexGPU):
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    return df, emb, index

def recommend(user_titles, df, index, embeddings, k=5):
    titles_lower = df["tmdb_original_title"].astype(str).str.lower()
    mask = titles_lower.isin([t.lower() for t in user_titles])
    if not mask.any():
        raise ValueError("None of the provided titles are found.")
    user_idx = np.where(mask)[0]
    query_emb = embeddings[user_idx].mean(axis=0, keepdims=True)
    _, idxs = index.search(query_emb, k + len(user_idx))
    seen = set(user_idx)
    rec_indices = [idx for idx in idxs[0] if idx not in seen][:k]
    return df.iloc[rec_indices].reset_index(drop=True)

def recommend_from_text(query_text, df, index, embeddings, k=5, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    model = SentenceTransformer(model_name)
    model.max_seq_length = 128
    query_emb = model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
    _, idxs = index.search(query_emb.astype("float32"), k)
    rec_indices = idxs[0].tolist()
    return df.iloc[rec_indices].reset_index(drop=True)

st.title("Movie Recommender System")
st.markdown("Recomienda películas a partir de tus favoritas o de una descripción de tu estado de ánimo.")

save_path = "recommender_artifacts"

if not os.path.exists(save_path):
    st.error(f"No se encontró la carpeta '{save_path}'.")
    st.stop()

with st.spinner("Cargando modelo y datos..."):
    try:
        df_loaded, emb_loaded, ix_loaded = load_artifacts(save_path, use_gpu=False)
        st.success("¡Modelo cargado correctamente!")
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

recommendation_type = st.sidebar.selectbox("Tipo de recomendación:", ["Por títulos", "Por descripción"])

columns_to_show = [
    "tmdb_original_title", "imdb_startyear", "imdb_director",
    "imdb_writer", "tmdb_production_countries", "combined_genres", "runtime"
]

if recommendation_type == "Por títulos":
    movie_options = df_loaded["tmdb_original_title"].astype(str).unique().tolist()
    selected_movies = st.multiselect("Selecciona una o más películas:", options=movie_options, default=["Lost Highway"])
    if st.button("Recomendar"):
        if not selected_movies:
            st.warning("Selecciona al menos una película.")
        else:
            try:
                recs = recommend(selected_movies, df_loaded, ix_loaded, emb_loaded, k=5)
                st.dataframe(recs[columns_to_show])
            except Exception as e:
                st.error(f"Error: {e}")
else:
    query = st.text_area("Describe tus gustos o tu estado de ánimo:", value="Me siento feliz y me gusta Scorsese.")
    if st.button("Recomendar"):
        if not query.strip():
            st.warning("Introduce una descripción.")
        else:
            try:
                recs = recommend_from_text(query, df_loaded, ix_loaded, emb_loaded, k=5)
                st.dataframe(recs[columns_to_show])
            except Exception as e:
                st.error(f"Error: {e}")
"""

with open("app_recommender.py", "w") as f:
    f.write(app_code)

print("✅ Archivo 'app_recommender.py' creado correctamente.")
