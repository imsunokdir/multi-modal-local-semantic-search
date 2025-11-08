import streamlit as st
import os
import time
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from index_texts import index_texts
from index_images import index_images

st.title("🧠 AI File Search")

mode = st.radio("Choose what to search:", ["📄 Text Files", "🖼️ Image Files"])
folder_path = st.text_input("📂 Enter folder path to search:")

index_info_file = "index_meta.pkl"

# -------------------- Helper functions --------------------
def get_folder_snapshot(folder_path):
    snapshot = {}
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            path = os.path.join(root, f)
            try:
                snapshot[path] = os.path.getmtime(path)
            except FileNotFoundError:
                continue
    return snapshot

def load_index_info():
    if os.path.exists(index_info_file):
        with open(index_info_file, "rb") as f:
            return pickle.load(f)
    return {}

def save_index_info(data):
    with open(index_info_file, "wb") as f:
        pickle.dump(data, f)

def needs_reindex(folder_path, current_snapshot, mode):
    data = load_index_info()
    key = f"{folder_path}_{mode}"
    if key not in data:
        return True
    return data[key] != current_snapshot

def update_index_metadata(folder_path, current_snapshot, mode):
    data = load_index_info()
    key = f"{folder_path}_{mode}"
    data[key] = current_snapshot
    save_index_info(data)

def distance_to_similarity(distance):
    """Convert L2 distance to similarity percentage (0-100%)"""
    # Using exponential decay: similarity = 100 * e^(-distance)
    # This gives ~100% for distance=0, ~37% for distance=1, ~14% for distance=2
    similarity = 100 * np.exp(-distance)
    return similarity

# -------------------- Search section --------------------
st.markdown("---")
st.subheader("🔎 Search Files")

query = st.text_input("Enter your search query:")

if st.button("Search"):
    if not os.path.isdir(folder_path):
        st.error("❌ Invalid folder path.")
    elif not query.strip():
        st.warning("⚠️ Please enter a search query.")
    else:
        # Step 1: Check folder for changes
        check_placeholder = st.empty()
        check_placeholder.info("🔍 Checking folder for changes...")
        snapshot = get_folder_snapshot(folder_path)
        time.sleep(0.5)
        check_placeholder.success("✅ Checked folder for changes")

        # Step 2: If needed, reindex
        if needs_reindex(folder_path, snapshot, mode):
            with st.spinner("⚙️ Reindexing changed files..."):
                start_time = time.time()
                if mode == "📄 Text Files":
                    count = index_texts(folder_path)
                else:
                    count = index_images(folder_path)
                update_index_metadata(folder_path, snapshot, mode)
                duration = time.time() - start_time
            st.success(f"✅ Reindexing completed — {count} files processed in {duration:.2f}s")
        else:
            st.info("✅ No reindexing needed. Using existing data.")

        # Step 3: Perform search
        st.markdown("---")
        if mode == "📄 Text Files":
            index_path = "text_docs.index"
            if not os.path.exists(index_path):
                st.error("⚠️ No text index found. Please search again.")
            else:
                model = SentenceTransformer("all-MiniLM-L6-v2")
                index = faiss.read_index(index_path)
                filenames = np.load("filenames.npy", allow_pickle=True)

                q_emb = model.encode([query], convert_to_numpy=True)
                D, I = index.search(q_emb, 5)

                st.write("### 📂 Top Results:")
                for idx, distance in zip(I[0], D[0]):
                    similarity = distance_to_similarity(distance)
                    # Color code based on similarity
                    if similarity >= 80:
                        color = "🟢"
                    elif similarity >= 60:
                        color = "🟡"
                    else:
                        color = "🔴"
                    st.write(f"{color} **{similarity:.1f}%** — {filenames[idx]}")

        else:
            index_path = "image_docs.index"
            if not os.path.exists(index_path):
                st.error("⚠️ No image index found. Please search again.")
            else:
                model = SentenceTransformer("clip-ViT-B-32")
                index = faiss.read_index(index_path)
                filenames = np.load("image_filenames.npy", allow_pickle=True)

                q_emb = model.encode([query], convert_to_numpy=True).reshape(1, -1)
                D, I = index.search(q_emb, 5)

                st.write("### 🖼️ Top Matches:")
                for idx, distance in zip(I[0], D[0]):
                    similarity = distance_to_similarity(distance)
                    # Color code based on similarity
                    if similarity >= 80:
                        color = "🟢"
                    elif similarity >= 60:
                        color = "🟡"
                    else:
                        color = "🔴"
                    st.write(f"{color} **{similarity:.1f}%** — {filenames[idx]}")