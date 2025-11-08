from sentence_transformers import SentenceTransformer
import faiss
import os
import numpy as np
from PIL import Image

def index_images(folder_path):
    model = SentenceTransformer("clip-ViT-B-32")

    embeddings, filenames = [], []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            filepath = os.path.join(root, file)
            ext = file.lower().split('.')[-1]

            if ext in ["png", "jpg", "jpeg", "bmp", "gif"]:
                try:
                    img = Image.open(filepath).convert("RGB")
                    emb = model.encode(img, convert_to_numpy=True)
                    filenames.append(filepath)
                    embeddings.append(emb)
                    print(f"🖼️ Indexed image: {file}")
                except Exception as e:
                    print(f"Error reading {file}: {e}")

    if not embeddings:
        print("⚠️ No image files found.")
        return 0

    embeddings = np.vstack(embeddings)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, "image_docs.index")
    np.save("image_filenames.npy", filenames)

    print(f"✅ Finished indexing {len(filenames)} image files from {folder_path}")
    return len(filenames)
