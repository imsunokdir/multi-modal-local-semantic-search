from sentence_transformers import SentenceTransformer
import faiss
import os
import numpy as np
from text_extractor import extract_text_from_file

def index_texts(folder_path):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts, filenames = [], []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            filepath = os.path.join(root, file)
            text = extract_text_from_file(filepath)

            if text.strip() != "":
                texts.append(text)
                filenames.append(filepath)
                print("Indexed:", file)

    if not texts:
        print("⚠️ No valid text found in folder.")
        return

    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, "text_docs.index")
    np.save("filenames.npy", filenames)

    print(f"✅ Finished indexing {len(filenames)} files from {folder_path}")
    return len(filenames)
