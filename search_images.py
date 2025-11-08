from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("clip-ViT-B-32")
index = faiss.read_index("image_docs.index")
filenames = np.load("image_filenames.npy", allow_pickle=True)

query = input("Enter image prompt (e.g., 'a dog running'): ")
query_emb = model.encode(query, convert_to_numpy=True).reshape(1, -1)

D, I = index.search(query_emb, 5)

print("\nTop matches:")
for idx, score in zip(I[0], D[0]):
    print(f"{filenames[idx]}  (distance: {score:.4f})")
