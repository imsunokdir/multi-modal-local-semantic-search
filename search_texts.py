from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-miniLM-L6-v2")
index = faiss.read_index("docs.index")
filenames = np.load("filenames.npy", allow_pickle=True)


def search(query, k=5):
    q_emb = model.encode([query])
    D, I=index.search(q_emb, k)

    print("\nTop Results:")
    for idx in I[0]:
        print("📄",filenames[idx])

if __name__ == "__main__":
    while True:
        q = input("\nSearch: ")
        if q.lower() in ("exit","quit"):
            break
        search(q)