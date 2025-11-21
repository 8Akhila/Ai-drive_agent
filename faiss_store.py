import os
import faiss
import numpy as np
import json

VECTOR_DIM = 384  # for all-MiniLM-L6-v2 model

# Save inside data folder (correct)
INDEX_PATH = "backend/app/data/faiss_index.bin"
META_PATH = "backend/app/data/faiss_meta.json"


class FaissStore:
    def __init__(self, dim: int = VECTOR_DIM):
        self.dim = dim

        # Load or create FAISS index
        if os.path.exists(INDEX_PATH):
            self.index = faiss.read_index(INDEX_PATH)
        else:
            self.index = faiss.IndexFlatL2(self.dim)

        # Load metadata
        if os.path.exists(META_PATH):
            with open(META_PATH, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
        else:
            self.meta = []

    def save(self):
        """Save FAISS index + metadata safely."""
        print(f"💾 Saving FAISS index → {INDEX_PATH}")
        print(f"💾 Saving metadata → {META_PATH}")

        faiss.write_index(self.index, INDEX_PATH)

        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2)

        print("✅ Save complete")

    def add(self, vector: np.ndarray, metadata: dict):
        """Add a single vector (not used now, kept for compatibility)."""
        if vector.ndim == 1:
            vector = np.expand_dims(vector, axis=0)

        vector = vector.astype("float32")
        self.index.add(vector)
        self.meta.append(metadata)
        self.save()

    def add_batch(self, vectors, metadata_list):
        """Add multiple vectors at once — Windows-safe."""
        vectors = np.asarray(vectors, dtype="float32")
        self.index.add(vectors)

        for m in metadata_list:
            self.meta.append(m)

        self.save()

    def search(self, vector: np.ndarray, k: int = 5):
        """Search closest K vectors and return their metadata."""
        if vector.ndim == 1:
            vector = np.expand_dims(vector, axis=0)

        vector = vector.astype("float32")
        distances, ids = self.index.search(vector, k)

        results = []
        for idx in ids[0]:
            if 0 <= idx < len(self.meta):
                results.append(self.meta[idx])

        return results
