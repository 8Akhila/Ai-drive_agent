# --------------------------------------------------------------
# Embedding Model (Local Offline Embeddings) – Safe Version
# --------------------------------------------------------------

import os
import numpy as np
from sentence_transformers import SentenceTransformer

LOCAL_MODEL_PATH = "./local_model"

MAX_CHARS = 2000   # prevent memory explodes
                  # (chunker creates ~800 char chunks so this is SAFE)


class EmbeddingModel:
    """
    Safe embedding wrapper for generating embeddings using a local model.
    Protects against memory overflow by truncating long text.
    """

    def __init__(self):
        # confirm the local model exists
        if not os.path.exists(LOCAL_MODEL_PATH):
            raise FileNotFoundError(
                "❌ Local model NOT found! Run: python backend/download_model.py"
            )

        print(f"🔵 Loading local embedding model from: {LOCAL_MODEL_PATH}")
        self.model = SentenceTransformer(LOCAL_MODEL_PATH)

    # ------------ Utility: ensure safe input ------------
    def _prepare(self, text: str) -> str:
        text = text.strip()
        if len(text) > MAX_CHARS:
            # truncate to avoid memory overload
            text = text[:MAX_CHARS]
        return text

    # ------------ Embed a regular text chunk ------------
    def embed(self, text: str) -> np.ndarray:
        safe_text = self._prepare(text)
        emb = self.model.encode(safe_text, convert_to_numpy=True)
        return emb.astype("float32")

    # ------------ Embed a query ------------
    def embed_query(self, query: str) -> np.ndarray:
        safe_query = self._prepare(query)
        emb = self.model.encode(safe_query, convert_to_numpy=True)
        return emb.astype("float32")

    # ------------ Embed a single text to maintain compatibility ------------
    def embed_text(self, text: str) -> np.ndarray:
        return self.embed(text)
