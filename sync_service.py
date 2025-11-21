"""
Resumable Google Drive Sync Service (Windows Safe)
--------------------------------------------------

This pipeline performs:

1. Connect to Google Drive
2. Download supported files:
      ✔ PDF
      ✔ DOCX
      ✔ TXT
      ✔ Images → OCR extracted
      ✔ Videos → Placeholder text only
3. Extract readable text (OCR, parser, or fallback)
4. Split text into chunks
5. Generate embeddings for each chunk
6. Store results in FAISS vector store (BATCH SAVE → Windows safe)
7. Save processed filenames → enabling RESUMABLE sync

Resume Logic:
-------------
If sync stops or errors, the system will pick up where it left off.
Previously processed files are skipped.

This version prevents:
✔ Windows file-lock errors
✔ Multiple FAISS writes per chunk
✔ NameError from drive_service
✔ Redundant Drive client creation
"""

import os
import io
import json
from googleapiclient.http import MediaIoBaseDownload

# Local Modules
from backend.app.drive.drive_client import get_drive_service
from backend.app.extractors.extractor import Extractor
from backend.app.embeddings.embedder import EmbeddingModel
from backend.app.processing.chunker import chunk_text
from backend.app.vectorstore.faiss_store import FaissStore


# -------------------------------------------------------------------------
# INITIAL SETUP — Loaded once when backend starts
# -------------------------------------------------------------------------
extractor = Extractor()
embedder = EmbeddingModel()
faiss_store = FaissStore()

RAW_DIR = "backend/app/data/raw"
PROCESSED_FILE = "backend/app/data/processed_files.json"

os.makedirs(RAW_DIR, exist_ok=True)


# -------------------------------------------------------------------------
# Resume Helpers
# -------------------------------------------------------------------------
def load_processed():
    """Load list of files already processed. Returns {} if none exist."""
    if not os.path.exists(PROCESSED_FILE):
        return {}
    try:
        with open(PROCESSED_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}  # corrupted file fallback


def save_processed(data):
    """Save updated dictionary of processed filenames."""
    with open(PROCESSED_FILE, "w") as f:
        json.dump(data, f, indent=4)


# -------------------------------------------------------------------------
# MAIN SYNC PIPELINE
# -------------------------------------------------------------------------
def sync_drive_files():
    """
    Full sync pipeline with resume capability.

    Returns:
        JSON summary of:
            - new_files_indexed
            - total processed files
    """

    print("\n⚡ SYNC STARTED...\n")

    # Files already processed earlier
    processed = load_processed()

    # Supported MIME Types
    mime_types = [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
        "image/png",
        "image/jpeg",
        "video/mp4",
        "video/quicktime",
        "video/x-msvideo",
        "video/x-matroska",
    ]

    # Build Drive search query
    query = " or ".join([f"mimeType='{m}'" for m in mime_types])

    # Lazy-load Drive service
    service = get_drive_service()
    results = service.files().list(q=query).execute()

    files = results.get("files", [])
    print(f"🔵 Total Drive files detected: {len(files)}")

    new_indexed = 0

    # ---------------------------------------------------------------------
    # PROCESS FILES ONE BY ONE
    # ---------------------------------------------------------------------
    for f in files:
        file_name = f["name"]
        file_id = f["id"]
        file_path = os.path.join(RAW_DIR, file_name)

        # Skip if already processed
        if file_name in processed:
            print(f"⏭ SKIP (already processed): {file_name}")
            continue

        print(f"\n📌 Processing new file: {file_name}")

        # -------------------------------------------------------------
        # STEP 1: DOWNLOAD FILE
        # -------------------------------------------------------------
        download_service = get_drive_service()
        request = download_service.files().get_media(fileId=file_id)
        fh = io.FileIO(file_path, "wb")
        downloader = MediaIoBaseDownload(fh, request)

        done = False
        while not done:
            _, done = downloader.next_chunk()

        print("   ✔ Download complete.")

        # -------------------------------------------------------------
        # STEP 2: EXTRACT TEXT
        # -------------------------------------------------------------
        print("   🔍 Extracting text...")
        text = extractor.extract(file_path)

        # If empty → placeholder for video
        if not text.strip():
            print("   ⚠ No extractable text. Saving placeholder.")
            text = f"This is a video file: {file_name}\nDrive Link: https://drive.google.com/file/d/{file_id}"

        # -------------------------------------------------------------
        # STEP 3: CHUNK TEXT
        # -------------------------------------------------------------
        chunks = chunk_text(text)
        print(f"   📚 Total chunks created: {len(chunks)}")

        # -------------------------------------------------------------
        # STEP 4: BATCH EMBED + SAVE (Windows Safe)
        # -------------------------------------------------------------
        vectors = []
        metas = []

        for idx, chunk in enumerate(chunks, 1):
            print(f"      🔹 Embedding chunk {idx}/{len(chunks)}")
            vector = embedder.embed(chunk)

            vectors.append(vector)
            metas.append({
                "file_name": file_name,
                "file_id": file_id,
                "drive_link": f"https://drive.google.com/file/d/{file_id}",
                "snippet": chunk[:250],
            })

        # Save batches ONCE → avoids Windows file locks
        faiss_store.add_batch(vectors, metas)

        print("   ✔ Embeddings saved.")

        # -------------------------------------------------------------
        # STEP 5: UPDATE PROCESSED LIST
        # -------------------------------------------------------------
        processed[file_name] = True
        save_processed(processed)

        new_indexed += 1

    # ---------------------------------------------------------------------
    # END OF SYNC SUMMARY
    # ---------------------------------------------------------------------
    print("\n✅ SYNC FINISHED.\n")

    return {
        "message": "Resumable Sync Completed Successfully",
        "new_files_indexed": new_indexed,
        "already_processed": len(processed)
    }
