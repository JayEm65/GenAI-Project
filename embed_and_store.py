import gc
import time
import psutil
import pickle
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Load your text list
with open("texts.pkl", "rb") as f:
    texts = pickle.load(f)

client = chromadb.Client(Settings(anonymized_telemetry=False))
collection = client.get_or_create_collection(name="news_titles")

model = SentenceTransformer("all-MiniLM-L6-v2")

batch_size = 50
total = len(texts)

def print_ram():
    mem = psutil.virtual_memory()
    print(f"RAM Usage: {mem.percent}% — Available: {round(mem.available / 1024**2)} MB")

for start in range(0, total, batch_size):
    end = min(start + batch_size, total)
    batch_texts = texts[start:end]
    try:
        print(f"\nBatch {start}-{end} — size: {end-start}")
        print_ram()
        batch_embeddings = model.encode(batch_texts).tolist()

        collection.add(
            documents=batch_texts,
            embeddings=batch_embeddings,
            ids=[f"doc_{i}" for i in range(start, end)]
        )
        print(f"✅ Stored batch {start}-{end}")
    except Exception as e:
        print(f"❌ Error in batch {start}-{end}: {str(e)}")
        break

    del batch_texts, batch_embeddings
    gc.collect()
    time.sleep(1)

print("Done.")
