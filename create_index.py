import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import time
import os

print("Starting index creation process for Material and Labour...")

# --- Configuration ---
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
FILES_TO_PROCESS = {
    "material": {
        "excel_path": "MaterialCode.xlsx",
        "index_path": "material.index",
        "db_path": "material_db.pkl"
    },
    "labour": {
        "excel_path": "LabourCode.xlsx",
        "index_path": "labour.index",
        "db_path": "labour_db.pkl"
    }
}

# --- Load Model ---
print(f"Loading sentence transformer model: {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)
print("Model loaded.")

# --- Process each file type ---
for file_type, config in FILES_TO_PROCESS.items():
    print("-" * 50)
    print(f"Processing: {file_type.upper()}")
    
    excel_path = config["excel_path"]
    if not os.path.exists(excel_path):
        print(f"Warning: File '{excel_path}' not found. Skipping this type.")
        continue

    # 1. Load data
    df = pd.read_excel(excel_path)
    df.rename(columns={'Product Code': 'Code', 'ชื่อในPO': 'DB_Description'}, inplace=True)
    df.dropna(subset=['DB_Description', 'Code'], inplace=True)
    df['DB_Description'] = df['DB_Description'].astype(str)
    df.reset_index(drop=True, inplace=True)
    print(f"Loaded {len(df)} items from {excel_path}.")

    # 2. Create embeddings
    print("Creating embeddings...")
    start_time = time.time()
    embeddings = model.encode(df['DB_Description'].tolist(), show_progress_bar=True, convert_to_numpy=True)
    end_time = time.time()
    print(f"Embedding creation took {end_time - start_time:.2f} seconds.")

    # 3. Normalize and build FAISS index
    faiss.normalize_L2(embeddings)
    embedding_dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(embedding_dimension)
    index.add(embeddings)
    print(f"Created FAISS index with {index.ntotal} vectors.")

    # 4. Save index and database
    faiss.write_index(index, config["index_path"])
    df.to_pickle(config["db_path"])
    print(f"Saved '{config['index_path']}' and '{config['db_path']}'.")

print("-" * 50)
print("\nProcess complete!")