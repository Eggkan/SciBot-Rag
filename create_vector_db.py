from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import json

print("--- Vektör Veritabanı Oluşturma Başladı ---")

# 1. Veri Setini Yükle
dataset_path = 'fen_bilimleri_training_data_x5.json'
if not os.path.exists(dataset_path):
    print(f"HATA: '{dataset_path}' dosyası bulunamadı.")
    exit()

print("Veri seti yükleniyor...")
dataset = load_dataset("json", data_files=dataset_path, split="train")


corpus = [
    f"Soru: {item['instruction']}\nCevap: {item['output']}"
    for item in dataset
]
print(f"Toplam {len(corpus)} adet doküman oluşturuldu.")

# 2. Embedding Modelini Yükle

print("Embedding modeli (dbmdz/bert-base-turkish-cased) yükleniyor...")
embedding_model_name = 'dbmdz/bert-base-turkish-cased'
try:
    embedding_model = SentenceTransformer(embedding_model_name)
    print("Embedding modeli yüklendi.")
except Exception as e:
    print(f"Embedding modeli yüklenirken bir hata oluştu: {e}")
    exit()

# 3. Metinleri Vektörlere Çevir (Embeddings)
print("Dokümanlar vektörlere çevriliyor (Bu işlem biraz zaman alabilir)...")
embeddings = embedding_model.encode(corpus, convert_to_tensor=False, show_progress_bar=True)
print("Vektörler oluşturuldu. Boyut:", embeddings.shape)

# 4. FAISS ile Vektör Veritabanı Oluştur
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index = faiss.IndexIDMap(index)

index.add_with_ids(np.array(embeddings), np.arange(len(corpus)))
print(f"Toplam {index.ntotal} vektör FAISS index'ine eklendi.")

# 5. Veritabanını ve Corpus'u Kaydet
db_folder = "fen_bilimleri_db"
os.makedirs(db_folder, exist_ok=True)

faiss.write_index(index, os.path.join(db_folder, "faiss_index.idx"))

with open(os.path.join(db_folder, "corpus.json"), "w", encoding="utf-8") as f:
    json.dump(corpus, f, ensure_ascii=False, indent=2)

print(f"--- Vektör Veritabanı '{db_folder}' klasörüne başarıyla kaydedildi! ---")