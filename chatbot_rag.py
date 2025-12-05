import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, CrossEncoder, util  # CrossEncoder'ı ekledik
import faiss
import numpy as np
import json
import os

print("--- RAG Chatbot Başlatılıyor ---")


db_folder = "fen_bilimleri_db"

print("Embedding modeli (retriever) yükleniyor...")

retriever_model_name = 'dbmdz/bert-base-turkish-cased'
retriever_model = SentenceTransformer(retriever_model_name)

print("Cross-Encoder modeli (reranker) yükleniyor...")

reranker_model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
reranker_model = CrossEncoder(reranker_model_name)

print("Vektör veritabanı yükleniyor...")
index = faiss.read_index(os.path.join(db_folder, "faiss_index.idx"))

with open(os.path.join(db_folder, "corpus.json"), "r", encoding="utf-8") as f:
    corpus = json.load(f)

llm_name = "microsoft/phi-1_5"
print(f"Dil modeli ({llm_name}) yükleniyor...")
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
except Exception as e:
    print(f"Dil modeli yüklenirken bir hata oluştu: {e}")
    exit()

print(f"Tüm modeller başarıyla {device} cihazına yüklendi.")


# --- 2. Arama ve Yanıt Üretme Fonksiyonları ---

def search_and_rerank_advanced(query, k_retriever=10, k_reranker=3, score_threshold=0.1):
    """
    Kullanıcının sorusuna en çok benzeyen dokümanları bulur (retrieval),
    bunları daha akıllı bir modelle yeniden sıralar (re-ranking) ve en iyilerini döndürür.
    """
    # 1. Aşama: Geniş Arama (Retrieval)
    query_embedding = retriever_model.encode(query, convert_to_tensor=False)
    distances, ids = index.search(np.array([query_embedding]), k_retriever)

    candidate_docs = [corpus[i] for i in ids[0]]

    # 2. Aşama: Yeniden Sıralama (Re-Ranking)

    reranker_inputs = [[query, doc] for doc in candidate_docs]

    # Skorları hesapla
    scores = reranker_model.predict(reranker_inputs)

    # Dokümanları skorlarıyla birleştir
    reranked_results = []
    for i in range(len(candidate_docs)):

        if scores[i] > score_threshold:
            reranked_results.append({'doc': candidate_docs[i], 'score': scores[i]})

    # Sonuçları skora göre büyükten küçüğe sırala
    reranked_results = sorted(reranked_results, key=lambda x: x['score'], reverse=True)


    return reranked_results[:k_reranker]


def generate_response(question, history):
    """
    Kullanıcının sorusuna RAG ile yanıt üretir.
    """
    print(f"Gelen Soru: {question}")

    print("Veritabanında arama yapılıyor ve sonuçlar yeniden sıralanıyor...")
    # Cross-Encoder için eşik değeri daha düşük olabilir, örneğin 0.1
    similar_docs_with_scores = search_and_rerank_advanced(question, k_retriever=10, k_reranker=3, score_threshold=0.1)

    if not similar_docs_with_scores:
        print("Yeterince alakalı doküman bulunamadı.")
        return "Bu konu hakkında veri setimde bir bilgi bulamadım. Lütfen fen bilimleri ile ilgili bir soru sorun."


    context_docs = [result['doc'] for result in similar_docs_with_scores]
    context = "\n\n---\n\n".join(context_docs)

    print(f"En Alakalı Dokümanlar:\n{context}")

    prompt = f"""Aşağıdaki bağlam bilgisini kullanarak verilen soruyu yanıtla. Sadece bağlamdaki bilgiyi kullan.

Bağlam:
{context}

Soru: {question}

Cevap:"""

    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to(device)

    print("Yanıt üretiliyor...")
    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=250,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id
        )

    response_full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response_full.split("Cevap:")[1].strip()

    print(f"Üretilen Yanıt: {response}")
    return response



iface = gr.ChatInterface(
    generate_response,
    chatbot=gr.Chatbot(height=500),
    textbox=gr.Textbox(placeholder="Fen bilimleri sorunuzu buraya yazın...", container=False, scale=7),
    title="Fen Bilimleri Chatbotu (Gelişmiş RAG ile)",
    description="Fen bilimleri ders kitabınızdaki bilgilere dayanarak sorularınızı yanıtlayan bir chatbot.",
    examples=["güneş olmasaydı ne olurdu?", "Kemik çeşitleri nelerdir?", "çekirdek nedir"],
    theme="soft"
)

if __name__ == "__main__":
    iface.launch(share=False)