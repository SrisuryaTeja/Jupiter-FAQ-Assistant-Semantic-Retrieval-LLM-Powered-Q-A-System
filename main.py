import os
import re
import html
import json
import faiss
import numpy as np
import requests
from langdetect import detect
from googletrans import Translator
import time
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv, find_dotenv


dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
API_KEY=os.getenv()

# ── Configure Your Hosted Open-Source LLM API ────────────────────────────────
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"
MULTILINGUAL_MODEL = "paraphrase-multilingual-MiniLM-L12-v2" 

translator = Translator()

# ── Cleaning Utility ─────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'@[\w\-]+', '', text)
    text = re.sub(r':[\w\-]+:', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'^\W+|\W+$', '', text)
    return text.strip()


def ensure_english(text: str) -> str:
    try:
        lang = detect(text)
        if lang != 'en':
            return translator.translate(text, dest='en').text
    except:
        pass
    return text

# ── Tag Classification via Together.ai ───────────────────────────────────────
def classify_tags_with_open_llm(query: str, possible_tags: list) -> list:
    prompt = f"""
You are a helpful assistant that assigns relevant tags to user questions about Jupiter.money. If the question is unrelated or ambiguous, respond with "none".

Possible tags: {', '.join(possible_tags)}

Classify this question into the most relevant 2–3 tags. If it is unrelated, say "none".

Question:
"{query}"

Respond with a comma-separated list of tags or "none".
"""

    response = requests.post(
        "https://api.together.xyz/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 100,
        },
    )

    content = response.json()["choices"][0]["message"]["content"]
    content = content.strip().lower()
    if "none" in content:
        return []
    return [tag.strip() for tag in content.split(",") if tag.strip()]

# ── Answer Generation via Together.ai ────────────────────────────────────────
def generate_answer_with_open_llm(query: str, context_posts: list) -> str:
    if not context_posts:
        return ("I'm not sure about that yet. You can try rephrasing your question "
                "or visit the Jupiter Community for more help.")

    # Build context: top 3 posts
    context = "\n\n".join(
        f"{post['user']['name']}: {post['text']}"
        for post in context_posts[:3]
    )

    system_prompt = (
        "You are a helpful assistant for Jupiter.money.\n"
        "Use the provided forum discussion to answer the user's question clearly and politely.\n"
        "Rephrase responses in friendly, natural language.\n"
        "If you're not sure or the context is unrelated, say so gracefully."
    )
    user_prompt = (
        f"User question: {query}\n\n"
        f"Forum context:\n{context}\n\n"
        "Answer the user based on the forum context:"
    )

    resp = requests.post(
        "https://api.together.xyz/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            "temperature": 0.5,
            "max_tokens": 512,
        },
    )
    return resp.json()["choices"][0]["message"]["content"].strip()

# ── File Constants ───────────────────────────────────────────────────────────
RAW_JSON       = "faq_data_raw.json"
INDEX_FILE     = "faq.index"
QUESTIONS_FILE = "faq_questions.json"
METADATA_FILE  = "faq_metadata.json"
EMBEDDINGS_FILE = "faq_embeddings.npy"

# ── Load & Clean Data ─────────────────────────────────────────────────────────
def load_and_prepare_data():
    with open(RAW_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions, metadatas = [], []
    seen = set()
    for topic in data:
        q = clean_text(topic.get("text", ""))
        if not q or q in seen:
            continue
        seen.add(q)

        # clean posts
        cleaned_posts = []
        for p in topic.get("posts", []):
            cp = p.copy()
            cp["text"] = clean_text(cp.get("text", ""))
            cleaned_posts.append(cp)

        questions.append(q)
        metadatas.append({
            "tags":  topic.get("tags", []),
            "title": clean_text(topic.get("title", "")),
            "url":   topic.get("url", ""),
            "posts": cleaned_posts,
        })

    print(f"✅ Loaded {len(questions)} unique questions.")
    return questions, metadatas

# ── Build or Load FAISS Index ─────────────────────────────────────────────────
def build_or_load_index_and_embeddings(questions, metadatas):
    model_multi = SentenceTransformer(MULTILINGUAL_MODEL)
    if os.path.exists(INDEX_FILE) and os.path.exists(EMBEDDINGS_FILE):
        idx = faiss.read_index(INDEX_FILE)
        embs = np.load(EMBEDDINGS_FILE)
        qs = json.load(open(QUESTIONS_FILE)); ms = json.load(open(METADATA_FILE))
        print("Loaded existing index and embeddings.")
        return idx, embs, qs, ms, model_multi
    embs = model_multi.encode(questions, convert_to_numpy=True, batch_size=32)
    faiss.normalize_L2(embs)
    idx = faiss.IndexFlatIP(embs.shape[1]); idx.add(embs)
    faiss.write_index(idx, INDEX_FILE); np.save(EMBEDDINGS_FILE, embs)
    json.dump(questions, open(QUESTIONS_FILE, "w"), indent=2)
    json.dump(metadatas, open(METADATA_FILE, "w"), indent=2)
    print(f"Built index with {idx.ntotal} vectors.")
    return idx, embs, questions, metadatas, model_multi 

query_history = []

def suggest_related(
    query: str,
    questions: list,
    idx: faiss.IndexFlatIP,
    model: SentenceTransformer,
    embeddings: np.ndarray,
    top_k: int = 5
) -> list:
    """
    Suggest past questions semantically similar to `query`.
    Uses the full FAISS index `idx` over `embeddings`.
    """
    # 1️⃣ Record the query
    query_history.append(query)

    # 2️⃣ Embed & normalize
    q_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)

    # 3️⃣ Search the **full** index
    D, I = idx.search(q_emb, top_k)
    scores, ids = D[0], I[0]

    # 4️⃣ Map indices back to question texts
    suggestions = [questions[i] for i in ids]

    return suggestions



# ── Retrieval + Post-Filtering ────────────────────────────────────────────────
def retrieve_with_prefilter(
    query: str,
    predicted_tags: list,
    questions: list,
    metadatas: list,
    model: SentenceTransformer,
    embeddings: np.ndarray,
    top_k: int = 10
):
    # 1️⃣ Build allowed_ids
    if predicted_tags:
        allowed_ids = [
            i for i, meta in enumerate(metadatas)
            if any(tag in meta["tags"] for tag in predicted_tags)
        ]
    else:
        allowed_ids = [i for i, meta in enumerate(metadatas) if not meta["tags"]]

    if not allowed_ids:
        return []

    # 2️⃣ Build a temporary sub-index
    sub_embs = embeddings[allowed_ids]
    faiss.normalize_L2(sub_embs)
    dim = sub_embs.shape[1]
    sub_index = faiss.IndexFlatIP(dim)
    sub_index.add(sub_embs)

    # 3️⃣ Encode & normalize query
    q_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)

    # 4️⃣ Search
    D, I = sub_index.search(q_emb, top_k)
    scores, idxs = D[0], I[0]

    # 5️⃣ Map back to original
    results = []
    for score, sub_i in zip(scores, idxs):
        orig_i = allowed_ids[sub_i]
        md = metadatas[orig_i]
        results.append({
            "question": questions[orig_i],
            "score":    float(score),
            "tags":     md["tags"],
            "title":    md["title"],
            "url":      md["url"],
            "posts":    md["posts"],
        })
    return results

# ── Main Interactive Loop ────────────────────────────────────────────────────
if __name__ == "__main__":
    questions, metadatas = load_and_prepare_data()
    idx, embs, questions, metadatas, model = build_or_load_index_and_embeddings(questions, metadatas)
    all_tags = sorted({t for m in metadatas for t in m["tags"]})
    while True:
        q = input("Ask a question (exit to quit): ").strip()
        if q.lower() == "exit": break
        tags = classify_tags_with_open_llm(q, all_tags)
        print(f"Predicted tags: {tags or 'none'}")
        hits = retrieve_with_prefilter(q, tags, questions, metadatas, model, embs, top_k=8)
        # 1) Retrieval-based
        start = time.time()
        ret_answer = generate_answer_with_open_llm(q, hits[0]["posts"] if hits else [])
        ret_latency = time.time() - start

        # 2) LLM-only (no retrieval context)
        start = time.time()
        llm_only_answer = generate_answer_with_open_llm(q, [])
        llm_latency = time.time() - start

        # 3) Display comparison
        print(f"\n⏱ Retrieval-based ({ret_latency:.2f}s):\n{ret_answer}\n")
        print(f"⏱ LLM-only    ({llm_latency:.2f}s):\n{llm_only_answer}\n")
        suggestions = suggest_related(q, questions, metadatas, model, embs, top_k=5)
        print("💡 You might also be interested in these related questions:")
        for s in suggestions:
            print("  •", s)

















# import os
# import re
# import html
# import json
# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import requests


# def clean_text(text: str) -> str:
#     text = html.unescape(text)                        # Decode HTML entities
#     text = re.sub(r'\s+', ' ', text)                  # Collapse multiple spaces/newlines
#     text = re.sub(r'@[\w\-]+', '', text)              # Remove mentions
#     text = re.sub(r':[\w\-]+:', '', text)             # Remove :emoji: syntax
#     text = re.sub(r'https?://\S+', '', text)          # Optionally remove links
#     text = re.sub(r'^\W+|\W+$', '', text)             # Trim non-word chars
#     return text.strip()

# def classify_tags_with_open_llm(query: str, possible_tags: list) -> list:
#     prompt = f"""
# You are a classifier that assigns tags to user queries.

# Possible tags: {', '.join(possible_tags)}

# Classify this question into the 2–3 most relevant tags:

# "{query}"

# Respond with only a comma-separated list of tags.
# """

#     response = requests.post(
#         "https://api.together.xyz/v1/chat/completions",
#         headers={
#             "Authorization": f"Bearer {TOGETHER_API_KEY}",
#             "Content-Type": "application/json"
#         },
#         json={
#             "model": MODEL_NAME,
#             "messages": [{"role": "user", "content": prompt}],
#             "temperature": 0.3,
#             "max_tokens": 100,
#         },
#     )

#     content = response.json()["choices"][0]["message"]["content"]
#     return [tag.strip().lower() for tag in content.split(",") if tag.strip()]


# def generate_answer_with_open_llm(query: str, context_posts: list) -> str:
#     context = "\n\n".join([
#         f"{post['user']['name']}: {post['text']}" for post in context_posts
#     ])

#     prompt = f"""
# You are an assistant for Jupiter.money.

# Using only the context below, answer the question accurately and helpfully.

# Question: "{query}"

# Context:
# {context}

# Answer:
# """

#     response = requests.post(
#         "https://api.together.xyz/v1/chat/completions",
#         headers={
#             "Authorization": f"Bearer {TOGETHER_API_KEY}",
#             "Content-Type": "application/json"
#         },
#         json={
#             "model": MODEL_NAME,
#             "messages": [{"role": "user", "content": prompt}],
#             "temperature": 0.5,
#             "max_tokens": 512,
#         },
#     )

#     return response.json()["choices"][0]["message"]["content"].strip()


# # Filenames
# RAW_JSON = "faq_data_raw.json"
# INDEX_FILE = "faq.index"
# QUESTIONS_FILE = "faq_questions.json"
# METADATA_FILE = "faq_metadata.json"

# # Load and deduplicate questions
# def load_and_prepare_data():
#     with open(RAW_JSON, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     questions = []
#     metadatas = []
#     seen = set()

#     for topic in data:
#         q_text = clean_text(topic.get("text", "").strip())
#         if not q_text or q_text in seen:
#             continue
#         seen.add(q_text)
#         cleaned_posts = []
#         for post in topic.get("posts", []):
#             cleaned_post = post.copy()
#             cleaned_post["text"] = clean_text(post.get("text", ""))
#             cleaned_posts.append(cleaned_post)
#         questions.append(q_text)
#         metadatas.append({
#             "tags": topic.get("tags", []),
#             "title": topic.get("title", ""),
#             "url": topic.get("url", ""),
#             "posts": topic.get("posts", []),
#         })

#     print(f"✅ Loaded {len(questions)} unique questions.")
#     return questions, metadatas

# # Build FAISS index from scratch
# def build_faiss_index(questions):
#     print("🔄 Building FAISS index...")
#     model = SentenceTransformer("all-MiniLM-L6-v2")
#     embeddings = model.encode(questions, convert_to_numpy=True, batch_size=32)
#     faiss.normalize_L2(embeddings)

#     dim = embeddings.shape[1]
#     index = faiss.IndexFlatIP(dim)
#     index.add(embeddings)

#     # Save index and data
#     faiss.write_index(index, INDEX_FILE)
#     with open(QUESTIONS_FILE, "w", encoding="utf-8") as f:
#         json.dump(questions, f, ensure_ascii=False, indent=2)
#     with open(METADATA_FILE, "w", encoding="utf-8") as f:
#         json.dump(metadatas, f, ensure_ascii=False, indent=2)

#     print(f"✅ Index built with {index.ntotal} entries and saved to disk.")
#     return index, model

# # Load index + data
# def load_index_and_data():
#     index = faiss.read_index(INDEX_FILE)
#     with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
#         questions = json.load(f)
#     with open(METADATA_FILE, "r", encoding="utf-8") as f:
#         metadatas = json.load(f)
#     model = SentenceTransformer("all-MiniLM-L6-v2")
#     return index, questions, metadatas, model

# # Search
# def retrieve_with_postfilter(query, predicted_tags, index, questions, metadatas, model, top_k=10):
#     q_emb = model.encode([query], convert_to_numpy=True)
#     faiss.normalize_L2(q_emb)
#     D, I = index.search(q_emb, top_k)
#     D, I = D[0], I[0]

#     results = []
#     for score, idx in zip(D, I):
#         meta = metadatas[idx]
#         if any(tag in meta["tags"] for tag in predicted_tags):
#             results.append({
#                 "question": questions[idx],
#                 "score": float(score),
#                 "tags": meta["tags"],
#                 "title": meta["title"],
#                 "url": meta["url"],
#                 "posts": meta["posts"],
#             })
#     return results

# # Main interactive block
# if __name__ == "__main__":
#     print("🔍 Jupiter FAQ Search with FAISS")
#     if not (os.path.exists(INDEX_FILE) and os.path.exists(QUESTIONS_FILE) and os.path.exists(METADATA_FILE)):
#         questions, metadatas = load_and_prepare_data()
#         index, model = build_faiss_index(questions)
#     else:
#         index, questions, metadatas, model = load_index_and_data()

#     # Interactive query loop
#     while True:
#         query = input("\n❓ Ask a question (or type 'exit'): ").strip()
#         if query.lower() == "exit":
#             break

#         tags_input = input("🏷️  Enter tags (comma-separated): ")
#         predicted_tags = [t.strip().lower() for t in tags_input.split(",") if t.strip()]
#         hits = retrieve_with_postfilter(query, predicted_tags, index, questions, metadatas, model, top_k=8)

#         if not hits:
#             print("⚠️  No matching questions found with those tags.")
#             continue

#         print(f"\n✅ Found {len(hits)} similar questions:\n")
#         for h in hits:
#             print(f"• Q: {h['question']}")
#             print(f"  ↪️  Score: {h['score']:.3f} | Tags: {h['tags']}")
#             print(f"  🔗 {h['url']}")
#             for post in h["posts"][1:3]:
#                 print(f"     👤 {post['user']['name']}: {post['text'][:100]}…")
#             print("-" * 60)
