# Phase 4: Embeddings + Vector Search

Notes to future me. What this project does in one sentence:

> It cuts a text file into chunks, turns each chunk into a vector with `all-MiniLM-L6-v2`, stores them in Chroma, and returns the chunks closest in meaning to a question.

**Important:** this does NOT answer questions. It only finds the best chunks. An LLM reads those chunks and writes the answer in Phase 5 (RAG).

---

## Install and run

```
pip install sentence-transformers chromadb langchain-text-splitters
python ingest_and_search.py
```

Put the text to search in `notes.txt`. Before changing `chunk_size`, delete the `chroma_db` folder (see "Traps" below).

---

# 1. Embeddings

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(["I love pasta", "Messi scored a goal"])
print(embeddings.shape)   # (2, 384)
```

- `SentenceTransformer("all-MiniLM-L6-v2")` loads a model that is already trained. I am not training anything. The first run downloads it (~90 MB).
- `model.encode(...)` does everything in one call: the text is split into tokens inside the model (I never do this myself), and the model returns **one vector per sentence**.
- **Embedding = vector.** They are the same thing. The flow is: sentence → tokens (internal) → model → one vector of 384 numbers.
- `(2, 384)` means 2 sentences, 384 numbers each. 384 is fixed by the model and does not depend on sentence length. Short sentences get padded to the same token length inside the model, but the output is always 384 numbers.
- Sentences with similar **meaning** end up with vectors that are close together, even with no shared words ("Messi scored" and "Ronaldo scored a goal" scored 0.525 and had no words in common).

## Measuring closeness: cosine similarity

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

- It measures the angle between two vectors. Close to 1 = same direction (similar meaning), close to 0 = unrelated.
- The number alone means little (0.36 is NOT "36% similar"). What matters is the **ranking**: is the right text above the wrong ones?

## Searching by hand (what Chroma does for me)

```python
query_embedding = model.encode("Who is struggling emotionally?")

scores = []
for i, sentence in enumerate(sentences):
    scores.append((cosine_similarity(query_embedding, embeddings[i]), sentence))

scores.sort(reverse=True)
```

- The query is embedded with the **same model** as the stored text. A different model would give vectors in a different space, so the comparison would be meaningless.
- If I change `sentences`, I must run `model.encode(sentences)` again. Otherwise the text and its vector no longer match.

---

# 2. Chroma (the vector database)

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("notes")
```

- `PersistentClient(path=...)` saves the database in a folder on disk, so it survives after the script ends. (`chromadb.Client()` is in-memory and is wiped every run.)
- `get_or_create_collection` opens the collection if it exists and creates it if it does not (`create_collection` crashes on the second run).
- Chroma never talks to the embedding model. **My script is the middle layer:** it embeds the text, then gives the vectors to Chroma.

```python
collection.upsert(
    ids=["0", "1"],
    documents=["I love pasta", "Messi scored a goal"],
    embeddings=embeddings.tolist(),
)
```

- Every item needs a unique `id`. The id lets me update, delete, fetch, or avoid storing the same item twice.
- `.tolist()` converts the numpy array into a plain Python list, which Chroma expects.

## add vs upsert

| | id is new | id already exists |
|---|---|---|
| `add` | inserted | **skipped** (old text stays) |
| `upsert` | inserted | **overwritten** with the new text |

- `collection.count()` counts ids, not how many times I called `add`. Adding the same 7 ids again still gives 7.
- `collection.get(ids=["0"])["documents"]` fetches one specific item, useful for checking what is stored.

## Querying

```python
results = collection.query(
    query_embeddings=[model.encode("my question").tolist()],
    n_results=3,
)
```

- `results["documents"][0]` is used because I can send several queries at once. The `[0]` takes the results for my one query.
- Chroma returns a **distance**, not a similarity. **Smaller = closer.** Chroma's default is squared L2 distance. For these vectors: `distance = 2 - 2 × cosine similarity`. So a similarity of 0.325 shows up as a distance of 1.35.
- The search **always returns n results**, even if none are relevant. It never says "nothing found". So the distance is how I tell a real match from filler.

---

# 3. Chunking

One vector cannot represent a whole document, so I cut it into chunks and embed each one. A search then returns the specific chunk that has the answer.

My first try cut every N characters. It split words in half ("nsupervised"), which is bad for the embedding and bad for the LLM that reads it later. A second try that packs whole sentences worked better but did not handle very long sentences. The standard tool handles all of this:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(text)
```

- It tries to split on the biggest natural boundary first (blank line), then newline, then space, and only cuts inside a word as a last resort.
- `chunk_overlap=50` makes neighbouring chunks share text, so a sentence cut at the border still appears whole in one of them.
- `chunk_size` counts characters by default.

---

# 4. The full script

```python
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("notes")

# 1. read and chunk
text = Path("notes.txt").read_text(encoding="utf-8")   # whole file as one string
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(text)

# 2. embed and store
collection.upsert(    # upsert: re-running updates instead of skipping
    ids=[f"notes-{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=model.encode(chunks).tolist(),
    metadatas=[{"source": "notes.txt", "chunk": i} for i in range(len(chunks))],
)   # metadata = which file and which chunk, so I can say where an answer came from

# 3. ask questions
questions = ["What are the first three steps in the ML workflow?"]

for q in questions:
    results = collection.query(
        query_embeddings=[model.encode(q).tolist()],
        n_results=3,
    )
    print("\nQ:", q)
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        print(f"  chunk {meta['chunk']}  dist={dist:.2f}  {doc[:60]!r}")
```

- `zip(...)` walks through the three result lists together, so each line shows chunk number, distance, and the first 60 characters of the text.

---

# 5. Experiment: chunk size

I tested the same questions on my ML course notes with three chunk sizes.

| chunk_size | what happened |
|---|---|
| **200** | Too small. The "ML Workflow Steps" list was split between step 6 and step 7. The tail chunk (`Step 7... Step 8...`) lost its header, so it matched weakly (distance 1.03 to 1.08) even for a question about the last steps. |
| **500** | Worked well. Each chunk was about one section, so the right chunk came first with good distances (0.5 to 0.8). |
| **1000** | Too big. The whole file became about 3 chunks, each mixing several topics. Best distances were around 1.0 to 1.1, as weak as the filler results. |

**Conclusion:** chunks too small lose their surroundings, chunks too big blur several topics into one vector. The best size depends on how the document is structured, so I test it with my own questions each time.

---

# 6. What I learned

- Search returns chunks. It does not understand "first two" or "last two", and it does not write answers. It only finds text that is about the same thing as the question. The LLM does the reading and answering in Phase 5.
- Retrieval only has to get the right chunk into the top 3, not necessarily first.
- Distance guide on my data: about 0.5 to 0.8 = good match, 1.0 and above = filler. In Phase 5 I will use a distance cutoff so weak matches never reach the LLM. The right cutoff must be found by looking at my own numbers.
- My test was easy: the notes were only about 8 chunks, so the right chunk often landed in the top 3 by luck. A bigger document will be a harder test.
- The embedding model is a replaceable part. Gemini's `gemini-embedding-001` can be used instead, but the free tier has low rate limits, and vectors from different models cannot be mixed in one collection.

# 7. Traps

- If I change `chunk_size`, a new run may produce fewer chunks, but the old extra chunks stay in the database. **Delete the `chroma_db` folder before each experiment.**
- Always re-run `model.encode(...)` after changing the text, so text and vectors match.
- Use the same embedding model for storing and for querying.